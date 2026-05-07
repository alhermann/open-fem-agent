"""Stress tests for session_journal.py — designed to BREAK it."""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from core.session_journal import SessionJournal, JournalEvent, get_journal, reset_journal


def separator(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def test_1_none_empty_missing():
    """Record with None/empty/missing values for every field."""
    separator("TEST 1: None/empty/missing values")
    j = SessionJournal()
    results = []

    # None event_type
    r = j.record(event_type=None, tool_name="test")
    results.append(f"  None event_type -> {r!r} (events: {len(j.events)})")

    # None tool_name
    r = j.record(event_type="tool_call", tool_name=None)
    results.append(f"  None tool_name -> type={type(r).__name__} (events: {len(j.events)})")

    # Empty event_type (not in EVENT_TYPES)
    r = j.record(event_type="", tool_name="test")
    results.append(f"  Empty event_type -> {r!r} (events: {len(j.events)})")

    # Empty tool_name (should work fine)
    r = j.record(event_type="tool_call", tool_name="")
    results.append(f"  Empty tool_name -> type={type(r).__name__} (events: {len(j.events)})")

    # All optional fields as None
    r = j.record(
        event_type="tool_call", tool_name="test",
        solver=None, physics=None, details=None,
        error_message=None, notes=None,
    )
    results.append(f"  All None optional -> type={type(r).__name__} (events: {len(j.events)})")

    # details as non-dict types
    for bad_details in ["not a dict", [1, 2, 3], 42, True, object()]:
        r = j.record(event_type="tool_call", tool_name="test", details=bad_details)
        det = r.details if r else "N/A"
        results.append(f"  details={type(bad_details).__name__} -> details={det!r}")

    # Try to serialize
    try:
        data = j.to_dict()
        json.dumps(data, default=str)
        results.append("  Serialization: OK")
    except Exception as e:
        results.append(f"  Serialization: FAILED - {e}")

    for r in results:
        print(r)
    print(f"  TOTAL events recorded: {len(j.events)}")


def test_2_extremely_long_strings():
    """Record with extremely long strings (1MB error messages)."""
    separator("TEST 2: Extremely long strings (1MB)")
    j = SessionJournal()

    mb_string = "A" * (1024 * 1024)  # 1MB
    print(f"  Input string length: {len(mb_string)}")

    # 1MB error_message
    r = j.record(event_type="tool_error", tool_name="test", error_message=mb_string)
    print(f"  1MB error_message -> stored len: {len(r.error_message)}")
    print(f"  Truncated? {len(r.error_message) < len(mb_string)}")

    # 1MB notes
    r = j.record(event_type="tool_call", tool_name="test", notes=mb_string)
    print(f"  1MB notes -> stored len: {len(r.notes)}")

    # 1MB tool_name (NOT truncated by __post_init__)
    r = j.record(event_type="tool_call", tool_name=mb_string)
    print(f"  1MB tool_name -> stored len: {len(r.tool_name)}")
    print(f"  tool_name truncated? {len(r.tool_name) < len(mb_string)}")

    # 1MB solver
    r = j.record(event_type="tool_call", tool_name="test", solver=mb_string)
    print(f"  1MB solver -> stored len: {len(r.solver)}")
    print(f"  solver truncated? {len(r.solver) < len(mb_string)}")

    # 1MB physics
    r = j.record(event_type="tool_call", tool_name="test", physics=mb_string)
    print(f"  1MB physics -> stored len: {len(r.physics)}")

    # Huge details dict
    huge_details = {f"key_{i}": mb_string for i in range(10)}
    r = j.record(event_type="tool_call", tool_name="test", details=huge_details)
    total_detail_size = sum(len(str(v)) for v in r.details.values())
    print(f"  10x 1MB details -> total detail chars: {total_detail_size:,}")

    # Serialize it
    t0 = time.time()
    try:
        data = j.to_dict()
        s = json.dumps(data, default=str)
        elapsed = time.time() - t0
        print(f"  Serialization: OK ({len(s):,} bytes, {elapsed:.2f}s)")
    except Exception as e:
        print(f"  Serialization: FAILED - {e}")


def test_3_100k_rapid_events():
    """Record 100k events rapidly and check memory/save time."""
    separator("TEST 3: 100k rapid events")

    j = SessionJournal()
    t0 = time.time()
    for i in range(100_000):
        j.record(
            event_type="tool_call",
            tool_name=f"tool_{i % 100}",
            solver="fenics",
            physics="heat",
            details={"iteration": i},
        )
    record_time = time.time() - t0
    print(f"  Recording 100k events: {record_time:.2f}s")
    print(f"  Events stored: {len(j.events)}")

    # Memory estimate (rough)
    import sys as _sys
    event_size = _sys.getsizeof(j.events)
    print(f"  List object size: {event_size:,} bytes")

    # Save time
    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time.time()
        path = j.save(tmpdir)
        save_time = time.time() - t0
        file_size = path.stat().st_size
        print(f"  Save time: {save_time:.2f}s")
        print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")

        # Load time
        t0 = time.time()
        j2 = SessionJournal.load(path)
        load_time = time.time() - t0
        print(f"  Load time: {load_time:.2f}s")
        print(f"  Events after load: {len(j2.events)}")

    # to_dict time
    t0 = time.time()
    d = j.to_dict()
    dict_time = time.time() - t0
    print(f"  to_dict() time: {dict_time:.2f}s")


def test_4_readonly_directory():
    """Save to a read-only directory."""
    separator("TEST 4: Save to read-only directory")

    j = SessionJournal()
    j.record(event_type="tool_call", tool_name="test")

    with tempfile.TemporaryDirectory() as tmpdir:
        readonly = Path(tmpdir) / "readonly"
        readonly.mkdir()
        os.chmod(str(readonly), 0o444)
        try:
            path = j.save(readonly)
            print(f"  UNEXPECTED: save succeeded -> {path}")
        except PermissionError as e:
            print(f"  PermissionError (expected): {e}")
        except Exception as e:
            print(f"  Other error: {type(e).__name__}: {e}")
        finally:
            os.chmod(str(readonly), 0o755)  # restore for cleanup

    # Non-existent deep path
    deep = Path("/tmp/stress_test_journal/a/b/c/d/e/f")
    try:
        path = j.save(deep)
        print(f"  Deep path auto-created: {path}")
        # Cleanup
        import shutil
        shutil.rmtree("/tmp/stress_test_journal")
    except Exception as e:
        print(f"  Deep path error: {type(e).__name__}: {e}")


def test_5_disk_full():
    """Save when disk is 'full' (write to /dev/full)."""
    separator("TEST 5: Disk full (/dev/full)")

    j = SessionJournal()
    j.record(event_type="tool_call", tool_name="test")

    if os.path.exists("/dev/full"):
        try:
            # /dev/full will fail on write
            path = j.save("/dev/full")
            print(f"  UNEXPECTED: save to /dev/full succeeded -> {path}")
        except OSError as e:
            print(f"  OSError (expected): {e}")
        except Exception as e:
            print(f"  Other error: {type(e).__name__}: {e}")
    else:
        print("  /dev/full not available, skipping")

    # Also try writing to a path where the file is read-only
    with tempfile.TemporaryDirectory() as tmpdir:
        fakefile = Path(tmpdir) / f"session_{j.session_id}.json"
        fakefile.write_text("existing")
        os.chmod(str(fakefile), 0o444)
        try:
            path = j.save(tmpdir)
            print(f"  Write over read-only file: SUCCEEDED (unexpected?) -> {path}")
        except PermissionError as e:
            print(f"  Write over read-only file: PermissionError -> {e}")
        except Exception as e:
            print(f"  Write over read-only file: {type(e).__name__}: {e}")
        finally:
            os.chmod(str(fakefile), 0o755)


def test_6_load_malformed_json():
    """Load from a JSON file with extra/missing/wrong-type fields."""
    separator("TEST 6: Load malformed JSON files")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extra fields
        data = {
            "session_id": "test123",
            "started_at": 1000.0,
            "extra_field": "should be ignored",
            "events": [
                {
                    "event_type": "tool_call",
                    "tool_name": "test",
                    "extra": "ignored?",
                }
            ],
        }
        p = Path(tmpdir) / "extra_fields.json"
        p.write_text(json.dumps(data))
        try:
            j = SessionJournal.load(p)
            print(f"  Extra fields: OK (events: {len(j.events)})")
        except Exception as e:
            print(f"  Extra fields: FAILED - {type(e).__name__}: {e}")

        # Missing session_id
        data2 = {"started_at": 1000.0, "events": []}
        p2 = Path(tmpdir) / "missing_session_id.json"
        p2.write_text(json.dumps(data2))
        try:
            j = SessionJournal.load(p2)
            print(f"  Missing session_id: FAILED silently (id={j.session_id!r})")
        except KeyError as e:
            print(f"  Missing session_id: KeyError (expected) -> {e}")
        except Exception as e:
            print(f"  Missing session_id: {type(e).__name__}: {e}")

        # Missing started_at
        data3 = {"session_id": "test", "events": []}
        p3 = Path(tmpdir) / "missing_started_at.json"
        p3.write_text(json.dumps(data3))
        try:
            j = SessionJournal.load(p3)
            print(f"  Missing started_at: FAILED silently (started_at={j.started_at!r})")
        except KeyError as e:
            print(f"  Missing started_at: KeyError (expected) -> {e}")
        except Exception as e:
            print(f"  Missing started_at: {type(e).__name__}: {e}")

        # Wrong type for session_id (int instead of str)
        data4 = {"session_id": 999, "started_at": "not_a_float", "events": []}
        p4 = Path(tmpdir) / "wrong_types.json"
        p4.write_text(json.dumps(data4))
        try:
            j = SessionJournal.load(p4)
            print(f"  Wrong types: loaded (id={j.session_id!r}, started_at={j.started_at!r})")
            # Try serializing
            d = j.to_dict()
            print(f"  Wrong types serialization: duration={d['duration_seconds']}")
        except Exception as e:
            print(f"  Wrong types: {type(e).__name__}: {e}")

        # Events with wrong-type event_type
        data5 = {
            "session_id": "test",
            "started_at": 1000.0,
            "events": [
                {"event_type": 123, "tool_name": "test"},
            ],
        }
        p5 = Path(tmpdir) / "wrong_event_type.json"
        p5.write_text(json.dumps(data5))
        try:
            j = SessionJournal.load(p5)
            print(f"  Wrong event_type type (int): loaded (events: {len(j.events)})")
        except Exception as e:
            print(f"  Wrong event_type type: {type(e).__name__}: {e}")

        # Events missing tool_name
        data6 = {
            "session_id": "test",
            "started_at": 1000.0,
            "events": [
                {"event_type": "tool_call"},
            ],
        }
        p6 = Path(tmpdir) / "missing_tool_name.json"
        p6.write_text(json.dumps(data6))
        try:
            j = SessionJournal.load(p6)
            print(f"  Missing tool_name in event: loaded (events: {len(j.events)})")
        except KeyError as e:
            print(f"  Missing tool_name in event: KeyError -> {e}")
        except Exception as e:
            print(f"  Missing tool_name: {type(e).__name__}: {e}")

        # Completely invalid JSON
        p7 = Path(tmpdir) / "invalid.json"
        p7.write_text("{not valid json!!")
        try:
            j = SessionJournal.load(p7)
            print(f"  Invalid JSON: loaded somehow??")
        except json.JSONDecodeError as e:
            print(f"  Invalid JSON: JSONDecodeError (expected)")
        except Exception as e:
            print(f"  Invalid JSON: {type(e).__name__}: {e}")

        # events is not a list
        data8 = {"session_id": "test", "started_at": 1000.0, "events": "not a list"}
        p8 = Path(tmpdir) / "events_not_list.json"
        p8.write_text(json.dumps(data8))
        try:
            j = SessionJournal.load(p8)
            print(f"  events as string: loaded (events: {len(j.events)})")
        except TypeError as e:
            print(f"  events as string: TypeError -> {e}")
        except Exception as e:
            print(f"  events as string: {type(e).__name__}: {e}")


def test_7_load_empty_file():
    """Load from an empty file."""
    separator("TEST 7: Load empty file")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Completely empty file
        p = Path(tmpdir) / "empty.json"
        p.write_text("")
        try:
            j = SessionJournal.load(p)
            print(f"  Empty file: loaded somehow (events: {len(j.events)})")
        except json.JSONDecodeError as e:
            print(f"  Empty file: JSONDecodeError (expected)")
        except Exception as e:
            print(f"  Empty file: {type(e).__name__}: {e}")

        # File with just whitespace
        p2 = Path(tmpdir) / "whitespace.json"
        p2.write_text("   \n\n  \t  ")
        try:
            j = SessionJournal.load(p2)
            print(f"  Whitespace file: loaded somehow")
        except json.JSONDecodeError as e:
            print(f"  Whitespace file: JSONDecodeError (expected)")
        except Exception as e:
            print(f"  Whitespace file: {type(e).__name__}: {e}")

        # File with just null
        p3 = Path(tmpdir) / "null.json"
        p3.write_text("null")
        try:
            j = SessionJournal.load(p3)
            print(f"  null file: loaded somehow")
        except (TypeError, AttributeError) as e:
            print(f"  null file: {type(e).__name__} -> {e}")
        except Exception as e:
            print(f"  null file: {type(e).__name__}: {e}")

        # File with just []
        p4 = Path(tmpdir) / "array.json"
        p4.write_text("[]")
        try:
            j = SessionJournal.load(p4)
            print(f"  [] file: loaded somehow")
        except (TypeError, KeyError) as e:
            print(f"  [] file: {type(e).__name__} -> {e}")
        except Exception as e:
            print(f"  [] file: {type(e).__name__}: {e}")


def test_8_concurrent_threads():
    """Concurrent recording from multiple threads."""
    separator("TEST 8: Concurrent multi-threaded recording")

    j = SessionJournal()
    errors = []
    n_threads = 20
    n_per_thread = 5000

    def worker(thread_id):
        try:
            for i in range(n_per_thread):
                j.record(
                    event_type="tool_call",
                    tool_name=f"thread_{thread_id}_tool_{i}",
                    solver=f"solver_{thread_id}",
                    details={"thread": thread_id, "iter": i},
                )
        except Exception as e:
            errors.append((thread_id, e))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0

    expected = n_threads * n_per_thread
    actual = len(j.events)
    print(f"  Threads: {n_threads}, events/thread: {n_per_thread}")
    print(f"  Expected: {expected}, Actual: {actual}")
    print(f"  Lost events: {expected - actual} ({100*(expected-actual)/expected:.1f}%)")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Solvers tracked: {len(j._solvers_used)}")

    if errors:
        for tid, e in errors[:5]:
            print(f"    Thread {tid}: {type(e).__name__}: {e}")

    # Also test singleton concurrent access
    reset_journal()
    j_list = []

    def singleton_worker():
        j_list.append(id(get_journal()))

    threads2 = [threading.Thread(target=singleton_worker) for _ in range(100)]
    for t in threads2:
        t.start()
    for t in threads2:
        t.join()
    unique_ids = set(j_list)
    print(f"  Singleton IDs from 100 threads: {len(unique_ids)} unique (should be 1)")
    if len(unique_ids) > 1:
        print(f"  BUG: Race condition in get_journal() singleton!")


def test_9_record_after_reset():
    """Record after reset."""
    separator("TEST 9: Record after reset")

    j = SessionJournal()
    j.record(event_type="tool_call", tool_name="before_reset", solver="fenics")
    print(f"  Before reset: events={len(j.events)}, solvers={j.solvers_used}")

    j.reset()
    print(f"  After reset: events={len(j.events)}, solvers={j.solvers_used}")

    j.record(event_type="tool_success", tool_name="after_reset", solver="dealii")
    print(f"  After new record: events={len(j.events)}, solvers={j.solvers_used}")

    # Double reset
    j.reset()
    j.reset()
    print(f"  After double reset: events={len(j.events)}")

    # Record, save, reset, save again
    j.record(event_type="tool_call", tool_name="test")
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = j.save(tmpdir)
        j.reset()
        p2 = j.save(tmpdir)
        d1 = json.loads(p1.read_text())
        d2 = json.loads(p2.read_text())
        print(f"  Save before reset: {d1['event_count']} events")
        print(f"  Save after reset: {d2['event_count']} events")

    # Global reset
    gj = reset_journal()
    gj.record(event_type="tool_call", tool_name="test")
    gj2 = reset_journal()
    print(f"  Global reset: old events={len(gj.events)}, new events={len(gj2.events)}")
    print(f"  Same object? {gj is gj2}")


def test_10_unicode_emoji_binary():
    """Non-ASCII/emoji/binary data in all string fields."""
    separator("TEST 10: Non-ASCII/emoji/binary data")

    j = SessionJournal()

    # Unicode
    r = j.record(
        event_type="tool_call",
        tool_name="Werkzeug_mit_Umlauten_aou",
        solver="Loser_fur_Warme",
        physics="Warmeleitung",
        error_message="Fehler: Konvergenz nicht erreicht",
        notes="Notiz mit Sonderzeichen",
    )
    print(f"  German umlauts: OK (tool={r.tool_name!r})")

    # Chinese/Japanese/Korean
    r = j.record(
        event_type="tool_call",
        tool_name="gong_ju",
        solver="qiu_jie_qi",
        notes="zhe_shi_yi_ge_ce_shi",
    )
    print(f"  CJK: OK (tool={r.tool_name!r})")

    # Emojis
    r = j.record(
        event_type="tool_error",
        tool_name="fire_tool_rocket",
        error_message="boom_explosion_skull",
        notes="check_mark_sparkles",
    )
    print(f"  Emojis: OK (tool={r.tool_name!r})")

    # Null bytes and control characters
    r = j.record(
        event_type="tool_call",
        tool_name="tool\x00with\x00nulls",
        solver="solver\ttab\nnewline",
        error_message="msg\r\n\x1b[31mred\x1b[0m",
    )
    print(f"  Control chars: OK (tool={r.tool_name!r})")

    # Binary-ish data
    binary_str = "".join(chr(i) for i in range(256))
    r = j.record(
        event_type="tool_call",
        tool_name=binary_str,
        solver=binary_str,
        notes=binary_str[:500],
    )
    print(f"  All byte values as chars: OK (tool_name len={len(r.tool_name)})")

    # RTL text (Arabic/Hebrew)
    r = j.record(
        event_type="tool_call",
        tool_name="mrhba",
        solver="hl",
    )
    print(f"  RTL text: OK")

    # Extremely long unicode
    r = j.record(
        event_type="tool_call",
        tool_name="e" * 10000,  # 10k emoji chars
    )
    print(f"  10k repeated chars tool_name: len={len(r.tool_name)}")

    # Serialize everything
    try:
        data = j.to_dict()
        s = json.dumps(data, default=str, ensure_ascii=False)
        print(f"  Full serialization: OK ({len(s):,} bytes)")
    except Exception as e:
        print(f"  Full serialization: FAILED - {type(e).__name__}: {e}")

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            path = j.save(tmpdir)
            j2 = SessionJournal.load(path)
            print(f"  Save+reload: OK (events: {len(j2.events)})")
        except Exception as e:
            print(f"  Save+reload: FAILED - {type(e).__name__}: {e}")


def test_bonus_event_to_dict_drops_falsy():
    """Test that _event_to_dict drops ALL falsy values including 0 and False."""
    separator("BONUS: _event_to_dict drops falsy values (potential bug)")
    from core.session_journal import _event_to_dict

    evt = JournalEvent(
        timestamp=0.0,  # falsy!
        event_type="tool_call",
        tool_name="test",
        details={"count": 0, "flag": False, "empty": "", "none": None},
    )
    d = _event_to_dict(evt)
    print(f"  timestamp=0.0 in output? {'timestamp' in d} (value: {d.get('timestamp', 'MISSING')})")
    print(f"  Full dict: {d}")
    print(f"  BUG: timestamp=0.0 is falsy and gets DROPPED from serialization!")

    # Also: details with falsy values
    evt2 = JournalEvent(
        timestamp=1000.0,
        event_type="tool_call",
        tool_name="test",
        details={},  # empty dict is falsy
    )
    d2 = _event_to_dict(evt2)
    print(f"  empty details in output? {'details' in d2}")


def test_bonus_save_tmp_rename_race():
    """What if .tmp file already exists from a crashed save?"""
    separator("BONUS: .tmp file left from crashed save")

    j = SessionJournal()
    j.record(event_type="tool_call", tool_name="test")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-create the .tmp file with garbage
        tmp_path = Path(tmpdir) / f"session_{j.session_id}.tmp"
        tmp_path.write_text("GARBAGE LEFT FROM CRASH")

        try:
            path = j.save(tmpdir)
            data = json.loads(path.read_text())
            print(f"  Save over existing .tmp: OK (events: {data['event_count']})")
            # Check if .tmp was cleaned up
            print(f"  .tmp file still exists? {tmp_path.exists()}")
        except Exception as e:
            print(f"  Save over existing .tmp: FAILED - {type(e).__name__}: {e}")


if __name__ == "__main__":
    print("SESSION JOURNAL STRESS TEST")
    print("=" * 60)

    tests = [
        test_1_none_empty_missing,
        test_2_extremely_long_strings,
        test_3_100k_rapid_events,
        test_4_readonly_directory,
        test_5_disk_full,
        test_6_load_malformed_json,
        test_7_load_empty_file,
        test_8_concurrent_threads,
        test_9_record_after_reset,
        test_10_unicode_emoji_binary,
        test_bonus_event_to_dict_drops_falsy,
        test_bonus_save_tmp_rename_race,
    ]

    passed = 0
    failed = 0
    crashed = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            crashed.append((test_fn.__name__, e))
            print(f"\n  *** CRASHED: {type(e).__name__}: {e} ***")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {passed} completed, {failed} crashed")
    if crashed:
        print(f"  Crashed tests:")
        for name, e in crashed:
            print(f"    - {name}: {type(e).__name__}: {e}")
    print(f"{'=' * 60}")
