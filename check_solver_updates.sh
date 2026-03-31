#!/bin/bash
# Check for solver source and pip updates that might affect the MCP knowledge base.
# Run periodically (e.g., before a test campaign) to catch API changes.

echo "=== Solver Freshness Check ==="
echo "Date: $(date)"
echo

check_source() {
    local name=$1 root=$2
    if [ -d "$root" ]; then
        echo "--- $name (source) ---"
        cd "$root"
        echo "  Branch: $(git branch --show-current 2>/dev/null || echo 'N/A')"
        echo "  Last commit: $(git log -1 --oneline 2>/dev/null || echo 'N/A')"
        DAYS=$(( ($(date +%s) - $(git log -1 --format=%ct 2>/dev/null || echo $(date +%s))) / 86400 ))
        echo "  Age: $DAYS days"
        if [ $DAYS -gt 14 ]; then
            echo "  WARNING: $name source is $DAYS days old. Consider: cd $root && git pull"
        fi
    else
        echo "--- $name (source) ---"
        echo "  Not configured (set ${name^^}_ROOT)"
    fi
}

check_pip() {
    local pkg=$1 name=$2
    VER=$(pip show $pkg 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
    if [ -n "$VER" ]; then
        echo "  pip: $pkg $VER"
    else
        echo "  pip: $pkg not installed"
    fi
}

# 4C
check_source "4C" "${FOURC_ROOT:-}"
if [ -n "${FOURC_BINARY:-}" ] && [ -f "$FOURC_BINARY" ]; then
    echo "  Binary: $FOURC_BINARY ($(stat -c '%y' "$FOURC_BINARY" | cut -d' ' -f1))"
fi
echo

# Kratos
check_source "Kratos" "${KRATOS_ROOT:-}"
check_pip "KratosMultiphysics" "Kratos"
check_pip "KratosDEMApplication" "Kratos DEM"
check_pip "KratosStructuralMechanicsApplication" "Kratos SMA"
echo

# deal.II
check_source "deal.II" "${DEALII_ROOT:-}"
DEALII_VER=$(dpkg -l 2>/dev/null | grep "libdeal.ii-dev" | awk '{print $3}' | head -1)
if [ -n "$DEALII_VER" ]; then
    echo "  System package: deal.II $DEALII_VER"
fi
echo

# FEniCSx
check_source "FEniCSx" "${FENICS_ROOT:-}"
check_pip "dolfinx" "FEniCSx"
echo

# NGSolve
check_source "NGSolve" "${NGSOLVE_ROOT:-}"
check_pip "ngsolve" "NGSolve"
echo

# scikit-fem
check_source "scikit-fem" "${SKFEM_ROOT:-}"
check_pip "scikit-fem" "scikit-fem"
echo

# DUNE-fem
check_source "DUNE-fem" "${DUNE_ROOT:-}"
check_pip "dune-fem" "DUNE-fem"
echo

echo "=== Summary ==="
echo "If any solver has been updated, review the changelog and update"
echo "the MCP knowledge base pitfalls if needed."
echo "Run: pytest tests/test_physics_coverage.py::TestSolverFreshness -v"
