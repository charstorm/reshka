#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FIX_MODE=false
if [[ "${1:-}" == "--fix" ]]; then
    FIX_MODE=true
fi

pass_count=0
fail_count=0

run_check() {
    local name="$1"
    shift
    echo -e "${BLUE}▶ $name${NC}"
    if "$@"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        ((pass_count++)) || true
    else
        echo -e "${RED}✗ $name failed${NC}"
        ((fail_count++)) || true
    fi
    echo
}

if ! command -v uv &>/dev/null; then
    echo -e "${RED}Error: 'uv' is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Running checks for pilab...${NC}"
echo

if $FIX_MODE; then
    run_check "Ruff format (fix)" uv run ruff format .
    run_check "Ruff check (fix)" uv run ruff check . --fix
else
    run_check "Ruff format (check)" uv run ruff format --check .
    run_check "Ruff check" uv run ruff check .
fi

run_check "Mypy (strict)" uv run mypy reshka.py

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Passed: $pass_count${NC}"
if ((fail_count > 0)); then
    echo -e "${RED}Failed: $fail_count${NC}"
    exit 1
fi

echo -e "${GREEN}All checks completed successfully.${NC}"
