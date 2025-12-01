#!/bin/bash
#
# Master Test Runner for Medical ML Service Registry
#
# Runs all test suites and generates a comprehensive report
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}   Medical ML Service Registry - Master Test Runner       ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Parse arguments
RUN_SMOKE=true
RUN_INTEGRATION=true
RUN_DEMO=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-only)
            RUN_INTEGRATION=false
            shift
            ;;
        --integration-only)
            RUN_SMOKE=false
            shift
            ;;
        --demo)
            RUN_DEMO=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --smoke-only       Run only smoke tests"
            echo "  --integration-only Run only integration tests"
            echo "  --demo            Run demo script after tests"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Track results
SMOKE_RESULT=0
INTEGRATION_RESULT=0
DEMO_RESULT=0

# Run Smoke Tests
if [ "$RUN_SMOKE" = true ]; then
    echo -e "\n${YELLOW}┌─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│          Running Smoke Tests                            │${NC}"
    echo -e "${YELLOW}└─────────────────────────────────────────────────────────┘${NC}\n"

    cd "$PROJECT_ROOT/tests"
    if bash smoke_test.sh; then
        SMOKE_RESULT=0
        echo -e "\n${GREEN}✓ Smoke tests PASSED${NC}"
    else
        SMOKE_RESULT=$?
        echo -e "\n${RED}✗ Smoke tests FAILED (exit code: $SMOKE_RESULT)${NC}"
    fi
fi

# Run Integration Tests
if [ "$RUN_INTEGRATION" = true ]; then
    echo -e "\n${YELLOW}┌─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│          Running Integration Tests                      │${NC}"
    echo -e "${YELLOW}└─────────────────────────────────────────────────────────┘${NC}\n"

    # Install test dependencies if needed
    if [ ! -f "$PROJECT_ROOT/tests/.deps_installed" ]; then
        echo -e "${BLUE}Installing test dependencies...${NC}"
        pip install -q -r "$PROJECT_ROOT/tests/requirements.txt"
        touch "$PROJECT_ROOT/tests/.deps_installed"
    fi

    cd "$PROJECT_ROOT/tests"
    if pytest test_integration.py -v --tb=short; then
        INTEGRATION_RESULT=0
        echo -e "\n${GREEN}✓ Integration tests PASSED${NC}"
    else
        INTEGRATION_RESULT=$?
        echo -e "\n${RED}✗ Integration tests FAILED (exit code: $INTEGRATION_RESULT)${NC}"
    fi
fi

# Run Demo
if [ "$RUN_DEMO" = true ]; then
    echo -e "\n${YELLOW}┌─────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│          Running Demo                                   │${NC}"
    echo -e "${YELLOW}└─────────────────────────────────────────────────────────┘${NC}\n"

    cd "$PROJECT_ROOT/tests"
    if python3 demo_unified_interface.py; then
        DEMO_RESULT=0
        echo -e "\n${GREEN}✓ Demo completed successfully${NC}"
    else
        DEMO_RESULT=$?
        echo -e "\n${RED}✗ Demo failed (exit code: $DEMO_RESULT)${NC}"
    fi
fi

# Final Report
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                    TEST SUMMARY                           ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ "$RUN_SMOKE" = true ]; then
    if [ $SMOKE_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Smoke Tests: PASSED${NC}"
    else
        echo -e "${RED}✗ Smoke Tests: FAILED${NC}"
    fi
fi

if [ "$RUN_INTEGRATION" = true ]; then
    if [ $INTEGRATION_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Integration Tests: PASSED${NC}"
    else
        echo -e "${RED}✗ Integration Tests: FAILED${NC}"
    fi
fi

if [ "$RUN_DEMO" = true ]; then
    if [ $DEMO_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Demo: PASSED${NC}"
    else
        echo -e "${RED}✗ Demo: FAILED${NC}"
    fi
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Exit with failure if any tests failed
if [ $SMOKE_RESULT -ne 0 ] || [ $INTEGRATION_RESULT -ne 0 ] || [ "$RUN_DEMO" = true -a $DEMO_RESULT -ne 0 ]; then
    echo -e "${RED}Some tests failed. Please check the output above.${NC}\n"
    exit 1
else
    echo -e "${GREEN}All tests passed successfully!${NC}\n"
    exit 0
fi
