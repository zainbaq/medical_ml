#!/bin/bash
#
# Start All Medical ML Services
#
# This script starts:
#   1. Registry Service (port 9000)
#   2. Cardiovascular Disease Service (port 8000)
#   3. Breast Cancer Service (port 8001)
#   4. Alzheimers Service (port 8002)
#
# Services will auto-register with the registry on startup.
#
# Usage:
#   ./start_all_services.sh
#
# To stop all services:
#   ./stop_all_services.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Log file
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}     Starting Medical ML Service Registry System            ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if ports are already in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_wait=${3:-30}

    echo -e "${CYAN}   Waiting for $name to be ready...${NC}"
    local count=0
    while [ $count -lt $max_wait ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}   ✓ $name is ready!${NC}"
            return 0
        fi
        sleep 1
        count=$((count + 1))
        echo -n "."
    done
    echo -e "\n${RED}   ✗ $name failed to start within ${max_wait}s${NC}"
    return 1
}

# Check for port conflicts
echo -e "\n${YELLOW}[1/5] Checking for port conflicts...${NC}"
PORTS_IN_USE=()

if check_port 9000; then PORTS_IN_USE+=(9000); fi
if check_port 8000; then PORTS_IN_USE+=(8000); fi
if check_port 8001; then PORTS_IN_USE+=(8001); fi
if check_port 8002; then PORTS_IN_USE+=(8002); fi

if [ ${#PORTS_IN_USE[@]} -gt 0 ]; then
    echo -e "${RED}   ✗ The following ports are already in use: ${PORTS_IN_USE[*]}${NC}"
    echo -e "${YELLOW}   Run './stop_all_services.sh' to stop existing services${NC}"
    echo -e "${YELLOW}   Or manually kill processes: lsof -ti:${PORTS_IN_USE[*]} | xargs kill -9${NC}"
    exit 1
fi

echo -e "${GREEN}   ✓ All ports available${NC}"

# Start Registry Service
echo -e "\n${YELLOW}[2/5] Starting Registry Service (port 9000)...${NC}"
cd "$PROJECT_ROOT/registry/backend"
nohup uvicorn main:app --host 0.0.0.0 --port 9000 > "$LOG_DIR/registry.log" 2>&1 &
REGISTRY_PID=$!
echo $REGISTRY_PID > "$LOG_DIR/registry.pid"
echo -e "${CYAN}   Registry PID: $REGISTRY_PID${NC}"
cd "$PROJECT_ROOT"

if ! wait_for_service "http://localhost:9000/health" "Registry" 30; then
    echo -e "${RED}   Failed to start registry service${NC}"
    exit 1
fi

# Start Cardiovascular Disease Service
echo -e "\n${YELLOW}[3/5] Starting Cardiovascular Disease Service (port 8000)...${NC}"
cd "$PROJECT_ROOT/cardiovascular_disease"
nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 > "$LOG_DIR/cvd.log" 2>&1 &
CVD_PID=$!
echo $CVD_PID > "$LOG_DIR/cvd.pid"
echo -e "${CYAN}   CVD Service PID: $CVD_PID${NC}"
cd "$PROJECT_ROOT"

if ! wait_for_service "http://localhost:8000/health" "CVD Service" 30; then
    echo -e "${RED}   Failed to start CVD service${NC}"
    exit 1
fi

# Start Breast Cancer Service
echo -e "\n${YELLOW}[4/5] Starting Breast Cancer Service (port 8001)...${NC}"
if [ -d "$PROJECT_ROOT/breast_cancer/backend" ]; then
    cd "$PROJECT_ROOT/breast_cancer"
    nohup uvicorn backend.main:app --host 0.0.0.0 --port 8001 > "$LOG_DIR/breast_cancer.log" 2>&1 &
    BC_PID=$!
    echo $BC_PID > "$LOG_DIR/breast_cancer.pid"
    echo -e "${CYAN}   Breast Cancer Service PID: $BC_PID${NC}"
    cd "$PROJECT_ROOT"

    if ! wait_for_service "http://localhost:8001/health" "Breast Cancer Service" 30; then
        echo -e "${YELLOW}   ⚠ Breast Cancer service may not be ready (check logs)${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠ Breast Cancer service not found (skipping)${NC}"
fi

# Start Alzheimers Service
echo -e "\n${YELLOW}[5/5] Starting Alzheimers Service (port 8002)...${NC}"
if [ -d "$PROJECT_ROOT/alzheimers/backend" ]; then
    cd "$PROJECT_ROOT/alzheimers"
    nohup uvicorn backend.main:app --host 0.0.0.0 --port 8002 > "$LOG_DIR/alzheimers.log" 2>&1 &
    ALZ_PID=$!
    echo $ALZ_PID > "$LOG_DIR/alzheimers.pid"
    echo -e "${CYAN}   Alzheimers Service PID: $ALZ_PID${NC}"
    cd "$PROJECT_ROOT"

    if ! wait_for_service "http://localhost:8002/health" "Alzheimers Service" 30; then
        echo -e "${YELLOW}   ⚠ Alzheimers service may not be ready (check logs)${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠ Alzheimers service not found (skipping)${NC}"
fi

# Wait for services to register
echo -e "\n${CYAN}Waiting for services to register with registry...${NC}"
sleep 3

# Check registry for registered services
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                  Services Status                           ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

SERVICES=$(curl -s "http://localhost:9000/api/v1/services" 2>/dev/null || echo "[]")
SERVICE_COUNT=$(echo "$SERVICES" | jq '. | length' 2>/dev/null || echo "0")

echo -e "\n${GREEN}✓ Registry Service${NC}"
echo -e "  URL: ${CYAN}http://localhost:9000${NC}"
echo -e "  Docs: ${CYAN}http://localhost:9000/docs${NC}"
echo -e "  Logs: ${CYAN}$LOG_DIR/registry.log${NC}"

echo -e "\n${GREEN}✓ Registered Services: $SERVICE_COUNT${NC}"
if [ "$SERVICE_COUNT" -gt "0" ]; then
    echo "$SERVICES" | jq -r '.[] | "  • \(.service_name) - \(.base_url)"' 2>/dev/null || echo "  (Unable to parse service list)"
fi

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}     All Services Started Successfully!                     ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "\n${CYAN}Service URLs:${NC}"
echo -e "  Registry:        ${CYAN}http://localhost:9000${NC}"
echo -e "  CVD Service:     ${CYAN}http://localhost:8000${NC}"
[ -f "$LOG_DIR/breast_cancer.pid" ] && echo -e "  Breast Cancer:   ${CYAN}http://localhost:8001${NC}"
[ -f "$LOG_DIR/alzheimers.pid" ] && echo -e "  Alzheimers:      ${CYAN}http://localhost:8002${NC}"

echo -e "\n${CYAN}API Documentation:${NC}"
echo -e "  Registry:        ${CYAN}http://localhost:9000/docs${NC}"
echo -e "  CVD Service:     ${CYAN}http://localhost:8000/docs${NC}"
[ -f "$LOG_DIR/breast_cancer.pid" ] && echo -e "  Breast Cancer:   ${CYAN}http://localhost:8001/docs${NC}"
[ -f "$LOG_DIR/alzheimers.pid" ] && echo -e "  Alzheimers:      ${CYAN}http://localhost:8002/docs${NC}"

echo -e "\n${CYAN}Logs Location:${NC}"
echo -e "  $LOG_DIR/"

echo -e "\n${CYAN}Useful Commands:${NC}"
echo -e "  Stop all services:        ${YELLOW}./stop_all_services.sh${NC}"
echo -e "  View registry logs:       ${YELLOW}tail -f $LOG_DIR/registry.log${NC}"
echo -e "  View CVD logs:            ${YELLOW}tail -f $LOG_DIR/cvd.log${NC}"
echo -e "  List registered services: ${YELLOW}curl http://localhost:9000/api/v1/services | jq${NC}"
echo -e "  Check health:             ${YELLOW}curl http://localhost:9000/api/v1/health/all | jq${NC}"

echo -e "\n${GREEN}Services are running in the background.${NC}"
echo -e "${YELLOW}Use './stop_all_services.sh' to stop all services.${NC}\n"
