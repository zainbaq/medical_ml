#!/bin/bash
#
# Start All Medical ML Services
#
# This script starts all medical ML services that use the medical_ml_sdk framework:
#   1. Registry Service (port 9000) - Central service registry
#   2. Cardiovascular Disease Service (port 8000)
#   3. Breast Cancer Service (port 8001)
#   4. Alzheimers Service (port 8002)
#
# All services auto-register with the registry on startup using the SDK.
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

# Check and activate virtual environment
VENV_DIR="$PROJECT_ROOT/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}✗ Virtual environment not found at: $VENV_DIR${NC}"
    echo -e "${YELLOW}Please run './setup_env.sh' first to create the environment${NC}"
    exit 1
fi

echo -e "${CYAN}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Log directory
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Service configuration array
# Format: "service_name:directory:port:required"
declare -a SERVICES=(
    "Registry:registry/backend:9000:true"
    "Cardiovascular Disease:cardiovascular_disease:8000:true"
    "Breast Cancer:breast_cancer:8001:false"
    "Alzheimers:alzheimers:8002:false"
)

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}     Starting Medical ML Service Registry System            ${NC}"
echo -e "${BLUE}     (Powered by medical_ml_sdk)                            ${NC}"
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
echo -e "\n${YELLOW}[Step 1] Checking for port conflicts...${NC}"
PORTS_IN_USE=()

for service in "${SERVICES[@]}"; do
    IFS=':' read -r name dir port required <<< "$service"
    if check_port "$port"; then
        PORTS_IN_USE+=($port)
    fi
done

if [ ${#PORTS_IN_USE[@]} -gt 0 ]; then
    echo -e "${RED}   ✗ The following ports are already in use: ${PORTS_IN_USE[*]}${NC}"
    echo -e "${YELLOW}   Run './stop_all_services.sh' to stop existing services${NC}"
    exit 1
fi

echo -e "${GREEN}   ✓ All ports available${NC}"

# Start services
step=2
for service in "${SERVICES[@]}"; do
    IFS=':' read -r name dir port required <<< "$service"

    # Determine log and pid file names
    log_name=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    pid_file="$LOG_DIR/${log_name}.pid"
    log_file="$LOG_DIR/${log_name}.log"

    # Check if service directory exists
    if [ ! -d "$PROJECT_ROOT/$dir" ]; then
        if [ "$required" = "true" ]; then
            echo -e "\n${RED}[Step $step] Error: Required service '$name' not found at $dir${NC}"
            exit 1
        else
            echo -e "\n${YELLOW}[Step $step] Skipping optional service '$name' (not found)${NC}"
            step=$((step + 1))
            continue
        fi
    fi

    echo -e "\n${YELLOW}[Step $step] Starting $name Service (port $port)...${NC}"

    # Determine the correct uvicorn command based on service structure
    cd "$PROJECT_ROOT/$dir"
    if [ "$name" = "Registry" ]; then
        # Registry has backend/ in the path, run from backend directory
        nohup uvicorn main:app --host 0.0.0.0 --port $port > "$log_file" 2>&1 &
    elif [ -f "backend/main.py" ]; then
        # Service has backend module, run from parent directory
        cd "$PROJECT_ROOT/${dir%%/backend*}"
        nohup uvicorn backend.main:app --host 0.0.0.0 --port $port > "$log_file" 2>&1 &
    else
        echo -e "${RED}   ✗ Unable to determine service structure${NC}"
        if [ "$required" = "true" ]; then
            exit 1
        else
            step=$((step + 1))
            continue
        fi
    fi

    SERVICE_PID=$!
    echo $SERVICE_PID > "$pid_file"
    echo -e "${CYAN}   $name PID: $SERVICE_PID${NC}"
    cd "$PROJECT_ROOT"

    # Wait for service to be ready (ML services need more time to load models)
    timeout=60  # Give ML services enough time to load models (~30s) + startup
    if ! wait_for_service "http://localhost:$port/health" "$name Service" $timeout; then
        if [ "$required" = "true" ]; then
            echo -e "${RED}   Failed to start required service: $name${NC}"
            echo -e "${YELLOW}   Check logs: tail -f $log_file${NC}"
            exit 1
        else
            echo -e "${YELLOW}   ⚠ Optional service may not be ready (check logs)${NC}"
        fi
    fi

    step=$((step + 1))
done

# Wait for services to register with registry
echo -e "\n${CYAN}Waiting for services to register with registry...${NC}"
sleep 3

# Check registry for registered services
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                  Services Status                           ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

SERVICES_JSON=$(curl -s "http://localhost:9000/api/v1/services" 2>/dev/null || echo "[]")
SERVICE_COUNT=$(echo "$SERVICES_JSON" | jq '. | length' 2>/dev/null || echo "0")

echo -e "\n${GREEN}✓ Registry Service${NC}"
echo -e "  URL: ${CYAN}http://localhost:9000${NC}"
echo -e "  Docs: ${CYAN}http://localhost:9000/docs${NC}"
echo -e "  Logs: ${CYAN}$LOG_DIR/registry.log${NC}"

echo -e "\n${GREEN}✓ Registered Services: $SERVICE_COUNT${NC}"
if [ "$SERVICE_COUNT" -gt "0" ]; then
    echo "$SERVICES_JSON" | jq -r '.[] | "  • \(.service_name) - \(.base_url)"' 2>/dev/null || echo "  (Unable to parse service list)"
fi

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}     All Services Started Successfully!                     ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Display service URLs
echo -e "\n${CYAN}Service URLs:${NC}"
for service in "${SERVICES[@]}"; do
    IFS=':' read -r name dir port required <<< "$service"
    log_name=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    if [ -f "$LOG_DIR/${log_name}.pid" ]; then
        printf "  %-20s ${CYAN}http://localhost:%s${NC}\n" "$name:" "$port"
    fi
done

# Display API documentation URLs
echo -e "\n${CYAN}API Documentation:${NC}"
for service in "${SERVICES[@]}"; do
    IFS=':' read -r name dir port required <<< "$service"
    log_name=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    if [ -f "$LOG_DIR/${log_name}.pid" ]; then
        printf "  %-20s ${CYAN}http://localhost:%s/docs${NC}\n" "$name:" "$port"
    fi
done

echo -e "\n${CYAN}Logs Location:${NC}"
echo -e "  $LOG_DIR/"

echo -e "\n${CYAN}Useful Commands:${NC}"
echo -e "  Stop all services:        ${YELLOW}./stop_all_services.sh${NC}"
echo -e "  View registry logs:       ${YELLOW}tail -f $LOG_DIR/registry.log${NC}"
echo -e "  List registered services: ${YELLOW}curl http://localhost:9000/api/v1/services | jq${NC}"
echo -e "  Check health:             ${YELLOW}curl http://localhost:9000/api/v1/health/all | jq${NC}"

echo -e "\n${GREEN}Services are running in the background.${NC}"
echo -e "${YELLOW}Use './stop_all_services.sh' to stop all services.${NC}\n"
