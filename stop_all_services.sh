#!/bin/bash
#
# Stop All Medical ML Services
#
# This script stops all running services started by start_all_services.sh
#
# Usage:
#   ./stop_all_services.sh
#

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

# Service configuration (must match start_all_services.sh)
# Format: "service_name:directory:port:required"
declare -a SERVICES=(
    "Registry:registry/backend:9000:true"
    "Cardiovascular Disease:cardiovascular_disease:8000:true"
    "Breast Cancer:breast_cancer:8001:false"
    "Alzheimers:alzheimers:8002:false"
)

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}     Stopping Medical ML Service Registry System            ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

STOPPED_COUNT=0

# Function to stop a service by PID file
stop_service() {
    local name=$1
    local pid_file=$2

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Stopping $name (PID: $pid)...${NC}"
            kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${RED}   ✗ Failed to stop $name${NC}"
            else
                echo -e "${GREEN}   ✓ Stopped $name${NC}"
                STOPPED_COUNT=$((STOPPED_COUNT + 1))
            fi
        else
            echo -e "${YELLOW}   • $name was not running (PID $pid)${NC}"
        fi
        rm -f "$pid_file"
    fi
}

# Stop services using PID files (in reverse order)
if [ -d "$LOG_DIR" ]; then
    echo -e "\n${YELLOW}Stopping services using PID files...${NC}\n"

    # Stop in reverse order (optional services first, then required, registry last)
    for ((idx=${#SERVICES[@]}-1 ; idx>=0 ; idx--)); do
        service="${SERVICES[idx]}"
        IFS=':' read -r name dir port required <<< "$service"
        log_name=$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        pid_file="$LOG_DIR/${log_name}.pid"

        stop_service "$name Service" "$pid_file"
    done
else
    echo -e "\n${YELLOW}No logs directory found${NC}"
fi

# Kill any remaining processes on our ports
echo -e "\n${YELLOW}Checking for remaining processes on ports...${NC}\n"

KILLED_COUNT=0

# Extract ports from service configuration
PORTS=()
for service in "${SERVICES[@]}"; do
    IFS=':' read -r name dir port required <<< "$service"
    PORTS+=($port)
done

for port in "${PORTS[@]}"; do
    PIDS=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$PIDS" ]; then
        echo -e "${YELLOW}Found process(es) on port $port: $PIDS${NC}"
        echo "$PIDS" | xargs kill -9 2>/dev/null
        echo -e "${GREEN}   ✓ Killed process(es) on port $port${NC}"
        KILLED_COUNT=$((KILLED_COUNT + 1))
    fi
done

if [ $KILLED_COUNT -eq 0 ]; then
    echo -e "${GREEN}   No remaining processes found${NC}"
fi

# Final summary
echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ $STOPPED_COUNT -gt 0 ] || [ $KILLED_COUNT -gt 0 ]; then
    echo -e "${GREEN}     All Services Stopped Successfully!                     ${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "\n${GREEN}✓ Stopped $STOPPED_COUNT service(s) using PID files${NC}"
    [ $KILLED_COUNT -gt 0 ] && echo -e "${GREEN}✓ Cleaned up processes on $KILLED_COUNT port(s)${NC}"
else
    echo -e "${YELLOW}     No Services Were Running                              ${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
fi

# Clean up log directory
if [ -d "$LOG_DIR" ]; then
    # Remove PID files
    rm -f "$LOG_DIR"/*.pid 2>/dev/null

    # Check if we should preserve logs
    if [ -z "$(ls -A $LOG_DIR)" ]; then
        echo -e "\n${YELLOW}Log directory is empty. Removing...${NC}"
        rmdir "$LOG_DIR"
    else
        echo -e "\n${YELLOW}Logs preserved in: $LOG_DIR/${NC}"
        echo -e "${YELLOW}To view logs: ls -la $LOG_DIR/${NC}"
    fi
fi

echo -e "\n${CYAN}To start services again, run:${NC}"
echo -e "  ${YELLOW}./start_all_services.sh${NC}\n"
