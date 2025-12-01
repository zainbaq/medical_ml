#!/bin/bash
#
# Check Status of All Medical ML Services
#
# This script checks which services are running and their health status
#
# Usage:
#   ./check_services.sh
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

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}        Medical ML Service Registry - Status Check          ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to check service health
check_health() {
    local url=$1
    local response=$(curl -s -w "\n%{http_code}" "$url" 2>/dev/null)
    local body=$(echo "$response" | head -n -1)
    local status_code=$(echo "$response" | tail -n 1)

    if [ "$status_code" = "200" ]; then
        local health_status=$(echo "$body" | jq -r '.status' 2>/dev/null || echo "unknown")
        echo "$health_status"
        return 0
    else
        echo "unreachable"
        return 1
    fi
}

echo -e "\n${YELLOW}Port Status:${NC}\n"

# Check Registry (port 9000)
echo -n "  Registry (9000):        "
if check_port 9000; then
    HEALTH=$(check_health "http://localhost:9000/health")
    if [ "$HEALTH" = "healthy" ]; then
        echo -e "${GREEN}✓ Running (${HEALTH})${NC}"
        REGISTRY_RUNNING=true
    else
        echo -e "${YELLOW}⚠ Running but ${HEALTH}${NC}"
        REGISTRY_RUNNING=true
    fi
else
    echo -e "${RED}✗ Not running${NC}"
    REGISTRY_RUNNING=false
fi

# Check CVD Service (port 8000)
echo -n "  CVD Service (8000):     "
if check_port 8000; then
    HEALTH=$(check_health "http://localhost:8000/health")
    if [ "$HEALTH" = "healthy" ]; then
        echo -e "${GREEN}✓ Running (${HEALTH})${NC}"
    else
        echo -e "${YELLOW}⚠ Running but ${HEALTH}${NC}"
    fi
else
    echo -e "${RED}✗ Not running${NC}"
fi

# Check Breast Cancer Service (port 8001)
echo -n "  Breast Cancer (8001):   "
if check_port 8001; then
    HEALTH=$(check_health "http://localhost:8001/health")
    if [ "$HEALTH" = "healthy" ]; then
        echo -e "${GREEN}✓ Running (${HEALTH})${NC}"
    else
        echo -e "${YELLOW}⚠ Running but ${HEALTH}${NC}"
    fi
else
    echo -e "${YELLOW}○ Not running${NC}"
fi

# Check Alzheimers Service (port 8002)
echo -n "  Alzheimers (8002):      "
if check_port 8002; then
    HEALTH=$(check_health "http://localhost:8002/health")
    if [ "$HEALTH" = "healthy" ]; then
        echo -e "${GREEN}✓ Running (${HEALTH})${NC}"
    else
        echo -e "${YELLOW}⚠ Running but ${HEALTH}${NC}"
    fi
else
    echo -e "${YELLOW}○ Not running${NC}"
fi

# If registry is running, query for registered services
if [ "$REGISTRY_RUNNING" = true ]; then
    echo -e "\n${YELLOW}Registered Services:${NC}\n"

    SERVICES=$(curl -s "http://localhost:9000/api/v1/services" 2>/dev/null || echo "[]")
    SERVICE_COUNT=$(echo "$SERVICES" | jq '. | length' 2>/dev/null || echo "0")

    if [ "$SERVICE_COUNT" -gt "0" ]; then
        echo "$SERVICES" | jq -r '.[] | "  • \(.service_name)\n    ID: \(.service_id)\n    URL: \(.base_url)\n    Tags: \(.tags | join(", "))\n"' 2>/dev/null || echo "  (Unable to parse service list)"
    else
        echo -e "${YELLOW}  No services registered${NC}\n"
    fi

    # Check aggregate health
    echo -e "${YELLOW}Aggregate Health Check:${NC}\n"
    HEALTH_DATA=$(curl -s "http://localhost:9000/api/v1/health/all" 2>/dev/null)

    if [ ! -z "$HEALTH_DATA" ]; then
        REGISTRY_STATUS=$(echo "$HEALTH_DATA" | jq -r '.registry.status' 2>/dev/null)
        echo -e "  Registry: ${GREEN}$REGISTRY_STATUS${NC}"

        SERVICES_HEALTH=$(echo "$HEALTH_DATA" | jq -r '.services[] | "  \(.service_name): \(.status)"' 2>/dev/null)
        if [ ! -z "$SERVICES_HEALTH" ]; then
            echo "$SERVICES_HEALTH" | while read line; do
                if [[ $line == *"healthy"* ]]; then
                    echo -e "${GREEN}$line${NC}"
                else
                    echo -e "${YELLOW}$line${NC}"
                fi
            done
        fi
    fi
fi

# Check for PID files
if [ -d "$LOG_DIR" ] && [ "$(ls -A $LOG_DIR/*.pid 2>/dev/null)" ]; then
    echo -e "\n${YELLOW}Process IDs (from PID files):${NC}\n"

    for pid_file in "$LOG_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            SERVICE_NAME=$(basename "$pid_file" .pid)
            PID=$(cat "$pid_file")
            echo -n "  $SERVICE_NAME: "
            if kill -0 "$PID" 2>/dev/null; then
                echo -e "${GREEN}PID $PID (running)${NC}"
            else
                echo -e "${RED}PID $PID (not running - stale PID file)${NC}"
            fi
        fi
    done
fi

# Check for log files
if [ -d "$LOG_DIR" ] && [ "$(ls -A $LOG_DIR/*.log 2>/dev/null)" ]; then
    echo -e "\n${YELLOW}Log Files:${NC}\n"
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            SERVICE_NAME=$(basename "$log_file" .log)
            SIZE=$(du -h "$log_file" | cut -f1)
            LINES=$(wc -l < "$log_file")
            echo -e "  $SERVICE_NAME: ${CYAN}$SIZE, $LINES lines${NC}"
            echo -e "    ${YELLOW}tail -f $log_file${NC}"
        fi
    done
fi

echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Quick Commands:${NC}"
echo -e "  Start all:   ${YELLOW}./start_all_services.sh${NC}"
echo -e "  Stop all:    ${YELLOW}./stop_all_services.sh${NC}"
echo -e "  Run tests:   ${YELLOW}./run_tests.sh${NC}"
echo -e "  API docs:    ${YELLOW}open http://localhost:9000/docs${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
