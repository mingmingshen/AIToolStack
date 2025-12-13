#!/bin/bash
# NE301 project initialization script
# Automatically check and clone NE301 project to the host directory on container start (if empty)

set -e

NE301_HOST_DIR="/workspace/ne301"  # Host directory path mounted by Docker Compose

# Check if running inside a Docker container
if [ -f "/.dockerenv" ]; then
    # Inside container: check host directory (via mount)
    if [ -d "$NE301_HOST_DIR" ]; then
        # Host directory exists (via docker-compose volume mount)
        echo "[NE301 Init] Detected host directory mount: $NE301_HOST_DIR"
        
        # Check if directory is empty or missing key files
        if [ ! "$(ls -A $NE301_HOST_DIR 2>/dev/null)" ] || [ ! -d "$NE301_HOST_DIR/Model" ]; then
            echo "[NE301 Init] Host directory is empty or incomplete, cloning from GitHub..."
            # If directory not empty but missing files, clean up first
            if [ "$(ls -A $NE301_HOST_DIR 2>/dev/null)" ]; then
                rm -rf "$NE301_HOST_DIR"/*
            fi
            git clone https://github.com/camthink-ai/ne301.git "$NE301_HOST_DIR"
            echo "[NE301 Init] Clone completed"
        else
            echo "[NE301 Init] Complete NE301 project found in host directory, skipping clone"
        fi
    else
        echo "[NE301 Init] Warning: Host directory mount not detected ($NE301_HOST_DIR)"
        echo "[NE301 Init] Please make sure ./ne301:/workspace/ne301 is configured in docker-compose.yml"
        echo "[NE301 Init] Falling back to container-internal directory..."
        
        # Fallback: use container internal directory
        NE301_CONTAINER_DIR="/app/ne301"
        if [ ! -d "$NE301_CONTAINER_DIR" ]; then
            echo "[NE301 Init] Cloning to container internal directory..."
            git clone https://github.com/camthink-ai/ne301.git "$NE301_CONTAINER_DIR"
        fi
    fi
else
    # On host: check project root directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
    NE301_DIR="$PROJECT_ROOT/ne301"
    
    if [ ! -d "$NE301_DIR" ]; then
        echo "[NE301 Init] Cloning NE301 project to: $NE301_DIR"
        git clone https://github.com/camthink-ai/ne301.git "$NE301_DIR"
    else
        echo "[NE301 Init] NE301 project directory already exists: $NE301_DIR"
    fi
fi

echo "[NE301 Init] Done"
