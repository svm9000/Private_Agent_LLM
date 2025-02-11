#!/bin/bash

# HOW TO RUN THIS SCRIPT:
# 1. Save this script to a file, e.g., 'setup_llama.sh'
# 2. Make the script executable: chmod +x setup_llama.sh
# 3. Run the script: ./setup_llama.sh
# 
# REQUIREMENTS:
# - Git, CMake, and Make should be installed on your system
# - Ensure you have sufficient permissions to create directories and symlinks
# - Run this script in a directory where you have write permissions

# Set the paths
LLAMA_CPP_DIR="$HOME/.ollama/.../llama.cpp"
BUILD_DIR="$LLAMA_CPP_DIR/build"
QUANTIZE_PATH="$BUILD_DIR/bin/quantize"
SYMLINK_PATH="$LLAMA_CPP_DIR/llama-quantize"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required dependencies
if ! command_exists git; then
  echo "Error: git is not installed. Please install git."
  exit 1
fi

if ! command_exists cmake; then
  echo "Error: cmake is not installed. Please install cmake."
  exit 1
fi

if ! command_exists make; then
  echo "Error: make is not installed. Please install make."
  exit 1
fi

# Check if llama.cpp exists and remove it if it does
if [ -d "$LLAMA_CPP_DIR" ]; then
  echo "Removing existing llama.cpp directory..."
  rm -rf "$LLAMA_CPP_DIR"
  echo "llama.cpp directory removed."
fi

# Clone the llama.cpp repository
echo "Cloning llama.cpp repository..."
git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
if [ $? -ne 0 ]; then
  echo "Error: Failed to clone llama.cpp repository."
  exit 1
fi
echo "llama.cpp repository cloned successfully."

# Build llama.cpp using CMake
echo "Building llama.cpp using CMake..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit

cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j$(nproc)
if [ $? -ne 0 ]; then
  echo "Error: Build failed."
  exit 1
fi
echo "Build completed successfully."

# Verify that the quantize executable exists
if [ ! -f "$QUANTIZE_PATH" ]; then
  echo "Error: Quantize executable not found at $QUANTIZE_PATH."
  exit 1
fi

# Create the symlink for llama-quantize
echo "Creating symlink for llama-quantize..."
ln -sf "$QUANTIZE_PATH" "$SYMLINK_PATH"
if [ $? -ne 0 ]; then
  echo "Error: Failed to create symlink."
  exit 1
fi
echo "Symlink created successfully."

echo "llama.cpp setup complete."

# AFTER RUNNING:
# - The llama.cpp project will be cloned and built in $LLAMA_CPP_DIR
# - A symlink to the quantize executable will be created at $SYMLINK_PATH
# - You can now use the quantize tool by running $SYMLINK_PATH

exit 0
