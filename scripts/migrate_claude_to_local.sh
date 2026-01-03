#!/bin/bash

# Claude Code Migration Script - From NPM to Local Installation
# This script helps migrate Claude Code from npm global installation to a local source installation

set -e

echo "====================================="
echo "Claude Code Migration Script"
echo "From NPM to Local Installation"
echo "====================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CLAUDE_LOCAL_DIR="$HOME/code/claude-code"
CLAUDE_BIN_DIR="$HOME/.local/bin"

# Step 1: Check current installation
echo -e "${YELLOW}Step 1: Checking current Claude Code installation...${NC}"
if command -v claude &> /dev/null; then
    CURRENT_CLAUDE=$(which claude)
    echo "Found Claude at: $CURRENT_CLAUDE"
    
    # Check npm installation
    if npm list -g @anthropic-ai/claude-code &> /dev/null; then
        NPM_VERSION=$(npm list -g @anthropic-ai/claude-code | grep @anthropic-ai/claude-code | awk '{print $2}')
        echo "NPM version: $NPM_VERSION"
    fi
else
    echo -e "${RED}Claude Code not found in PATH${NC}"
fi
echo

# Step 2: Clone repository
echo -e "${YELLOW}Step 2: Cloning Claude Code repository...${NC}"
if [ -d "$CLAUDE_LOCAL_DIR" ]; then
    echo "Directory $CLAUDE_LOCAL_DIR already exists"
    read -p "Do you want to update it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$CLAUDE_LOCAL_DIR"
        git pull origin main
        echo -e "${GREEN}Repository updated${NC}"
    fi
else
    echo "Cloning Claude Code to $CLAUDE_LOCAL_DIR..."
    git clone https://github.com/anthropics/claude-code.git "$CLAUDE_LOCAL_DIR"
    echo -e "${GREEN}Repository cloned successfully${NC}"
fi
echo

# Step 3: Install dependencies
echo -e "${YELLOW}Step 3: Installing dependencies...${NC}"
cd "$CLAUDE_LOCAL_DIR"
if [ -f "package.json" ]; then
    echo "Installing npm dependencies..."
    npm install
    echo -e "${GREEN}Dependencies installed${NC}"
else
    echo -e "${RED}package.json not found. Make sure you're in the correct directory${NC}"
    exit 1
fi
echo

# Step 4: Build the project (if needed)
echo -e "${YELLOW}Step 4: Building the project...${NC}"
if [ -f "package.json" ] && grep -q '"build"' package.json; then
    echo "Building Claude Code..."
    npm run build
    echo -e "${GREEN}Build completed${NC}"
else
    echo "No build script found, skipping..."
fi
echo

# Step 5: Create local bin directory if it doesn't exist
echo -e "${YELLOW}Step 5: Setting up local bin directory...${NC}"
if [ ! -d "$CLAUDE_BIN_DIR" ]; then
    mkdir -p "$CLAUDE_BIN_DIR"
    echo "Created $CLAUDE_BIN_DIR"
fi

# Step 6: Create symlink or wrapper script
echo -e "${YELLOW}Step 6: Creating executable link...${NC}"
CLAUDE_EXEC="$CLAUDE_LOCAL_DIR/bin/claude"
if [ -f "$CLAUDE_EXEC" ]; then
    # Create symlink
    ln -sf "$CLAUDE_EXEC" "$CLAUDE_BIN_DIR/claude"
    chmod +x "$CLAUDE_BIN_DIR/claude"
    echo "Created symlink: $CLAUDE_BIN_DIR/claude -> $CLAUDE_EXEC"
else
    # Create wrapper script if direct executable not found
    cat > "$CLAUDE_BIN_DIR/claude" << 'EOF'
#!/bin/bash
# Claude Code wrapper script
CLAUDE_HOME="$HOME/code/claude-code"
cd "$CLAUDE_HOME"
node "$CLAUDE_HOME/dist/index.js" "$@"
EOF
    chmod +x "$CLAUDE_BIN_DIR/claude"
    echo "Created wrapper script at $CLAUDE_BIN_DIR/claude"
fi
echo

# Step 7: Update PATH
echo -e "${YELLOW}Step 7: Updating PATH configuration...${NC}"
SHELL_RC=""
if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ] || [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    if ! grep -q "$CLAUDE_BIN_DIR" "$SHELL_RC"; then
        echo "" >> "$SHELL_RC"
        echo "# Claude Code local installation" >> "$SHELL_RC"
        echo "export PATH=\"$CLAUDE_BIN_DIR:\$PATH\"" >> "$SHELL_RC"
        echo -e "${GREEN}Added PATH to $SHELL_RC${NC}"
        echo "Please run: source $SHELL_RC"
    else
        echo "PATH already includes $CLAUDE_BIN_DIR"
    fi
fi
echo

# Step 8: Remove npm global installation (optional)
echo -e "${YELLOW}Step 8: Remove npm global installation?${NC}"
read -p "Do you want to remove the npm global installation? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing npm global installation..."
    npm uninstall -g @anthropic-ai/claude-code
    echo -e "${GREEN}NPM global installation removed${NC}"
else
    echo "Keeping npm global installation"
fi
echo

# Step 9: Verify installation
echo -e "${YELLOW}Step 9: Verifying installation...${NC}"
export PATH="$CLAUDE_BIN_DIR:$PATH"
if command -v claude &> /dev/null; then
    echo -e "${GREEN}✓ Claude Code is available in PATH${NC}"
    claude --version 2>/dev/null || echo "Version check not available"
else
    echo -e "${RED}✗ Claude Code not found in PATH${NC}"
    echo "Please run: source $SHELL_RC"
fi

echo
echo "====================================="
echo -e "${GREEN}Migration Complete!${NC}"
echo "====================================="
echo
echo "Next steps:"
echo "1. Run: source $SHELL_RC"
echo "2. Test with: claude --help"
echo "3. To update Claude Code in the future:"
echo "   cd $CLAUDE_LOCAL_DIR && git pull && npm install"
echo
echo "Local installation directory: $CLAUDE_LOCAL_DIR"
echo "Executable location: $CLAUDE_BIN_DIR/claude"