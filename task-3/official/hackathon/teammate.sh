#!/bin/bash
# =============================================================
# HACKATHON SETUP: STEP 2 OF 2 - TEAMMATES RUN AFTER TEAM LEAD
# =============================================================
# EXECUTION ORDER (IMPORTANT):
# 1. The TEAM LEAD must first run hackathon_setup.sh and
#    finish creating the shared folders, ACLs, datasets and envs.
# 2. ONLY AFTER THE TEAM LEAD FINISHES, each remaining team
#    member runs THIS script from their own JURECA account.
#    Run: source teammate.sh
#
# The team lead runs hackathon_setup.sh, not this script.
# Every remaining member runs teammate.sh individually.
#
# Before running, set TEAM_LEAD to the team lead's Juelich
# username and TEAM_FOLDER to the EXACT team name used by
# the team lead in hackathon_setup.sh.
# =============================================================

# =============================================================
# CONFIGURATION - EACH TEAMMATE EDITS THESE VALUES
# =============================================================
TEAM_LEAD=""    # Juelich username of your team lead
TEAM_FOLDER=""  # Exact team name from the team lead's setup
# =============================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

_log()   { echo -e "${BLUE}[INFO]${NC}  $1"; }
_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    _error "This script must be sourced: source teammate.sh"
    exit 1
fi

if [[ -z "$TEAM_LEAD" || -z "$TEAM_FOLDER" || "$TEAM_LEAD" == */* || "$TEAM_FOLDER" == */* || "$TEAM_FOLDER" == "." || "$TEAM_FOLDER" == ".." ]]; then
    _error "Set valid TEAM_LEAD and TEAM_FOLDER values before running."
    return 1
fi

PROJECT_NAME="training2666"
WRITE_TO_BASHRC="yes"

SCRATCH_BASE="/p/scratch/$PROJECT_NAME"
TEAM_DIR="$SCRATCH_BASE/$TEAM_LEAD/$TEAM_FOLDER"
UV_BASE="$TEAM_DIR/.uv"
HF_CACHE="$TEAM_DIR/.cache"

echo ""
echo "============================================="
echo "   Shared Team Setup - Teammates (Step 2/2)"
echo "============================================="
echo ""
echo "  Project     : $PROJECT_NAME"
echo "  Team lead   : $TEAM_LEAD"
echo "  Team folder : $TEAM_DIR"
echo ""

_log "Step 1/4 - Checking cluster tools..."

if ! command -v jutil &> /dev/null; then
    _error "jutil not found. Are you on JURECA?"
    return 1
fi
_ok "jutil found."

_log "Step 2/4 - Activating project $PROJECT_NAME..."

if ! jutil env activate -p "$PROJECT_NAME"; then
    _error "Could not activate project $PROJECT_NAME."
    return 1
fi

if [[ -z "$PROJECT" ]]; then
    _error '\$PROJECT is empty after activation.'
    return 1
fi
_ok "Project activated. \$PROJECT = $PROJECT"

# Verify the team lead has finished the shared setup before writing config.
if [[ ! -d "$TEAM_DIR" ]]; then
    _error "Shared team folder does not exist: $TEAM_DIR"
    echo "       Ask the team lead to finish hackathon_setup.sh first."
    return 1
fi

if [[ ! -r "$TEAM_DIR" || ! -w "$TEAM_DIR" || ! -x "$TEAM_DIR" ]]; then
    _error "You do not have read/write/traverse access to: $TEAM_DIR"
    echo "       Ask the team lead to check the shared-folder ACLs."
    return 1
fi

_log "Step 3/4 - Exporting shared cache paths..."

export UV_CACHE_DIR="$UV_BASE/cache"
export UV_TOOL_DIR="$UV_BASE/tools"
export UV_PYTHON_INSTALL_DIR="$UV_BASE/python"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"

_ok "Shared environment variables exported in current shell."

if [[ "$WRITE_TO_BASHRC" == "yes" ]]; then
    if grep -Fq "TEAM SHARED UV/HF CONFIG: $TEAM_DIR" "$HOME/.bashrc" 2>/dev/null; then
        _warn "Shared config already present in ~/.bashrc"
    else
        cat <<EOF >> "$HOME/.bashrc"

# TEAM SHARED UV/HF CONFIG: $TEAM_DIR
export PATH="\$HOME/.local/bin:\$PATH"
export UV_CACHE_DIR="$UV_BASE/cache"
export UV_TOOL_DIR="$UV_BASE/tools"
export UV_PYTHON_INSTALL_DIR="$UV_BASE/python"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"
EOF
        _ok "Shared config written to ~/.bashrc"
    fi
fi

_log "Step 4/4 - Checking shared folder and uv..."
_ok "You can access the shared team folder."

if command -v uv &> /dev/null; then
    _ok "uv found: $(uv --version 2>&1)"
else
    _warn "uv is not available in your shell. Installing for this user..."

    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        _error "uv installation failed."
        return 1
    fi

    if [[ -f "$HOME/.local/bin/env" ]]; then
        source "$HOME/.local/bin/env"
    fi

    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &> /dev/null; then
        _ok "uv installed: $(uv --version 2>&1)"
    else
        _error "uv installation seems to have failed."
        echo "Run these manually:"
        echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo 'source $HOME/.local/bin/env'
        return 1
    fi
fi

echo ""
echo "============================================="
echo -e "${GREEN}   Teammate shell setup complete!${NC}"
echo "============================================="
echo ""
echo "Use the shared folder with:"
echo "  cd $TEAM_DIR/<dataset-name>"
echo "  source .venv/bin/activate"
echo ""
echo "Or run from inside a task directory:"
echo "  cd $TEAM_DIR/<dataset-name>"
echo "  uv run python main.py"
echo ""
