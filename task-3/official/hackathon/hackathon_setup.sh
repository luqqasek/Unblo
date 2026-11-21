#!/bin/bash
# =============================================================
# HACKATHON SETUP: STEP 1 OF 2 - TEAM LEAD RUNS FIRST
# =============================================================
# EXECUTION ORDER (IMPORTANT):
# 1. The TEAM LEAD configures and runs THIS script first.
#    Run: source hackathon_setup.sh
#    Wait until it finishes successfully.
# 2. ONLY AFTER THAT, each remaining team member configures
#    and runs teammate.sh from their own JURECA account.
#    Run: source teammate.sh
#
# The team lead runs only this script. Teammates run only
# teammate.sh. Do not run both scripts from the same account.
#
# Before running, set TEAM_LEAD to your Juelich username,
# TEAMMATE_1 through TEAMMATE_3 to your teammates' Juelich
# usernames, and TEAM_FOLDER to the agreed team name.
# =============================================================

# =============================================================
# CONFIGURATION - TEAM LEAD EDITS THESE VALUES
# =============================================================
TEAM_LEAD=""    # Your Juelich username
TEAMMATE_1=""   # Teammate's Juelich username; leave blank if unused
TEAMMATE_2=""
TEAMMATE_3=""
TEAM_FOLDER=""  # Your team name; teammates must use the exact same value
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
    _error "This script must be sourced: source hackathon_setup.sh"
    exit 1
fi

if [[ -z "$TEAM_LEAD" || -z "$TEAM_FOLDER" || "$TEAM_LEAD" == */* || "$TEAM_FOLDER" == */* || "$TEAM_FOLDER" == "." || "$TEAM_FOLDER" == ".." ]]; then
    _error "Set valid TEAM_LEAD and TEAM_FOLDER values before running."
    return 1
fi

if [[ "$TEAM_LEAD" != "$(id -un)" ]]; then
    _error "TEAM_LEAD must match the username running this script: $(id -un)"
    return 1
fi

# Hugging Face datasets to download (org/repo format)
HF_DATASETS=(
    "SprintML/Hackathon_Watermark_Localisation_Task"
    "SprintML/model_equality"
    "SprintML/unlearning"
)

PROJECT_NAME="training2666"
SCRATCH_BASE="/p/scratch/$PROJECT_NAME"
TEAM_LEAD_DIR="$SCRATCH_BASE/$TEAM_LEAD"
TEAM_DIR="$TEAM_LEAD_DIR/$TEAM_FOLDER"

# Build teammates array, skipping blank entries
TEAMMATES=()
for t in "$TEAMMATE_1" "$TEAMMATE_2" "$TEAMMATE_3"; do
    [[ -n "$t" ]] && TEAMMATES+=("$t")
done

if [[ "${#TEAMMATES[@]}" -eq 0 ]]; then
    _error "No teammates specified. Add at least one."
    return 1
fi

for t in "${TEAMMATES[@]}"; do
    if [[ "$t" == */* || "$t" == "$TEAM_LEAD" ]]; then
        _error "Invalid teammate username: $t"
        return 1
    fi
done

echo ""
echo "============================================="
echo "   Hackathon Setup - Team Lead (Step 1/2)"
echo "============================================="
echo ""
echo "  Team lead   : $TEAM_LEAD"
echo "  Teammates   : ${TEAMMATES[*]}"
echo "  Lead folder : $TEAM_LEAD_DIR"
echo "  Team folder : $TEAM_DIR"
echo ""

# -------------------------------------------------------------
# STEP 1: Check cluster
# -------------------------------------------------------------
_log "Step 1/6 - Checking cluster environment..."

if ! command -v jutil &> /dev/null; then
    _error "jutil not found. Are you on JURECA?"
    echo "       SSH in: ssh <username>@jureca.fz-juelich.de"
    return 1
fi
_ok "jutil found - on JURECA."

# -------------------------------------------------------------
# STEP 2: Activate project
# -------------------------------------------------------------
_log "Step 2/6 - Activating project $PROJECT_NAME..."

if ! jutil env activate -p "$PROJECT_NAME"; then
    _error "Could not activate project $PROJECT_NAME."
    return 1
fi

if [[ -z "$PROJECT" ]]; then
    _error '\$PROJECT empty after activation.'
    echo "       Try manually: jutil env activate -p $PROJECT_NAME"
    return 1
fi
_ok "Project activated. \$PROJECT = $PROJECT"

# -------------------------------------------------------------
# STEP 3: Create folders
# -------------------------------------------------------------
_log "Step 3/6 - Creating folders in scratch..."

cd "$SCRATCH_BASE" || { _error "Cannot cd to $SCRATCH_BASE"; return 1; }

if [[ -d "$TEAM_LEAD_DIR" ]]; then
    _warn "Team lead folder $TEAM_LEAD already exists. Skipping mkdir."
else
    mkdir "$TEAM_LEAD_DIR" || { _error "Could not create $TEAM_LEAD_DIR"; return 1; }
    _ok "Created $TEAM_LEAD_DIR"
fi

if [[ -d "$TEAM_DIR" ]]; then
    _warn "Team folder $TEAM_FOLDER already exists. Skipping mkdir."
else
    mkdir "$TEAM_DIR" || { _error "Could not create $TEAM_DIR"; return 1; }
    _ok "Created $TEAM_FOLDER inside $TEAM_LEAD."
fi

chmod g+s "$TEAM_LEAD_DIR" || { _error "Could not set directory group inheritance."; return 1; }

# -------------------------------------------------------------
# STEP 4: Set ACLs - team lead + teammates get full access
# -------------------------------------------------------------
_log "Step 4/6 - Setting ACLs on $TEAM_LEAD_DIR ..."

_lock_down_tree() {
    local dir="$1"

    # Restrict the top-level directory itself and all existing contents
    setfacl -m g::--- "$dir" || return 1
    setfacl -m o::--- "$dir" || return 1
    setfacl -R -m g::--- "$dir" || return 1
    setfacl -R -m o::--- "$dir" || return 1

    # Restrict default access on existing and future subdirectories
    find "$dir" -type d -exec setfacl -d -m g::--- {} + || return 1
    find "$dir" -type d -exec setfacl -d -m o::--- {} + || return 1
}

_grant_user_access() {
    local uname="$1"
    local dir="$2"

    # Existing directory tree
    setfacl -R -m "u:$uname:rwx" "$dir" || return 1

    # Default ACLs for new files and subdirectories
    find "$dir" -type d -exec setfacl -d -m "u:$uname:rwx" {} + || return 1
}

if ! _lock_down_tree "$TEAM_LEAD_DIR"; then
    _error "Could not restrict the team lead directory."
    return 1
fi

for uname in "$TEAM_LEAD" "${TEAMMATES[@]}"; do
    _log "  Granting access to: $uname"
    if ! _grant_user_access "$uname" "$TEAM_LEAD_DIR"; then
        _error "Could not grant ACL permissions to $uname."
        return 1
    fi
    _ok "  ACL set for $uname"
done

# Ensure ACL named-user entries have an effective rwx mask.
setfacl -R -m m:rwx "$TEAM_LEAD_DIR" || return 1
find "$TEAM_LEAD_DIR" -type d -exec setfacl -d -m m:rwx {} + || return 1

_log "Final ACL for team lead folder:"
getfacl "$TEAM_LEAD_DIR"

_log "Final ACL for team folder:"
getfacl "$TEAM_DIR"

# -------------------------------------------------------------
# STEP 5: Install uv + configure shared cache
# -------------------------------------------------------------
_log "Step 5/6 - Installing uv and configuring shared cache..."

if command -v uv &> /dev/null; then
    _warn "uv already installed. Skipping."
else
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        _error "uv installation failed."
        return 1
    fi
    if [[ -f "$HOME/.local/bin/env" ]]; then
        source "$HOME/.local/bin/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        _error "uv is still unavailable after installation."
        return 1
    fi
    _ok "uv installed."
fi

UV_BASE="$TEAM_DIR/.uv"
HF_CACHE="$TEAM_DIR/.cache"
mkdir -p "$UV_BASE" "$HF_CACHE" || return 1

# The marker makes the configuration check exact and repeat-safe.
if grep -Fq "TEAM SHARED UV/HF CONFIG: $TEAM_DIR" "$HOME/.bashrc" 2>/dev/null; then
    _warn "uv/HF cache config already in ~/.bashrc. Skipping."
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
    _ok "uv + HF cache paths written to ~/.bashrc"
fi

export UV_CACHE_DIR="$UV_BASE/cache"
export UV_TOOL_DIR="$UV_BASE/tools"
export UV_PYTHON_INSTALL_DIR="$UV_BASE/python"
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE/hub"

# -------------------------------------------------------------
# STEP 6: Download each dataset then create its per-task uv env
# -------------------------------------------------------------
_log "Step 6/6 - Downloading datasets and creating per-task environments..."

if [[ -n "$HF_TOKEN" ]]; then
    _ok "HF_TOKEN found - will be used for authentication."
else
    _log "No HF_TOKEN set - proceeding without (fine for public datasets)."
fi

BOOTSTRAP_VENV="$TEAM_DIR/.bootstrap"
if [[ ! -d "$BOOTSTRAP_VENV" ]]; then
    _log "Creating bootstrap venv for downloads..."
    uv venv -p 3.12 "$BOOTSTRAP_VENV" || return 1
    VIRTUAL_ENV="$BOOTSTRAP_VENV" uv pip install huggingface_hub || return 1
    _ok "Bootstrap venv ready."
else
    _warn "Bootstrap venv already exists. Skipping."
fi

BOOTSTRAP_PYTHON="$BOOTSTRAP_VENV/bin/python"

for dataset in "${HF_DATASETS[@]}"; do
    dataset_name="${dataset##*/}"
    dest="$TEAM_DIR/$dataset_name"
    VENV_DIR="$dest/.venv"
    REQ="$dest/requirements.txt"

    echo ""
    _log "--- Task: $dataset_name ---"

    if [[ -d "$dest" ]]; then
        _warn "  Dataset directory already exists. Skipping download."
    else
        _log "  Downloading $dataset..."
        "$BOOTSTRAP_PYTHON" - "$dataset" "$dest" <<'PYEOF'
import os
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    repo_type="dataset",
    local_dir=sys.argv[2],
    token=os.environ.get("HF_TOKEN"),
)
PYEOF
        if [[ $? -ne 0 ]]; then
            _error "  Download failed - skipping venv setup."
            continue
        fi
        _ok "  Downloaded $dataset_name."
    fi

    mkdir -p "$dest/output" || { _error "Cannot create output directory"; continue; }
    setfacl -m m:rwx "$dest/output"
    setfacl -d -m m:rwx "$dest/output"

    if [[ -d "$VENV_DIR" ]]; then
        _warn "  Venv already exists. Skipping."
        continue
    fi

    if [[ ! -f "$REQ" ]]; then
        _warn "  No requirements.txt found in $dataset_name - skipping venv."
        continue
    fi

    _log "  Setting up uv project for $dataset_name..."
    cd "$dest" || { _error "  Cannot cd into $dest"; continue; }

    if [[ ! -f pyproject.toml ]]; then
        uv init -p 3.12 --no-workspace || { _error "  uv init failed"; cd "$TEAM_DIR"; continue; }
    fi

    uv venv -p 3.12 "$VENV_DIR" || { _error "  uv venv failed"; cd "$TEAM_DIR"; continue; }

    # Remove standalone comments and blank lines from requirements.
    # Original behavior uses uv add to create pyproject.toml and uv.lock.
    mapfile -t DEPS < <(grep -vE '^[[:space:]]*(#|$)' "$REQ")

    if [[ "${#DEPS[@]}" -eq 0 ]]; then
        _warn "  requirements.txt is empty - skipping uv add."
    else
        if VIRTUAL_ENV="$VENV_DIR" uv add "${DEPS[@]}"; then
            _ok "  Done: pyproject.toml + uv.lock + .venv ready."
        else
            _error "  Failed to install requirements for $dataset_name."
        fi
    fi

    cd "$TEAM_DIR" || return 1
done

if [[ -d "$BOOTSTRAP_VENV" ]]; then
    rm -rf "$BOOTSTRAP_VENV"
    _ok "Bootstrap venv removed."
fi

echo ""
echo "============================================="
echo -e "${GREEN}   Team lead setup finished!${NC}"
echo "============================================="
echo "  Team access : $TEAM_LEAD + ${TEAMMATES[*]}"
echo "  uv          : $(uv --version 2>&1)"
echo ""
echo "  NEXT: Each remaining team member must now run:"
echo "        source teammate.sh"
echo "  with TEAM_LEAD=$TEAM_LEAD and TEAM_FOLDER=$TEAM_FOLDER"
echo ""
echo "  Activate a task environment:"
echo "  source $TEAM_DIR/<dataset-name>/.venv/bin/activate"
echo ""
echo "  Submit a job: sbatch job.sh"
echo "============================================="
echo ""
