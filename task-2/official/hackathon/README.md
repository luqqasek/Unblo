# Paris Hackathon: JURECA Environment Setup

**TLDR:** The **team lead** runs `hackathon_setup.sh` once. **Every other teammate** runs `teammate.sh` after the lead finishes. On future logins, activate the project rather than repeating initial setup.

This guide explains how to set up and use your team's shared environment on JURECA.

> **Project ID:** `training2666`. All commands below use this ID. If the organizers assign a different project, use that ID instead. The downloaded setup scripts must use the same ID.

## Setup scripts

- [`hackathon_setup.sh`](https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh): run **once by the team lead**, before anyone else sets up.
- [`teammate.sh`](https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh): run **individually by every other team member**, after the team lead finishes.

**Setup order:** Team lead first → all other teammates afterward. Sign in to JURECA with your **own Jülich account**. Do not run both scripts from the same account.

## 1. Team lead: initial setup

### Download the scripts

```bash
mkdir -p ~/hackathon-scripts
cd ~/hackathon-scripts

wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh -O hackathon_setup.sh
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
```

You **do not** need to create your scratch folder manually. The setup script creates it.

### Configure `hackathon_setup.sh`

Edit the configuration section, using actual Jülich usernames:

```bash
TEAM_LEAD="<your-juelich-username>"
TEAMMATE_1="<teammate-1-username>"
TEAMMATE_2="<teammate-2-username>"
TEAMMATE_3=""  # Leave unused entries empty
TEAM_FOLDER="<your-team-name>"
```

- `TEAM_LEAD` is the team lead's Jülich username.
- `TEAMMATE_1` through `TEAMMATE_3` are the other members' usernames. Leave unused slots as `""`.
- `TEAM_FOLDER` is the shared folder name. Tell all teammates the exact value.
- At least one teammate must be configured.

### Run the script

```bash
source hackathon_setup.sh
```

**Wait until the script finishes successfully before asking teammates to continue.** It activates the project, creates the scratch folder, configures access permissions, downloads datasets, creates task-specific `.venv` environments, and configures shared uv/Hugging Face cache locations.

The shared folder is:

```text
/p/scratch/training2666/<team-lead-username>/<team-name>/
```

## 2. Other teammates: join the shared environment

**Only start after the team lead has completed Section 1.** Each teammate follows these steps from their own JURECA account.

### Download `teammate.sh`

```bash
mkdir -p ~/hackathon-scripts
cd ~/hackathon-scripts
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
```

### Configure `teammate.sh`

```bash
TEAM_LEAD="<team-lead-username>"
TEAM_FOLDER="<team-name>"
```

These values must match the team lead's configuration. `TEAM_LEAD` is **not** your own username unless you are the lead.

### Run the script

```bash
source teammate.sh
```

This activates the project, configures shared cache paths, checks shared-folder access, installs `uv` if needed, and writes the shell configuration to `~/.bashrc`.

**Do not redownload datasets or recreate environments.** The team lead has already prepared the shared task folders.

## 3. Work on a task

Change into a task directory, replacing the placeholders with the real names:

```bash
cd /p/scratch/training2666/<team-lead>/<team-folder>/<dataset-name>
```

Activate its Python environment and run your code:

```bash
source .venv/bin/activate
python main.py
```

If the task is configured as a uv project, you may instead use:

```bash
uv run python main.py
```

Replace `main.py` with the actual script name. Be aware that `uv run` may attempt to synchronize dependencies; coordinate changes to shared environments with teammates.

### Expected folder layout

```text
/p/scratch/training2666/
└── <team-lead-username>/
    └── <team-folder>/
        ├── unlearning/
        │   ├── .venv/
        │   ├── requirements.txt
        │   └── output/
        ├── model_equality/
        │   ├── .venv/
        │   ├── requirements.txt
        │   └── output/
        ├── Hackathon_Watermark_Localisation_Task/
        │   ├── .venv/
        │   ├── requirements.txt
        │   └── output/
        ├── .uv/
        └── .cache/
```

The specific files present may vary by task and dataset.

## 4. Every new login/session

Activate the project:

```bash
jutil env activate -p training2666
```

If the cache variables are not available in the current shell:

```bash
source ~/.bashrc
```

Then enter your task directory and activate its `.venv` as needed. **Do not rerun the initial setup scripts at every login.**

## 5. Submit GPU jobs

Create `main.sh` in the relevant task directory:

```bash
#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

# Change to your task directory
cd /p/scratch/training2666/<team-lead>/<team-folder>/<dataset-name> || exit 1

# Activate the shared task environment
source .venv/bin/activate

# Run your program
srun python code.py

echo "done!"
```

Replace placeholders and `code.py` with the correct values. **The account, reservation, partition, and GPU request must match the resources assigned by the organizers.**

Create `output/` **before** submitting the job, since Slurm needs the log directory to exist:

```bash
mkdir -p output
sbatch main.sh
```

Check queued/running jobs:

```bash
squeue -u "$USER"
```

Logs are written to `output/<job-id>.out` and `output/<job-id>.err`.

## 6. Adding a teammate after initial setup

If someone joins later, **the team lead grants access**. Existing teammates do not need to rerun their scripts.

From the **team lead's account**, with the correct usernames:

```bash
TEAM_LEAD="<team-lead-username>"
NEW_MEMBER="<new-member-username>"
TEAM_LEAD_DIR="/p/scratch/training2666/$TEAM_LEAD"

# Grant access to existing files and directories
setfacl -R -m "u:$NEW_MEMBER:rwx" "$TEAM_LEAD_DIR"

# Grant default access on new files and subdirectories
find "$TEAM_LEAD_DIR" -type d \
    -exec setfacl -d -m "u:$NEW_MEMBER:rwx" {} +
```

These commands provide broad read/write/execute access within the lead's scratch tree. Run them **only** when that level of access is intended. The new member then follows **Section 2** and runs `source teammate.sh` from their own account.

## 7. Rules for shared work

- Do not delete shared folders or datasets.
- Do not modify ACL permissions unless you are the team lead managing team access.
- Do not recreate `.venv` environments without coordinating with the team.
- Avoid unnecessary or conflicting package changes in shared environments.
- Keep work within your assigned team folder and task directories.
- Put outputs and logs in the relevant task's `output/` folder.

## 8. Troubleshooting

### Cannot access the team folder

Confirm that `TEAM_LEAD` and `TEAM_FOLDER` are correct and that the lead completed setup. If you joined later, ask the lead to follow **Section 6**.

### `uv: command not found`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

### Python environment is not active

Enter the task directory and run:

```bash
source .venv/bin/activate
```

If `.venv` does not exist, ask the team lead to check whether initial setup succeeded.

### Project is not active

```bash
jutil env activate -p training2666
```

### A dataset or environment is missing

Ask the team lead to inspect the initial setup output. Do not overwrite existing shared environments without coordinating with the team.

---

**Quick recap:** The **team lead** runs `hackathon_setup.sh` once. **Every other teammate** runs `teammate.sh` after the lead finishes. On future logins, activate the project rather than repeating initial setup.
