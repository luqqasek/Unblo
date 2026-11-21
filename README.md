# European AI Championship

- [Task 1 — Localize Image Watermark](task-1/README.md)
- [Task 2](task-2/README.md)
- [Task 3](task-3/README.md)

Repository: https://github.com/luqqasek/European-AI-Hackathon (private).
Each task has an independent workspace. Data, weights, environments and outputs are ignored.

## Set up another machine

Install Git and Python 3.11 first. Sign in to GitHub using an account with access
(`gh auth login` if using GitHub CLI), then:

```bash
git clone https://github.com/luqqasek/European-AI-Hackathon.git
cd European-AI-Hackathon/task-1
bash scripts/setup.sh
source .venv/bin/activate
```

The setup script installs dependencies, downloads the official train/test archives
and model weights, extracts data, and verifies the environment. Allow several GB
of disk space. Downloads need internet access. Start coding in
`task-1/submission_template.py`, in the `predict()` function.

macOS CPU inference has been verified. The Linux/Juelich dependency list targets
NVIDIA CUDA; that setup has not been verified on a second machine. Juelich team
accounts and shared folder configuration are described in the downloaded official
setup README.

### Windows (PowerShell)

Install Git for Windows and Python 3.11 with the Python launcher (`py`).
After cloning, run from `task-1`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\.venv\Scripts\python.exe submission_template.py --data data --out outputs/submission.pt
```

This uses CPU inference and needs no Bash, WSL, or CUDA installation. Select
`.venv\Scripts\python.exe` in your editor. The Windows installer has not yet
been executed on Windows; the Mac setup and decoder checks were verified locally.
