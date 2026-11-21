"""Run on the cluster login node. Never queue a second node for this user."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=int, default=30)
    p.add_argument("--gpus", type=int, choices=[1, 2, 3, 4], default=1)
    p.add_argument("--development", action="store_true", help="Authorized rapid-turnaround partition, no hackathon reservation")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    jobs = subprocess.check_output(["squeue", "-h", "-u", "pszenny1", "-o", "%i %T %D"], text=True).strip()
    if jobs:
        raise SystemExit(f"Refusing another job; one-node-at-a-time policy. Existing jobs:\n{jobs}")
    if a.development:
        minutes = min(a.minutes,30)
    else:
        reservation = subprocess.check_output(["scontrol", "show", "reservation=cispahack", "-o"], text=True)
        values = dict(x.split("=", 1) for x in reservation.split() if "=" in x)
        if values.get("State") != "ACTIVE":
            raise SystemExit("Hackathon reservation is not active")
        remaining = (datetime.fromisoformat(values["EndTime"])-datetime.now()).total_seconds()
        minutes = min(a.minutes, int(remaining//60)-2)
    if minutes < 1:
        raise SystemExit("Reservation is ending; no allocation submitted")
    command = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not command:
        p.error("Provide Python module and arguments after --")
    Path("../logs").mkdir(exist_ok=True)
    script=Path(__file__).with_name("cluster_job.sh")
    if a.development:
        text=script.read_text()
        text="\n".join(line for line in text.splitlines() if not line.startswith("#SBATCH --reservation="))+"\n"
        script=Path("../development-job.sh").resolve()
        script.write_text(text)
    subprocess.run(["sbatch", "--parsable", f"--time={minutes}", f"--gres=gpu:{a.gpus}",
                    f"--partition={'dc-gpu-devel' if a.development else 'dc-gpu'}",
                    f"--cpus-per-task={min(32, a.gpus*8)}",
                    str(script), *command], check=True)


if __name__ == "__main__":
    main()
