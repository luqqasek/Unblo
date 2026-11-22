"""Verify the selected Task 1 artifact with the portable final-solution entry point."""
import runpy,sys
from pathlib import Path
sys.argv=["reproduce.py","validate"]
runpy.run_path(str(Path(__file__).resolve().parent/"solution/reproduce.py"),run_name="__main__")
