import subprocess,sys,os
from pathlib import Path
def main():
    r=Path(__file__).resolve().parents[2];os.chdir(r)
    for f in ['pixelseal','trustmark','stegastamp']:
     p=subprocess.run([sys.executable,'-X','utf8','-u','-m','experiments.rtx_all_information.prepare','--family',f])
     if p.returncode:sys.exit(p.returncode)

if __name__ == "__main__":
    main()
