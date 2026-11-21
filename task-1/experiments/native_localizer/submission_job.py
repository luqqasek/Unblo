"""Run predeclared inference and validate package after both training jobs succeed."""
import argparse
from pathlib import Path
import subprocess
import sys
from .common import PROJECT

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args()
 out=a.root/'submissions/native-three-family-pilot.pt'
 for module,args in [('infer',['--recipe',str(PROJECT/'submission_recipe.json'),'--data','data','--out',str(out),'--batch','16']),('package_submission',['--submission',str(out),'--data','data'])]:
  subprocess.run([sys.executable,'-m','experiments.native_localizer.'+module,*args],check=True)
 print('SUBMISSION_READY',out,flush=True)
if __name__=='__main__':main()
