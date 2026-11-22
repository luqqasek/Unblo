"""Portable entrypoint for auxiliary diagnostics, final report and dashboard export."""
import argparse,os,sys,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
p=argparse.ArgumentParser();p.add_argument('--families',nargs='+',default=['pixelseal','trustmark','stegastamp']);p.add_argument('--final',action='store_true');a=p.parse_args()
os.chdir(ROOT);os.environ.update(CUDA_VISIBLE_DEVICES='0',PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'src'),PYTHONUTF8='1',PYTHONUNBUFFERED='1')
commands=[('auxiliary audit',[sys.executable,'-X','utf8','-u','-m','experiments.rtx_all_information.audit_auxiliaries','--families',*a.families])]
if a.final:
    commands.extend([('report',[sys.executable,'-X','utf8','-m','experiments.rtx_all_information.report']),('dashboard',[sys.executable,'-X','utf8',str(HERE/'install_dashboard.py')]),('export',[sys.executable,'-X','utf8',str(HERE/'export_results.py')])])
log_path=HERE/'results/postprocess.log'
for name,cmd in commands:
    print('START',name,flush=True)
    with log_path.open('a',encoding='utf-8') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        print(log_path.read_text(encoding='utf-8')[-5000:],flush=True);sys.exit(result.returncode)
    print('FINISHED',name,flush=True)
