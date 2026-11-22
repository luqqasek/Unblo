import json,shutil
def add_rows(experiments,dist,row):
    from polish import COPY
    COPY.update({'TriForensic unified training':'TriForensic: wspólny trening','Single fixed fold1 screening,57 images':'Jeden ustalony fold1,57 obrazów'})
    root=experiments/'triforensic_wm';path=root/'results/measurements.json'
    if not path.exists():return
    for m in json.loads(path.read_text()):
        g=m['groups'];row('triforensic_wm',m['family'],m['run']+' / update '+m['step'],m['protocol'],m['miou'],f"Delta {m['delta']:+.5f}; FG<=1% {g['fg_le1']['fg_iou']}; FG1–5% {g['fg_1to5']['fg_iou']}; clean<=1% {g['clean_le1']['clean_iou']}; clean1–5% {g['clean_1to5']['clean_iou']}. Fixed development fold; not public leaderboard.")
    target=dist/'reports/triforensic_wm/results';target.mkdir(parents=True,exist_ok=True)
    for source in [*(root/'results').rglob('*.json'),*(root/'results').rglob('*.png')]:
        dest=target/source.relative_to(root/'results');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
