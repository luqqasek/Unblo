import json,shutil
def add_rows(experiments,dist,row):
    from polish import COPY
    COPY.update({'local presence':'Lokalny sygnał i małe watermarki','Single fixed fold1 screening, 57 images':'Jeden ustalony fold 1, 57 obrazów','Existing 170-image OOF plus frozen inference':'Istniejące OOF, 170 obrazów; zamrożona inferencja'})
    root=experiments/'local_presence';path=root/'results/measurements.json'
    if not path.exists():return
    for m in json.loads(path.read_text(encoding='utf-8')):
        g=m['groups'];row('local_presence',m['family'],m['run']+' / '+m['stage'],m['protocol'],g['all']['miou'],f"Delta {m['delta']:+.5f}; FG<=1% {g['fg_le1']['fg_iou']}; FG1–5% {g['fg_1to5']['fg_iou']}; clean<=1% {g['clean_le1']['clean_iou']}; clean1–5% {g['clean_1to5']['clean_iou']}. Development; no new public score.")
    target=dist/'reports/local_presence/results';target.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,target/'measurements.json')
    for source in (root/'results').glob('*/*.json'):
        if source.name=='patch_provenance.json':continue
        dest=target/source.parent.name/source.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest)
