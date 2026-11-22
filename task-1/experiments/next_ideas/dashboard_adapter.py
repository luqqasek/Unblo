import json,shutil
def add_rows(experiments,dist,row):
    from polish import COPY
    COPY.update({'A1 baseline':'A1: baza','A5 top1pct':'A5: najlepszy1% komórek','A5 best13box':'A5: najlepszy prostokąt13x13'})
    root=experiments/'next_ideas';path=root/'results/measurements.json'
    if not path.exists():return
    for m in json.loads(path.read_text()):row('next_ideas',m['family'],m['name'],m['protocol'],m['miou'],m['note'])
    target=dist/'reports/next_ideas/results';target.mkdir(parents=True,exist_ok=True)
    for source in [*(root/'results').rglob('*.json'),*(root/'results').rglob('*.png')]:
        dest=target/source.relative_to(root/'results');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
