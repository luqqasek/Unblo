import json,shutil

def add_rows(experiments,dist,row):
    root=experiments/'iml_holes';path=root/'results/measurements.json'
    if not path.exists():return
    for m in json.loads(path.read_text()):row('iml_holes',m['family'],m['name'],m['protocol'],m['miou'],m['note'])
    target=dist/'reports/iml_holes/results';target.mkdir(parents=True,exist_ok=True)
    for source in [*(root/'results').rglob('*.json'),*(root/'results').rglob('*.png')]:
        dest=target/source.relative_to(root/'results');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
