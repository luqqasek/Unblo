import json,shutil

def add_rows(experiments,dist,row):
    root=experiments/'pixelseal_regions';path=root/'results/measurements.json'
    if not path.exists():return
    from polish import COPY
    COPY['Native region presence / inner-calibrated search']='Obecność w całym regionie / wyszukiwanie z kalibracją wewnętrzną'
    for m in json.loads(path.read_text()):row('pixelseal_regions',m['family'],m['variant'],m['protocol'],m['miou'],m['note'])
    for source in [*(root/'results').rglob('*.json'),*(root/'results').rglob('*.png')]:
        target=dist/'reports/pixelseal_regions/results'/source.relative_to(root/'results');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
