import json,shutil


def add_rows(experiments,dist,row):
    root=experiments/'pcmv';path=root/'results/measurements.json'
    if not path.exists():return
    from polish import COPY
    COPY.update({'Global carrier EMA1561 / exact inference':'Globalna gałąź z cechami lokalnymi EMA1561 / dokładna inferencja','PCMV control_b':'PCMV: kontrola propozycji','PCMV pcmv':'PCMV: weryfikacja wiadomością i JND'})
    for m in json.loads(path.read_text()):row('pcmv',m['family'],m['name'],m['protocol'],m['miou'],m['note'])
    for source in [*(root/'results').rglob('*.json'),*(root/'results').rglob('*.png')]:
        target=dist/'reports/pcmv/results'/source.relative_to(root/'results');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
