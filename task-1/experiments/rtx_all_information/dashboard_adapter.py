import json,shutil

def add_rows(experiments,dist,row):
    from polish import COPY
    COPY.update({'All-information RTX3070':'Wszystkie informacje RTX3070',
      'Matched control RTX3070':'Dopasowana kontrola RTX3070',
      'Matched 3-fold development CV, seed42':'Dopasowana walidacja 3-fold, seed42',
      'Stopped after PixelSeal and TrustMark: 340 real images evaluated; StegaStamp not trained. Full synthetic pools for completed families; fixed updates and threshold; not a public score.':'Zatrzymano po PixelSeal i TrustMark: oceniono 340 obrazów; StegaStamp niewytrenowany. Pełne pule syntetyczne ukończonych rodzin; stały budżet i próg; to nie wynik publiczny.'})
    root=experiments/'rtx_all_information';path=root/'results/measurements.json'
    if not path.exists():return
    for m in json.loads(path.read_text(encoding='utf-8')):
        name='All-information RTX3070' if m['arm']=='all_information' else 'Matched control RTX3070'
        row('rtx_all_information',m['family'],name,'Matched 3-fold development CV, seed42',m['miou'],'Stopped after PixelSeal and TrustMark: 340 real images evaluated; StegaStamp not trained. Full synthetic pools for completed families; fixed updates and threshold; not a public score.')
    target=dist/'reports/rtx_all_information';target.mkdir(parents=True,exist_ok=True)
    for source in [root/'PROTOCOL.md',root/'reference_best.json',*(root/'results').glob('*.json'),*(root/'results').glob('*.csv'),*(root/'results').glob('*.png')]:
        if source.exists():
            dest=target/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    for source in (root/'results/runs').rglob('*.json'):
        dest=target/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
