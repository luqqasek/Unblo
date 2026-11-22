import json,shutil

def add_rows(experiments,dist,row):
    from polish import COPY
    COPY.update({'local_top164':'Najlepsze164 komórki lokalnego detektora','local_ring_rectangle':'Prostokąt według kontrastu z otoczeniem','Actual deployment / local top164':'Rzeczywista inferencja /164 komórki lokalne','raw1000':'Wagi po1000 krokach','equal_probability_450_1000':'Średnia prawdopodobieństw450/1000'})
    for experiment in ('carrier_fallback','ss_continuation','data_expansion'):
        root=experiments/experiment;path=root/'results/measurements.json'
        if path.exists():
            for m in json.loads(path.read_text()):row(experiment,m['family'],m['variant'],m['protocol'],m['miou'],m['note'])
        for source in [*(root/'results').rglob('*.json'),*(root/'results').rglob('*.png')]:
            target=dist/'reports'/experiment/'results'/source.relative_to(root/'results');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
