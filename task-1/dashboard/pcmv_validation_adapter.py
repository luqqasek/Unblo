"""Keep oracle mIoU separate from decoder-ranking AUC; publish measured artifacts."""
import json, shutil

def add_rows(experiments, dist, row):
    from polish import COPY
    COPY.update({
        'Frozen PS baseline':'Zamrożona baza PS',
        'GT oracle: at most one toggle':'Oracle GT: najwyżej jedna zmiana',
        'GT oracle: at most two disjoint toggles':'Oracle GT: najwyżej dwie rozłączne zmiany',
        'Fixed-fold reference · 57 PixelSeal images':'Baza na stałym foldzie · 57 obrazów PixelSeal',
        'GT oracle diagnostic · 57 PixelSeal images':'Diagnostyka oracle GT · 57 obrazów PixelSeal',
        'Incomplete proposal set; GT selects actions. No learned PCMV improvement or public score. Decoder-ranking AUC is reported separately in the report.':'Niepełny zestaw propozycji; GT wybiera zmiany. Brak zmierzonej poprawy wyuczonego PCMV i wyniku publicznego. AUC rankingu dekodera podano osobno w raporcie.'})
    root=experiments/'pcmv_validation';path=root/'results/proposal_screen.json'
    if not path.exists():return
    d=json.loads(path.read_text());s=d['summary']['all']
    detail='Incomplete proposal set; GT selects actions. No learned PCMV improvement or public score. Decoder-ranking AUC is reported separately in the report.'
    for key,label in [('base','Frozen PS baseline'),('oracle_one','GT oracle: at most one toggle'),('oracle_two','GT oracle: at most two disjoint toggles')]:
        protocol='Fixed-fold reference · 57 PixelSeal images' if key=='base' else 'GT oracle diagnostic · 57 PixelSeal images'
        row('pcmv_validation','pixelseal',label,protocol,s[key],detail)
    target=dist/'reports/pcmv_validation'
    target.mkdir(parents=True,exist_ok=True)
    reproduction=root/'REPRODUCE.md'
    if reproduction.exists():
        from report_view import render_report
        shutil.copyfile(reproduction,target/reproduction.name)
        (target/'REPRODUCE.html').write_text(render_report(reproduction,[root/'REPORT.md',reproduction]),encoding='utf-8')
    for source in (root/'results').glob('*'):
        if source.suffix in {'.json','.png'}:
            dest=target/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
