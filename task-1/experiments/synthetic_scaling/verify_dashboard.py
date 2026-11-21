"""Check exact dashboard scores, report links, two languages and safe artifacts."""
from html.parser import HTMLParser
from pathlib import Path
import json
import shutil
import urllib.request

HERE=Path(__file__).resolve().parent
DIST=HERE.parents[1]/'dashboard/dist'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links += [v for k,v in attrs if k in ('href','src') and v]

def main():
    data=json.loads((DIST/'data.json').read_text(encoding='utf-8'))
    summary=json.loads((HERE/'results/summary.json').read_text())
    rows=[r for r in data['rows'] if r['experiment']=='synthetic_scaling'];assert len(rows)==15
    for r in summary['comparisons']:
        actual=next(x for x in rows if x['method']==r['method'] and x['variant']==r['variant']+' / seed mean')
        assert actual['miou']==r['miou'] and len(actual['folds'])==9
    base=DIST/'reports/synthetic_scaling';missing=[]
    for filename in ('REPORT.html','REPRODUCE.html','README.html','results/gallery/index.html'):
        path=base/filename;parser=Links();parser.feed(path.read_text(encoding='utf-8'))
        missing += [(filename,u) for u in parser.links if not u.startswith(('http','#','data:','mailto:')) and not (path.parent/u.split('#')[0]).exists()]
    assert not missing,missing
    assert not any(p.suffix in ('.pt','.onnx','.npz','.npy','.zip') for p in base.rglob('*'))
    for filename in ('index.html','index-en.html'):assert 'synthetic_scaling' in (DIST/filename).read_text(encoding='utf-8')
    urls=['reports/synthetic_scaling/REPORT.html','reports/synthetic_scaling/results/gallery/index.html',
          'reports/synthetic_scaling/results/per_image_metrics.csv','index.html','index-en.html']
    statuses={u:urllib.request.urlopen('http://127.0.0.1:8877/'+u).status for u in urls}
    assert set(statuses.values())=={200}
    result=dict(status='passed',dashboard_rows=len(rows),languages=['pl','en'],missing_report_links=missing,
                no_checkpoints_or_full_datasets_exposed=True,http=statuses)
    (HERE/'results/dashboard_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    shutil.copyfile(HERE/'results/dashboard_verification.json',base/'results/dashboard_verification.json')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
