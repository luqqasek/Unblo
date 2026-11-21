"""Known-message diagnostics: only measured mIoU goes into score columns."""
import json
import shutil


def add_rows(experiments,dist,row):
    from polish import COPY
    COPY.update({
        'Known-message evidence audit':'Audyt informacji ze znanej wiadomości',
        'Dense multiscale evidence, blurred-baseline IG, re-embedding NCC, label-free Keep/Drop and matched fusion.':'Gęste mapy w dwóch skalach, IG od rozmytej bazy, re-embedding NCC, Keep/Drop i kontrolowana fuzja.',
        'Development OOF refinement / 170 images':'Korekta rozwojowego OOF / 170 obrazów',
        'Matched fusion pilot / fold 1 / seed 42':'Kontrolowany pilot fuzji / fold 1 / seed 42',
        'No new submission; paired image bootstrap.':'Bez nowego zgłoszenia; bootstrap sparowanych obrazów.',
        'Saved baseline':'Zapisana baza',
        'Message fusion':'Fuzja ze znaną wiadomością',
        'Wrong-message fusion control':'Kontrola fuzji z błędną wiadomością',
    })
    folder=experiments/'known_message_audit'
    source=folder/'results/summary.json'
    if not source.exists():return
    d=json.loads(source.read_text(encoding='utf-8'))
    for r in d['thresholds']:
        if r['group']!='all':continue
        lo,hi=r['delta_ci95']
        protocol='Oracle diagnostic · GT selection' if r['variant']=='oracle' else 'Development OOF refinement / 170 images'
        row('known_message_audit',r['method'],r['variant'],protocol,r['miou'],
            f"Delta {r['delta']:+.5f}; CI95 [{lo:+.5f}, {hi:+.5f}]. No new submission; paired image bootstrap.")
    for f in d.get('fusion',[]):
        for r in f['folds']:
            protocol='Matched fusion pilot / fold 1 / seed 42' if r['fold']==1 else 'Development fusion confirmation'
            row('known_message_audit',f['method'],'Saved baseline',protocol,r['baseline_miou'])
            for key,label in [('real','Message fusion'),('wrong','Wrong-message fusion control')]:
                v=r['variants'][key]
                row('known_message_audit',f['method'],label,protocol,v['miou'],
                    f"Epoch {v['epoch']}; threshold {v['threshold']}; counterfactual mIoU {v['counterfactual_miou']:.5f}.")
    target=dist/'reports/known_message_audit'
    target.mkdir(parents=True,exist_ok=True)
    for file in folder.glob('*.py'):shutil.copyfile(file,target/file.name)
    for file in (folder/'results/fusion').glob('*_summary.json'):
        dest=target/'results/fusion'/file.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
