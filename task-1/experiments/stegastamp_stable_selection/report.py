"""Paired comparisons against all original seeds, never just the weakest seed."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def main():
    data=json.loads((OUT/'metrics.json').read_text()); runs=data['runs']
    assert len(runs)==9
    lines=['# StegaStamp: stabilniejszy wybór epoki i progu','',
        'Zmiana: pojedynczy wewnętrzny holdout 80/20 zastąpiono trzema wewnętrznymi foldami, '
        'stratyfikowanymi według full_watermark i udziału maski (granice 25%, 50%, 75%). '
        'Epokę wybiera średnie per-image mIoU z połączonych predykcji wewnętrznych przy progu 0,5. '
        'Próg z [0,3; 0,4; 0,5; 0,6; 0,7] dobiera się z tych samych predykcji dla wybranej epoki. '
        'Następnie głowica jest trenowana od nowa na całym zewnętrznym train.','',
        'Bez zmian: zamrożone cechy, architektura głowicy, AdamW, loss 0,5 BCE + 0,5 Dice, '
        'batch 8, maks. 50 epok, patience 8. Zewnętrzne foldy i seedy 42/7/123 identyczne '
        'jak w poprzedniej diagnozie. Nie stosowano minimalnej liczby epok ani wymuszenia 18 epok.','',
        '## Wyniki porównania parami','',
        '| Fold | Seed | Poprzednio | Nowa selekcja | Zmiana mIoU | Epoki | Próg |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in runs:lines.append(f"| {r['fold']} | {r['seed']} | {r['baseline']:.4f} | {r['miou']:.4f} | {r['delta']:+.4f} | {r['epoch']} | {r['threshold']} |")
    summaries=[]
    lines+=['','| Seed | Poprzednia średnia foldów | Nowa średnia | Zmiana | Poprzednie SD foldów | Nowe SD foldów |',
            '|---|---:|---:|---:|---:|---:|']
    for seed in [42,7,123]:
        rs=[r for r in runs if r['seed']==seed]
        old=np.array([r['baseline'] for r in rs]); new=np.array([r['miou'] for r in rs])
        entry=dict(seed=seed,baseline=float(old.mean()),miou=float(new.mean()),delta=float((new-old).mean()),
            baseline_sd=float(old.std(ddof=1)),sd=float(new.std(ddof=1)),oof=float(np.mean([x for r in rs for x in r['per_image']])))
        summaries.append(entry)
        lines.append(f"| {seed} | {entry['baseline']:.4f} | {entry['miou']:.4f} | {entry['delta']:+.4f} | {entry['baseline_sd']:.4f} | {entry['sd']:.4f} |")
    overall=dict(baseline=float(np.mean([r['baseline'] for r in runs])),miou=float(np.mean([r['miou'] for r in runs])),
        delta=float(np.mean([r['delta'] for r in runs])),improved=sum(r['delta']>0 for r in runs),
        seeds=summaries)
    fold_summaries=[]
    lines+=['','| Fold | Poprzednia średnia seedów | Nowa średnia | Poprzednie SD seedów | Nowe SD seedów |',
            '|---|---:|---:|---:|---:|']
    for fold in (1,2,3):
        rs=[r for r in runs if r['fold']==fold]
        old=np.array([r['baseline'] for r in rs]);new=np.array([r['miou'] for r in rs])
        entry=dict(fold=fold,baseline=float(old.mean()),miou=float(new.mean()),baseline_sd=float(old.std(ddof=1)),sd=float(new.std(ddof=1)))
        fold_summaries.append(entry)
        lines.append(f"| {fold} | {entry['baseline']:.4f} | {entry['miou']:.4f} | {entry['baseline_sd']:.4f} | {entry['sd']:.4f} |")
    overall['folds']=fold_summaries
    (OUT/'summary.json').write_text(json.dumps(overall,indent=2))
    lines+=['',f"Średnia po 3 foldach × 3 seedach: **{overall['baseline']:.4f} → {overall['miou']:.4f}**, "
        f"zmiana **{overall['delta']:+.4f} mIoU** ({100*overall['delta']:+.2f} punktu procentowego). "
        f"Poprawa w {overall['improved']}/9 porównań fold–seed.", '',
        'Średnia foldów waży foldy równo; wynik OOF zapisany w summary.json waży obrazy równo (57/57/56). '
        'Powtórzone seedy używają tych samych obrazów, więc 9 przebiegów nie jest 9 niezależnymi próbami. '
        'Nie wyliczamy na tej podstawie pozornego przedziału ufności.','',
        '## Wniosek','',
        'Nowa procedura ogranicza załamanie oryginalnego seedu 42 i poprawia średnią wszystkich '
        'porównań. Nie poprawia każdego przebiegu: seed 123 ma nieznacznie niższą średnią. '
        'Największy zysk dotyczy stabilności drugiego foldu, a nie podniesienia najlepszego '
        'osiągniętego wyniku. Dla dalszych eksperymentów StegaStamp rekomendujemy tę procedurę '
        'jako lepiej uzasadniony sposób selekcji, zachowując raportowanie wszystkich trzech seedów. '
        'Kosztem jest trening trzech modeli wewnętrznych zamiast jednego. Poprzedni benchmark '
        'pozostaje zachowany do porównań; nie zmieniono go wstecznie.','',
        'Kontrola wizualna czterech najgorszych oryginalnych przykładów foldu 2 pokazuje, że '
        'nie wszystkie błędy lokalizacji zniknęły. Na 53934df0429a i 5b7127ca0a3a nadal '
        'przewidywana jest niemal pełna maska; b00270778e48 pogorszył się z 0,080 do 0,032. '
        'Poprawa średniej nie oznacza rozwiązania problemu małych obszarów.','',
        '## Ograniczenia i odtworzenie','',
        'Poprawka powstała po obejrzeniu błędów tych zewnętrznych foldów: jest to porównanie '
        'rozwojowe, nie nowy nietknięty test. Żadna zewnętrzna etykieta nie wybiera epoki ani progu '
        'w nowym kodzie. Stratyfikacja korzysta wyłącznie z masek zewnętrznego train. '
        'Zmiana obejmuje naraz rozmiar wewnętrznego train, stratyfikację i agregację trzech foldów; '
        'ich oddzielnych efektów nie zidentyfikowano. Nie użyto testu konkursowego ani leaderboardu.','',
        'Pełny przebieg (wznawia ukończone pary fold–seed):','```powershell',
        'task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stable_selection/run.py',
        'task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stable_selection/verify.py',
        'python task-1/dashboard/build.py','```','',
        'Pliki results/: konfiguracja, manifest podziałów, historie walidacji, 9 checkpointów, '
        'prawdopodobieństwa, metryki i wykres porównania.']
    receipt_path=OUT/'full_train/upload_receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        response=json.loads(receipt['response'])
        fit=json.loads((OUT/'full_train/prediction_summary.json').read_text())
        lines+=['','## Trening końcowy i leaderboard','',
            f"Po zakończeniu CV wytrenowano StegaStamp na wszystkich {fit['train_count']} obrazach, "
            f"seed {fit['seed']}, {fit['epochs']} epok, próg {fit['threshold']}. "
            'Parametry są medianami nowego CV dla seedu 42. Nie wybierano najlepszego seedu.', '',
            'Zmieniono 340 masek StegaStamp; 680 masek PixelSeal i TrustMark pochodzi bez zmian '
            'z wcześniejszego submission #1153. Zweryfikowano checkpoint, zgodność predykcji, '
            'unikalne nazwy, komplet 1020 binarnych masek 128×128 i limit rozmiaru.', '',
            f"Zgłoszenie **#{response['submission_id']}**, status **{response['status']}**, "
            f"publiczne mIoU **{response['score']:.4f}**, poprzedni najlepszy wynik "
            f"**{response['previous_best']:.4f}**, poprawa według serwera **{response['delta']:+.4f}**. "
            'Różnica zaokrąglonych wyświetlanych wyników może odbiegać od delty serwera. '
            'Wynik dotyczy publicznej części testu; wynik prywatny pozostaje nieznany.', '',
            f"Submission: `{receipt['file']}`, {fit['bytes']} bajtów; SHA256 `{receipt['sha256']}`.", '',
            'Odtworzenie pliku (nie wysyła automatycznie):','```powershell',
            'task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stable_selection/train_full_predict.py','```']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for seed in [42,7,123]:
        rs=[r for r in runs if r['seed']==seed]
        axes[0].plot([r['fold'] for r in rs],[r['delta'] for r in rs],'-o',label=f'seed {seed}')
        axes[1].scatter([r['baseline'] for r in rs],[r['miou'] for r in rs],label=f'seed {seed}')
    axes[0].axhline(0,color='gray',linestyle='--');axes[0].set(xlabel='Outer fold',ylabel='Paired mIoU improvement',xticks=[1,2,3])
    axes[1].plot([.3,.55],[.3,.55],color='gray',linestyle='--');axes[1].set(xlabel='Previous selection mIoU',ylabel='Stratified inner CV mIoU')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(OUT/'stegastamp_selection_examples.png',dpi=140);plt.close(fig)
    old=np.load(HERE.parent/'frozen_extractor/results/cv3/stegastamp_oof.npz')
    new=np.load(OUT/'stegastamp_fold2_seed42.npz')
    names=old['names'].tolist()
    row=next(r for r in runs if r['fold']==2 and r['seed']==42)
    # Choose examples by original score only, not by the magnitude of improvement.
    original_metrics=json.loads((HERE.parent/'frozen_extractor/results/cv3/metrics.json').read_text())
    before=original_metrics['methods']['stegastamp']['folds'][1]['selected']['per_image']
    chosen=np.argsort(before)[:4]
    fig,axes=plt.subplots(4,4,figsize=(12,12))
    for panel,j in zip(axes,chosen):
        name=new['names'][j];i=names.index(name)
        pictures=[Image.open(HERE.parents[1]/'data/train/images'/f'{name}.png'),old['target'][i],
            old['probability'][i]>=old['threshold'][i],new['probability'][j]>=new['threshold']]
        titles=[name,'GT',f'Before {before[j]:.3f}',f'After {row["per_image"][j]:.3f}']
        for ax,pic,title in zip(panel,pictures,titles):ax.imshow(pic,vmin=0,vmax=1);ax.set_title(title,fontsize=9);ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'stegastamp_fold2_examples.png',dpi=120);plt.close(fig)
    print(json.dumps(overall,indent=2),flush=True)


if __name__=='__main__':main()
