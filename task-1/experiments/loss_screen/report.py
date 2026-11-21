"""Report measured loss comparisons without using test data."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    s=json.loads((HERE/'results/summary.json').read_text())
    baseline=s['baseline']
    lines=['# Porównanie BCE, równoważenia klas i focal loss','',
           'Głowica 32×32 i zamrożone ekstraktory bez zmian. Seed 42, batch 8, AdamW 0.001, maksymalnie 50 epok i patience 8. Epoka i próg dobierane na zapisanym wewnętrznym podziale; potem trening od nowa na całej części uczącej foldu.',
           'Screening: ten sam pierwszy fold co poprzednio, po 57 obrazów walidacyjnych na metodę. CV tylko przy poprawie średniego mIoU >= 0.005 i poprawie w co najmniej dwóch metodach. CV obejmuje fold screeningowy, więc nie jest niezależnym testem wyboru wariantu.',
           'Focal nie używa wag alpha; gamma wynosi 1 lub 2. Zmniejsza też skalę składnika pikselowego względem Dice: poprawy nie można przypisać wyłącznie wyborowi trudnych pikseli.',
           'Równoważenie per obraz: osobne średnie BCE dla każdej obecnej klasy, potem średnia klas; wynik mieszany 50/50 ze zwykłym BCE. Małe obszary nadal mogą dostać dużą wagę.', '',
           'Stan: '+s['status'], '',
           '| Wariant | PixelSeal | TrustMark | StegaStamp | Łącznie | Zmiana | CV |',
           '|---|---:|---:|---:|---:|---:|---|']
    base=[baseline['methods'][m]['folds'][0]['selected']['miou'] for m in ('pixelseal','trustmark','stegastamp')]
    lines.append('| Bazowy BCE + Dice | '+' | '.join(f'{v:.6f}' for v in base+[np.mean(base)])+' | — | już wykonane |')
    for variant,data in s['variants'].items():
        vals=[r['selected']['miou'] for r in data['screen']]
        status='ukończone' if data['cv'] else ('oczekuje' if data['passed'] else 'pominięte')
        lines.append(f'| {variant} | '+' | '.join(f'{v:.6f}' for v in vals+[np.mean(vals)])+f" | {data['screen_delta']:+.6f} | {status} |")
    for variant,data in s['variants'].items():
        lines+=['',f'## {variant}', '',s['formulas'][variant],'',
                '| Metoda | IoU czyste | IoU watermark | FPR | FNR | Udział predykcji | Udział GT | Epoka | Próg |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in data['screen']:
            d=r['diagnostics']; v=r['selected']
            values=[v['iou_clean'],v['iou_watermark'],d['fpr'],d['fnr'],d['predicted_fraction'],d['target_fraction']]
            lines.append(f"| {r['method']} | "+' | '.join(f'{x:.4f}' for x in values)+f" | {r['epoch']} | {r['threshold']} |")
        if data['cv']:
            lines+=['','| Metoda | Fold 1 | Fold 2 | Fold 3 | Średnia | SD | OOF | Zmiana OOF |','|---|---:|---:|---:|---:|---:|---:|---:|']
            for m in ('pixelseal','trustmark','stegastamp','overall'):
                groups=[[r for r in data['cv'] if r['fold']==f and (m=='overall' or r['method']==m)] for f in (1,2,3)]
                vals=[np.mean([x for r in group for x in r['selected']['per_image']]) for group in groups]
                oof=np.mean([x for group in groups for r in group for x in r['selected']['per_image']])
                reference=baseline['overall']['oof_miou'] if m=='overall' else baseline['methods'][m]['oof_miou']
                lines.append(f'| {m} | '+' | '.join(f'{v:.6f}' for v in vals+[np.mean(vals),np.std(vals,ddof=1),oof,oof-reference])+' |')
    lines+=['','IoU klas: średnia po obrazach. FPR=FP/liczba czystych pikseli; FNR=FN/liczba oznaczonych pikseli (agregacja pikselowa). Pełne liczniki FP/FN i wyniki według powierzchni masek w results/summary.json.',
            'Checkpointy odtwarzają predykcje z tolerancją 1e-7; metryki sprawdzono z zapisanych prawdopodobieństw. Nie użyto danych testowych ani leaderboardu.',
            'Odtworzenie: `task-1/.venv/Scripts/python.exe task-1/experiments/loss_screen/run.py`. Po każdej ukończonej próbie skrypt aktualizuje raport i dashboard PL/EN.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
