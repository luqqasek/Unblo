"""Summarize measured screening/CV runs and verify saved scores."""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'frozen_extractor'))
import cross_validate as cv


def main():
    summary = json.loads((HERE/'results/summary.json').read_text())
    baseline = json.loads((cv.OUT/'metrics.json').read_text())
    lines = ['# Symetryczny Dice i dekoder 128×128', '',
             'Dwa niezależne eksperymenty: BCE + Dice obu klas przy dotychczasowej głowicy 32×32 oraz oryginalny BCE + Dice watermarku z fuzją i konwolucjami w 128×128.',
             'Wariant 128 korzysta z ponownie wyciągniętych cech: zachowuje natywne rozmiary do 128×128. Nie powiększa starego cache 32×32. PixelSeal/TrustMark nie mają natywnych cech 128×128 w wybranych warstwach; są interpolowane po projekcji.',
             'Seed 42, AdamW 0.001, batch 8, maksymalnie 50 epok, patience 8. Te same podziały i dobór epoki/progu wewnątrz treningu co poprzednie CV.',
             'Screening: pierwszy fold, 57 obrazów walidacyjnych na metodę. Próg wejścia do CV ustalony przed wynikami: średnia poprawa >= 0.005 oraz poprawa w co najmniej 2 metodach.', '',
             '| Wariant | Metoda | Bazowy fold 1 | Nowy fold 1 | Zmiana | Epoka | Próg |',
             '|---|---|---:|---:|---:|---:|---:|']
    for variant, data in summary['variants'].items():
        if variant != next(iter(summary['variants'])):
            lines += ['', '| Wariant | Metoda | Bazowy fold 1 | Nowy fold 1 | Zmiana | Epoka | Próg |',
                      '|---|---|---:|---:|---:|---:|---:|']
        for r in data['screen']:
            b=baseline['methods'][r['method']]['folds'][0]['selected']['miou']
            lines.append(f"| {variant} | {r['method']} | {b:.6f} | {r['selected']['miou']:.6f} | {r['selected']['miou']-b:+.6f} | {r['epoch']} | {r['threshold']} |")
        for r in (data['cv'] or data['screen']):
            p=HERE/'results'/variant/f"{r['method']}_fold{r['fold']}.npz"
            with np.load(p) as z:
                assert z['names'].tolist()==r['val_names']
                score=cv.model.metrics(z['probability'],z['target'],r['threshold'])
                np.testing.assert_allclose(score['per_image'],r['selected']['per_image'],atol=1e-12)
        lines += ['', f"{variant}: zmiana łącznego wyniku pierwszego foldu {data['screen_delta']:+.6f}; próg CV {'spełniony' if data['passed'] else 'niespełniony'}."]
        if data['cv']:
            lines += ['', '| Metoda | Fold 1 | Fold 2 | Fold 3 | Średnia ± SD | OOF |', '|---|---:|---:|---:|---:|---:|']
            for method in (*cv.METHODS, 'overall'):
                groups=[[r for r in data['cv'] if r['fold']==f and (method=='overall' or r['method']==method)] for f in (1,2,3)]
                values=[float(np.mean([x for r in group for x in r['selected']['per_image']])) for group in groups]
                oof=float(np.mean([x for group in groups for r in group for x in r['selected']['per_image']]))
                lines.append(f"| {method} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {np.mean(values):.6f} ± {np.std(values,ddof=1):.6f} | {oof:.6f} |")
    lines += ['', 'Screening służy wyborowi wariantu, nie stanowi niezależnego testu. Ewentualne CV obejmuje również fold użyty do screeningu; wynik ma obciążenie selekcją. Nie zmieniano podziałów w celu poprawy wyniku. Nie używano zbioru testowego ani leaderboardu.',
              'Zapisane checkpointy odtworzyły predykcje z dokładnością 1e-7; metryki przeliczono z zapisanych prawdopodobieństw.',
              'Odtworzenie: `task-1/.venv/Scripts/python.exe task-1/experiments/head_ablation/run.py`, następnie `report.py`. Zakończone dopasowania są wznawiane z plików wynikowych.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Verified metrics and wrote report.')


if __name__=='__main__':
    main()
