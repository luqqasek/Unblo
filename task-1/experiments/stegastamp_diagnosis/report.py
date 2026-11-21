"""Generate the diagnostic report from completed runs; no fabricated measurements."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'


def main():
    data=json.loads((OUT/'metrics.json').read_text())
    rows=list(csv.DictReader((OUT/'per_image.csv').open(encoding='utf-8')))
    ss=[r for r in data['summaries'] if r['method']=='stegastamp']
    lines=['# Diagnoza StegaStamp: dane, seed i wybór parametrów','',
        '## Zakres i wiarygodność','',
        'Odtworzono 510 zapisanych predykcji OOF z checkpointów oraz oficjalną metrykę (`frozen_extractor/verify_cv.py`). '
        'StegaStamp: średnia foldów 0,410596; odchylenie standardowe próby między foldami 0,062998. '
        'To zmienność podziałów, a nie przedział ufności ani pomiar zmienności seedów.','',
        'Użyto wyłącznie oryginalnego train i zamrożonych cache cech. Identyczne podziały zewnętrzne i wewnętrzne '
        'jak w poprzednim 3-fold CV. Seed 42 to oryginalny przebieg; 7 i 123 to nowe treningi. '
        'Każdy nowy seed sprawdzono przy stałej liczbie epok i progu z oryginalnego przebiegu oraz '
        'po ponownym doborze epoki i progu na wewnętrznej walidacji. Zmieniono jednocześnie inicjalizację '
        'i kolejność batchy; ten eksperyment nie rozdziela tych dwóch składników losowości. '
        'Nie użyto zbioru testowego ani leaderboardu. Wyniki są diagnostyczne na już analizowanych foldach.','',
        '## Zawartość foldów i rodzaj błędów','',
        '| Fold | mIoU | Udział GT | Udział predykcji | FP / obraz | FN / obraz | Małe regiony <25% | full_watermark=True |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in ss:
        lines.append(f"| {r['fold']} | {r['miou']:.4f} | {r['area']:.1%} | {r['predicted_area']:.1%} | {r['fp']:.1%} | {r['fn']:.1%} | {r['small_region_n']}/{r['n']} | {r['full_watermark']:.1%} |")
    lines+=['','FP/FN to odsetek wszystkich komórek obrazu, nie precision/recall.','',
        '| Fold | mIoU częściowego watermarku (False) | mIoU pełnego watermarku z przywróconym fragmentem (True) | Pole otworów | Krawędzie / piksel | Trafność bitów | Maska przestrzenna |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in ss:
        groups=r['by_full_watermark']
        lines.append(f"| {r['fold']} | {groups['False']['miou']:.4f} (n={groups['False']['n']}) | {groups['True']['miou']:.4f} (n={groups['True']['n']}) | {r['holes']:.2%} | {r['boundary']:.4f} | {r['bit_accuracy']:.4f} | {r['prior']:.4f} |")
    lines+=['','Dostępna etykieta `full_watermark` odróżnia pełne oznakowanie z przywróconym czystym fragmentem od częściowego oznakowania. '
        'Metadane nie zawierają osobnych etykiet outpainting / usunięcie / dodanie / transplantacja, '
        'więc nie przypisujemy tych typów na podstawie samego kształtu maski. '
        'Otwory obliczono przez binary_fill_holes na siatce 128×128, a krawędzie przez różnice sąsiednich komórek.','',
        '## Powtórzenia na tych samych podziałach','',
        '| Fold | Seed | Procedura | Epoki | Próg | mIoU |',
        '|---|---:|---|---:|---:|---:|']
    for r in data['runs']:
        lines.append(f"| {r['fold']} | {r['seed']} | {r['mode']} | {r['epoch']} | {r['threshold']} | {r['miou']:.4f} |")
    lines+=['','`fixed`: niezmienione epoki i próg z seed=42; `nested`: ponowny dobór na tym samym wewnętrznym podziale. '
        '`original`: oryginalny seed=42. Stałe epoki izolują zmienność treningu od zmienności doboru hiperparametrów.','',
        '| Fold | Średnia 3 seedów, stałe parametry | SD seedów | Średnia 3 seedów, dobór wewnętrzny | SD seedów |',
        '|---|---:|---:|---:|---:|']
    for fold in (1,2,3):
        fixed=[r['miou'] for r in data['runs'] if r['fold']==fold and r['mode'] in ('fixed','original')]
        nested=[r['miou'] for r in data['runs'] if r['fold']==fold and r['mode'] in ('nested','original')]
        lines.append(f'| {fold} | {np.mean(fixed):.4f} | {np.std(fixed,ddof=1):.4f} | {np.mean(nested):.4f} | {np.std(nested,ddof=1):.4f} |')
    # Standardize the observed group mix, without claiming causal attribution.
    weights={flag:sum(r['full_watermark']==flag for r in rows if r['method']=='stegastamp')/170 for flag in ('True','False')}
    standardized=[sum(weights[k]*r['by_full_watermark'][k]['miou'] for k in weights) for r in ss]
    lines+=['','Po przeważeniu obu kategorii full_watermark do wspólnych proporcji wszystkich 170 obrazów, '
        'mIoU foldów wynosi: '+', '.join(f'{v:.4f}' for v in standardized)+'. '
        'To opisowa korekta składu zbioru, nie dowód przyczynowości; modele w foldach trenowano na innych danych.','',
        '## Odtworzenie','',
        'Z katalogu głównego repozytorium:',
        '```powershell',
        'task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_diagnosis/run.py',
        'task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_diagnosis/report.py',
        'python task-1/dashboard/build.py','```','',
        'Artefakty: `results/metrics.json`, `per_image.csv`, checkpointy i prawdopodobieństwa nowych '
        'przebiegów nested, po 6 przykładów na fold oraz wykres rozrzutu. '
        'Hipotezy i interpretacja: [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md).']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
