"""Count failure symptoms for the retained #1202 pipeline; verify against OOF masks."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from common import HERE, OUT, METHODS, load_cohort, ious


def main():
    source = OUT / 'per_image_metrics.csv'
    df = pd.read_csv(source)
    df = df[(df.seed == 42) & (
        ((df.watermark_method != 'stegastamp') & (df.variant == 'original')) |
        ((df.watermark_method == 'stegastamp') & (df.variant == 'stable'))
    )].copy()
    assert len(df) == df.image_id.nunique() == 510
    assert df.groupby('watermark_method').size().to_dict() == dict.fromkeys(METHODS, 170)
    for method in METHODS:
        cohort = load_cohort(method, 'stable' if method == 'stegastamp' else 'original', 42)
        pred = cohort['prob'] >= cohort['threshold'][:, None, None]
        gt = cohort['gt']
        rows = df.set_index('image_id').loc[cohort['names']]
        values = ious(pred, gt)
        for key, value in zip(['official_miou', 'iou_watermark', 'iou_clean', 'tp', 'fp', 'fn', 'tn'], values):
            np.testing.assert_allclose(rows[key], value, atol=1e-10, rtol=0)
        np.testing.assert_allclose(rows.gt_foreground_fraction, gt.mean((1, 2)), atol=1e-10)
        np.testing.assert_allclose(rows.predicted_foreground_fraction, pred.mean((1, 2)), atol=1e-10)
        for i, (g, p) in enumerate(zip(gt, pred)):
            a = g & ~ndi.binary_erosion(g, border_value=1)
            b = p & ~ndi.binary_erosion(p, border_value=1)
            if a.any() and b.any():
                precision = (ndi.distance_transform_edt(~a)[b] <= 2).mean()
                recall = (ndi.distance_transform_edt(~b)[a] <= 2).mean()
                score = 2 * precision * recall / (precision + recall) if precision + recall else 0.
            else:
                score = 0. if a.any() or b.any() else np.nan
            np.testing.assert_allclose(rows.boundary_internal_f1_2.iloc[i], score, atol=1e-10)

    rules = [
        ('wrong_boundary', 'Błędna lub brakująca granica maski', 'Internal boundary F1 < 0,10; tolerancja 2 komórki (8 px obrazu), bez ramki obrazu.', df.boundary_internal_f1_2 < .1),
        ('missed_clean_regions', 'Niewykrywanie czystych fragmentów przy dużym watermarku', 'GT > 50% obrazu i wykryte mniej niż 50% czystych pikseli.', (df.gt_foreground_fraction > .5) & (df.tn / (df.tn + df.fp) < .5)),
        ('missed_watermark', 'Pominięcie większości watermarku', 'Recall watermarku < 50%.', df.foreground_recall < .5),
        ('false_positive_majority', 'Większość zaznaczenia trafia w czyste miejsca', 'Niepusta predykcja i precision watermarku < 50%.', (df.predicted_foreground_fraction > 0) & (df.foreground_precision < .5)),
        ('zero_overlap', 'Ani jeden piksel watermarku nie został trafiony', 'TP = 0, przy niepustej masce GT.', df.tp == 0),
        ('wrong_area', 'Duży błąd powierzchni maski', 'Bezwzględna różnica powierzchni predykcji i GT > 10 punktów procentowych obrazu.', df.absolute_area_error > .1),
        ('small_failure', 'Bardzo słaba lokalizacja małych watermarków', 'GT <= 5% obrazu i IoU klasy watermark < 0,10.', (df.gt_foreground_fraction <= .05) & (df.iou_watermark < .1)),
        ('empty', 'Całkowicie pusta predykcja', 'Zero pikseli dodatnich przy niepustej masce GT.', df.predicted_foreground_fraction == 0),
    ]
    summaries = []
    for key, title, criterion, selected in rules:
        df[key] = selected
        summaries.append(dict(key=key, title=title, criterion=criterion, count=int(selected.sum()),
                              denominator=510, percent=float(selected.mean() * 100),
                              by_method={m: int(selected[df.watermark_method == m].sum()) for m in METHODS}))
    summaries.sort(key=lambda r: -r['count'])
    review = json.loads((HERE / 'visual_review.json').read_text(encoding='utf-8'))
    gallery = json.loads((OUT / 'examples/index.json').read_text(encoding='utf-8'))
    assert len(review) == len({r['image_id'] for r in review}) == 30
    index = {r['name']: r for r in gallery}
    for r in review:
        r['panel'] = index[r['image_id']]['file']
        assert (OUT / 'examples' / r['panel']).exists()
    result = dict(scope='Retained best submission #1202; OOF proxy, not full-train resubstitution or hidden test.',
                  unique_images=510, visually_reviewed=30, categories_overlap=True,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  verification='Counts, IoU, areas and internal boundaries verified against saved OOF probabilities and GT.',
                  small_total=int((df.gt_foreground_fraction <= .05).sum()),
                  large_total=int((df.gt_foreground_fraction > .5).sum()),
                  ranking=summaries, visual_review=review)
    (OUT / 'frequency_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    df.to_csv(OUT / 'frequency_per_image.csv', index=False)
    lines = ['# Częstość problemów obecnego najlepszego modelu', '',
             'Zakres: zachowany najlepszy wariant #1202 — PixelSeal/TrustMark original seed42 oraz StegaStamp stable seed42. Nowszy eksperyment #1299 nie zastąpił go według raportu decoder_evidence. Oceniamy zapisane predykcje 3-fold OOF dla 510 unikalnych obrazów (170 na metodę). Każdy obraz oceniano modelem, który nie trenował na tym obrazie. To rozwojowy zbiór walidacyjny, nie wynik końcowego checkpointu na ukrytym teście.', '',
             'Obejrzano pojedynczo 30 paneli RGB / GT / probability / prediction / błędy: 27 wybranych deterministycznie (najgorszy, środkowy i najlepszy mIoU w każdej metodzie i grupie powierzchni <=5%, >5–50%, >50%) oraz po jednej środkowej według mIoU pustej predykcji na metodę. Dobór celowy, nie losowy; częstości policzono na wszystkich 510 obrazach, nie na tych 30.', '',
             'Kategorie są nakładającymi się objawami, a nie rozłącznymi przyczynami. Progi są jawnymi kryteriami tego audytu, nie oficjalnymi kategoriami konkursu. Kolejność oznacza częstość przy tych kryteriach, nie wagę straty. Nie należy sumować procentów.', '',
             '| Problem | Obrazy / 510 | Odsetek | PixelSeal / 170 | TrustMark / 170 | StegaStamp / 170 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        counts = r['by_method']
        lines.append(f"| {r['title']} | {r['count']} | {r['percent']:.1f}% | {counts['pixelseal']} | {counts['trustmark']} | {counts['stegastamp']} |")
    lines += ['', '## Definicje i interpretacja', '']
    for r in summaries:
        lines.append(f"- **{r['title']}:** {r['criterion']}")
    lines += ['', 'Wśród 255 obrazów z watermarkiem zajmującym ponad połowę obrazu 227 (89,0%) traci większość czystych pikseli. Dotyczy to zarówno otworów, jak i czystych obszarów przy krawędzi — nie wszystkie są zamkniętymi dziurami. Wśród 133 małych watermarków 124 (93,2%) ma IoU watermarku poniżej 0,10. Z 129 przypadków bez trafienia 53 to puste maski, a 76 to zaznaczenie wyłącznie błędnych miejsc.', '',
              'Błąd granic obejmuje również brak maski i zaznaczenie całego obrazu; nie oznacza 363 przypadków samego rozmycia poprawnej lokalizacji. Podobna powierzchnia predykcji i GT nie gwarantuje poprawnego położenia (np. 6d05e72e601c i e300d14d9ad9). W oglądanych panelach zdarzają się poprawne lokalizacje z niedokładną krawędzią; nie wszystkie obrazy są porażką.', '',
              '## Przegląd obrazów jeden po drugim', '', '| Obraz | Obserwacja wizualna |', '|---|---|']
    for r in review:
        lines.append(f"| [{r['image_id']}](results/examples/{r['panel']}) | {r['note']} |")
    lines += ['', '## Odtworzenie i weryfikacja', '',
              'Uruchom z katalogu repo: `task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/frequency_audit.py`, następnie `task-1/.venv/Scripts/python.exe task-1/dashboard/build.py`.', '',
              'Skrypt weryfikuje TP/FP/FN/TN, mIoU, powierzchnie i granice bezpośrednio względem zapisanych probabilities i GT. Nie zmienia modelu ani progów. [Pełne flagi per obraz](results/frequency_per_image.csv), [liczebności i zapis przeglądu](results/frequency_audit.json).', '',
              'Przyczyny takie jak wpływ STN, globalny kontekst i kalibracja wymagają osobnych eksperymentów; nie przypisano im częstości na podstawie samego wyglądu masek.']
    (HERE / 'FREQUENCY_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({r['key']: r['count'] for r in summaries}))
    print('Verified 510 OOF masks and 30 visual-review records.')


if __name__ == '__main__':
    main()
