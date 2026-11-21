"""Publish segmentation and auxiliary-decoding evidence."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    s=json.loads((HERE/'results/summary.json').read_text())
    lines=['# Lokalny odczyt wiadomości jako zadanie pomocnicze','',
           'Wariant inspirowany WAM: wspólna głowica 32×32, osobne wyjścia segmentacji i bitów. Frozen ekstraktory i podstawowy loss 0.5 BCE + 0.5 Dice pozostają bez zmian. Wiadomość jest wyłącznie celem treningowym; nie jest wejściem modelu.',
           'Pomocniczy BCE bitów ważony udziałem watermarku w komórce 32×32 (average pooling maski 128×128). Normalizacja osobno dla obrazu przez liczbę bitów i masę maski; następnie średnia obrazów. Czyste komórki nie dostają gradientu wiadomości; graniczne dostają wagę częściowego pokrycia. To adaptacja równania WAM, nie identyczna implementacja batchowej normalizacji jego kodu.',
           'Seed 42, AdamW 0.001, batch 8, maksymalnie 50 epok, patience 8. Wybór epoki i progu wyłącznie według segmentacji na wewnętrznym podziale. Następnie nowy trening na całym outer-train.',
           'Pierwszy fold: 57 obrazów na metodę. Do CV kwalifikuje poprawa łączna >=0.005 i poprawa w co najmniej dwóch metodach. Tylko najlepsza zakwalifikowana waga przechodzi do pełnego CV. To screening rozwojowy na znanym podziale, nie niezależny test.',
           'Waga 0 odtwarza istniejący baseline: jednakowe wagi początkowe segmentacji, kolejność batchy i wyniki; dodanie głowicy nie zmienia punktu odniesienia.',
           '[Research i uzasadnienie wyboru](RESEARCH.md).','', 'Stan: '+s['status'],'',
           '| Wariant | PixelSeal | TrustMark | StegaStamp | Łącznie | Zmiana | Próg CV |',
           '|---|---:|---:|---:|---:|---:|---|']
    for variant,data in s['variants'].items():
        v=[r['selected']['miou'] for r in data['screen']]
        status='kontrola' if variant=='control' else ('spełniony' if data['passed'] else 'niespełniony')
        lines.append(f'| {variant} | '+' | '.join(f'{x:.6f}' for x in v+[np.mean(v)])+f" | {data['screen_delta']:+.6f} | {status} |")
    lines+=['','## Czy model odczytuje wiadomość?','',
            'Trafność bitów: średnia ważona pokryciem watermarku, następnie średnia po obrazach. Kontrola shuffled porównuje te same predykcje z wiadomością kolejnego obrazu walidacyjnego; majority to stały bit większości wyznaczony wyłącznie z outer-train. Wiadomości kontrolne są użyte tylko diagnostycznie. Duże pole recepcyjne może rozgłaszać globalny odczyt na wiele komórek, więc sama trafność nie dowodzi lokalizacji.', '',
            '| Wariant | Metoda | Bity na watermarku | Shuffled | Majority train | Bity na czystym | BCE bitów | Epoka | Próg |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant,data in s['variants'].items():
        for r in data['screen']:
            d=r['diagnostics']
            vals=[d[k] for k in ('bit_accuracy_watermark','bit_accuracy_shuffled_message','training_majority_accuracy','bit_accuracy_clean','masked_bit_bce')]
            lines.append(f"| {variant} | {r['method']} | "+' | '.join(f'{x:.4f}' for x in vals)+f" | {r['epoch']} | {r['threshold']} |")
        if data['cv']:
            lines+=['',f'## CV: {variant}','','| Metoda | Fold 1 | Fold 2 | Fold 3 | Średnia | SD | OOF |','|---|---:|---:|---:|---:|---:|---:|']
            for method in ('pixelseal','trustmark','stegastamp','overall'):
                groups=[[r for r in data['cv'] if r['fold']==f and (method=='overall' or r['method']==method)] for f in (1,2,3)]
                vals=[np.mean([x for r in g for x in r['selected']['per_image']]) for g in groups]
                oof=np.mean([x for g in groups for r in g for x in r['selected']['per_image']])
                lines.append(f'| {method} | '+' | '.join(f'{x:.6f}' for x in vals+[np.mean(vals),np.std(vals,ddof=1),oof])+' |')
    if s['status']=='complete' and s.get('cv_variant') is None:
        lines+=['','## Wniosek','',
                'Żadna waga zadania pomocniczego nie przeszła screeningu. Pozostawiamy dotychczasową głowicę i pomijamy pełne CV. Sprawdzić trafność bitów w tabeli względem shuffled i majority: wynik bliski 50% nie potwierdza użytecznego odczytu wiadomości.',
                'To wynik tej adaptacji: losowo inicjalizowana głowica bitów, zamrożone cechy, mały zbiór i selekcja epoki według segmentacji. Nie obala wspólnego treningu embeddera i ekstraktora w WAM ani wykorzystania istniejącego, wytrenowanego dekodera bitów.']
    lines+=['','Checkpointy odtwarzają zapisane predykcje z tolerancją 1e-7. Nie użyto danych testowych i nie wysyłano submission.',
            'Odtworzenie: `task-1/.venv/Scripts/python.exe task-1/experiments/local_message/run.py`. Artefakty w results/: checkpointy, predykcje, diagnostyka i podsumowanie.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
