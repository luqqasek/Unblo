import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    s=json.loads((HERE/'results/summary.json').read_text())
    lines=['# Wytrenowane dekodery i zgodność ze znaną wiadomością','',
        'Celem jest wykorzystanie informacji z wiadomości do lokalizacji, bez uczenia dekodowania od zera. Wszystkie cechy powstają bez masek GT.',
        'PixelSeal: ostatnia mapa ConvNeXt → oryginalny pixel_decoder.output_upscaling → oryginalna warstwa linear zastosowana w każdej pozycji. TrustMark: ostatnia mapa ResNet → oryginalna fc w każdej pozycji. Średnia przestrzenna logitów jest sprawdzana względem prawdziwego globalnego odczytu dla wszystkich 170 obrazów. To projekcje globalnie trenowanego dekodera, nie gwarancja lokalności.',
        'Mapy PS/TM: średnia sigmoid logitu zgodnego ze znanym bitem, trafność znaku, średnia zgodnego logitu podzielona przez średni moduł logitów oraz globalna zgodność. Brak maskowania GT podczas ekstrakcji.',
        'StegaStamp: zasłonięcie kolejno 16 obszarów 128×128 w oryginalnym obrazie 512×512 średnim kolorem tego obrazu. Dekoder pozostaje w całości zamrożony. Mapy opisują spadek zgodności sigmoid i log-prawdopodobieństwa, globalną zgodność oraz trafność bitów po zasłonięciu. To wrażliwość na interwencję, nie prawdopodobieństwo lokalnej obecności; zasłanianie może wprowadzać artefakty.',
        'map_only: mała głowica uczona tylko na czterech mapach. residual: istniejąca głowica + uczona korekta z map, początkowo zerowa. residual_wrong: identyczna architektura, ale wiadomości cyklicznie zamienione między obrazami; kontrola dodatkowej pojemności i informacji niezwiązanej ze znajomością poprawnej wiadomości.',
        'Kontrfaktyczna ocena oznacza podmianę poprawnej wiadomości na błędną tylko na wejściu wytrenowanego modelu; dla residual_wrong odwrotnie. Nie zmieniamy przy tym progu.',
        'blend: średnia ważona prawdopodobieństw bazowej głowicy i map_only. Waga z [0,.25,.5,.75,1] i próg z [.3,.4,.5,.6,.7] dobierane wyłącznie na odtworzonych modelach wewnętrznych. Następnie te same ustawienia łączą modele wytrenowane na całym outer-train. Wariant dodany po obejrzeniu screeningu map_only; to dodatkowy element selekcji rozwojowej.',
        'Drugi etap po słabym wyniku zbiorczych map: bit_map, bit_residual i bit_residual_wrong zachowują każdą z 256/100 map bitów osobno. PS/TM: sigmoid logitu pomnożonego przez znak oczekiwanego bitu. StegaStamp: tanh spadku log-prawdopodobieństwa danego bitu /5 po zasłonięciu. Architektury i kontrole analogiczne do pierwszego etapu; wybór po screeningu również obciąża ocenę selekcją.',
        'Screening: pierwszy zapisany fold, 57 obrazów na metodę. Seed 42, batch 8, AdamW .001, 50 epok / patience 8, loss .5 BCE + .5 Dice. Epoka i próg wybrane wyłącznie na wewnętrznym podziale; następnie trening od nowa na całym outer-train.',
        'Próg CV ustalony przed wynikami: dla danej metody poprawa >=.005 nad baseline oraz >=.005 nad kontrfaktyczną błędną wiadomością. Korekta residual musi dodatkowo przewyższyć model trenowany z błędną wiadomością. CV tylko najlepszego wariantu spełniającego warunki dla danej metody. Baseline to identyczny wcześniejszy protokół frozen_extractor/cv3, nie późniejszy wariant stabilnej selekcji StegaStamp.',
        'To porównanie rozwojowe; CV zawiera fold screeningowy i jest obciążone wyborem wariantu. Do selekcji wariantów i pomiaru CV nie użyto testu konkursowego ani leaderboardu.','', 'Stan: '+s['status'],'',
        '| Metoda | Wariant | Baseline | mIoU | Zmiana | Kontrfaktyczna wiadomość | Epoka | Próg |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for method,data in s['methods'].items():
        base=s['baseline']['methods'][method]['folds'][0]['selected']['miou']
        for row in data['runs']:
            score=row['selected']['miou']
            lines.append(f"| {method} | {row['variant']} | {base:.6f} | {score:.6f} | {score-base:+.6f} | {row['counterfactual']['miou']:.6f} | {row['epoch']} | {row['threshold']} |")
    for method,data in s['methods'].items():
        lines+=['',f'## {method}','',f"Globalna trafność bitów zamrożonego dekodera: {data['global_bit_accuracy']:.4f}. Rodzaj map: {data['kind']}. Maksymalny błąd odtworzenia projekcji: {data['projection_max_error']}.",
                f"Wariant do CV: {data['winner'] or 'brak'}."]
        if len(data['cv'])==3:
            vals=[r['selected']['miou'] for r in data['cv']]
            all_values=[x for r in data['cv'] for x in r['selected']['per_image']]
            base=s['baseline']['methods'][method]['oof_miou']
            lines += [f"Foldy: {', '.join(f'{v:.6f}' for v in vals)}. Średnia {np.mean(vals):.6f}; SD {np.std(vals,ddof=1):.6f}. OOF {np.mean(all_values):.6f}, zmiana względem baseline {np.mean(all_values)-base:+.6f}."]
    if (HERE/'results/stable_summary.json').exists():
        stable=json.loads((HERE/'results/stable_summary.json').read_text())
        lines+=['','## Porównanie z nowszą stabilną selekcją StegaStamp','',
                'Ta sama procedura trzech stratyfikowanych foldów wewnętrznych co stegastamp_stable_selection, seed 42. Zmieniono wyłącznie model przez dodanie korekty z pretrained informacji bitowej. Screening na foldzie 1: delta >=.005 i przewaga nad błędną wiadomością >=.005, następnie pozostałe foldy. Wariant wybrany po poprzednich analizach; to nadal porównanie rozwojowe.',
                '| Fold | Aktualny baseline | Wiadomość + stabilna selekcja | Zmiana | Błędna wiadomość |',
                '|---|---:|---:|---:|---:|']
        for r in stable['runs']:
            lines.append(f"| {r['fold']} | {r['baseline']:.6f} | {r['selected']['miou']:.6f} | {r['delta']:+.6f} | {r['counterfactual']['miou']:.6f} |")
        lines.append(f"Stan: {stable['status']}; próg screeningu: {stable['passed']}.")
        if len(stable['runs'])==3:
            a=[r['selected']['miou'] for r in stable['runs']]; b=[r['baseline'] for r in stable['runs']]
            lines.append(f"Średnia foldów: {np.mean(b):.6f} → {np.mean(a):.6f}, zmiana {np.mean(a)-np.mean(b):+.6f}; SD nowego wariantu {np.std(a,ddof=1):.6f}.")
            oof=np.mean([x for r in stable['runs'] for x in r['selected']['per_image']])
            lines+=['',f"OOF (obrazy ważone równo): {oof:.6f}. Poprawa w {sum(x>y for x,y in zip(a,b))}/3 foldach.",
                    'Wniosek: użytecznym wariantem dla StegaStamp jest korekta z per-bit wrażliwością wytrenowanego dekodera na zasłanianie, połączona ze stabilną selekcją. PixelSeal i TrustMark nie uzyskały potwierdzonej poprawy w screeningu, więc zachowują bazową głowicę.',
                    'To wyniki jednego seedu 42 na wcześniej używanych foldach. Podmiana wiadomości pokazuje jej znaczenie dla predykcji, ale nie izoluje wkładu każdego kanału ani nie dowodzi dokładnej lokalności dekodera. Wyniki CV należy odróżniać od osobnej oceny leaderboardu poniżej.']
    full=HERE/'results/full_train'
    if (full/'training_summary.json').exists():
        train=json.loads((full/'training_summary.json').read_text())
        lines+=['','## Trening końcowy i predykcje testowe','',
                f"Model bit_residual StegaStamp wytrenowany na wszystkich {train['train_count']} obrazach treningowych, seed {train['seed']}, {train['epochs']} epok, próg {train['threshold']}. Epoki i próg to mediany z trzech foldów stabilnego CV; ustalone przed predykcjami testowymi.",
                f"Checkpoint: `results/full_train/stegastamp_head.pt`; SHA256 `{train['checkpoint_sha256']}`."]
        if (full/'prediction_summary.json').exists():
            pred=json.loads((full/'prediction_summary.json').read_text())
            lines += [f"Wygenerowano submission dla {pred['test_count']} obrazów: nowy model dla {pred['updated_stegastamp_images']} StegaStamp, {pred['unchanged_other_images']} predykcji PixelSeal/TrustMark identycznych z submission #1202. Sprawdzono nazwy, rozmiary, binarność, zgodność po ponownym załadowaniu checkpointu i niezmienność pozostałych metod.",
                      f"Plik `results/full_train/submission.pt`, {pred['bytes']} bajtów, SHA256 `{pred['submission_sha256']}`. Udział pikseli dodatnich StegaStamp: {pred['watermark_fraction']:.4f}. Globalna trafność bitów na teście: {pred['global_bit_accuracy']:.4f}; nie jest to metryka lokalizacji."]
        else:
            lines.append('Generowanie predykcji testowych w toku.')
        if (full/'upload_receipt.json').exists():
            receipt=json.loads((full/'upload_receipt.json').read_text())
            response=json.loads(receipt['response'])
            if response.get('status')=='success':
                previous=json.loads(json.loads((HERE.parent/'stegastamp_stable_selection/results/full_train/upload_receipt.json').read_text())['response'])
                lines+=['','## Publiczny leaderboard','',
                        f"Zaakceptowane submission #{response['submission_id']}: publiczne mIoU **{response['score']:.6f}**, poprzednie #1202 **{previous['score']:.6f}**, zmiana **{response['score']-previous['score']:+.6f}**.",
                        'To wynik całego zgłoszenia na ukrytej publicznej części testowej, nie mIoU samego StegaStamp ani kolejny wynik CV.']
                if response['score'] < previous['score']:
                    lines.append('Wniosek po wysyłce: poprawa rozwojowego CV nie przełożyła się na publiczny test. Ten wariant nie zastępuje najlepszego zgłoszenia #1202. Sam wynik nie rozstrzyga, czy przyczyną jest dobór wariantu na znanych foldach, zmienność treningu czy różnica rozkładu danych.')
            else:
                lines.append(f"Wysyłka bez potwierdzonej akceptacji: HTTP {receipt['http_status']}; zobacz upload_receipt.json.")
        else:
            lines.append('Brak potwierdzonej wysyłki tego wariantu.')
    lines+=['','Odtworzenie: `prepare.py` generuje zamrożone cechy; `run.py` wykonuje screening i warunkowe CV; `stable.py` sprawdza nowy wariant selekcji; `train_full_predict.py` trenuje końcową głowicę i zapisuje kompletne zgłoszenie testowe. Raport i dashboard aktualizują się po kolejnych wynikach. Checkpointy odtwarzają zapisane predykcje z tolerancją 1e-7.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
