"""Measured results only; distinct inference, synthetic-only and adapted protocols."""
import json
from pathlib import Path
import numpy as np
from experiments.small_region_diagnosis.common import metadata,bucket,select,mean,save

def grouped(rows):
    meta=metadata();rs=[dict(r,bucket=bucket(r['true_area']),full_watermark=meta[r['image_name']]['full_watermark']) for r in rows]
    return {g:{'n':len(select(rs,g)),**{k:mean(select(rs,g),k) for k in ['miou','fg_iou','clean_iou','fg_ap','empty']}} for g in ['all','le1','1to5','5to20','gt20','clean_holes']}

def main():
    here=Path(__file__).resolve().parent;r=here/'results';measurements=[]
    for run,title in [('runs','NRGMark512'),('multiscale','NRGMark512+1536'),('wide','NRGMark512+1536+2560'),('finalscale','NRGMark512+1536+2560+4096')]:
        path=r/f'{run}_summary.json'
        if not path.exists():continue
        d=json.loads(path.read_text())
        for f,v in d.items():
            measurements.append({'family':f,'variant':title,'protocol':'Existing 3-fold OOF + frozen GT-free crop correction; development','groups':v['fixed_crop90'],'delta':v['paired_delta'],'source':str(path.relative_to(here))})
    for v in ['control','tiny','tiny_aux']:
        for stage,path in [('synthetic',r/f'tiny/{v}/real/per_image.json'),('adapt',r/f'tiny/{v}/adapt/per_image.json'),('screen',r/f'tiny/{v}/adapt/screen_per_image.json')]:
            if not path.exists():continue
            rows=json.loads(path.read_text());assert len(rows)==len({x['image_name'] for x in rows})==(57 if stage=='screen' else 170)
            measurements.append({'family':'pixelseal','variant':v,'protocol':{'synthetic':'Synthetic-only / 170 real images','adapt':'Fixed 3-fold adapted OOF / 170 real images','screen':'Single fixed fold1 screening / 57 real images'}[stage],'groups':grouped(rows),'source':str(path.relative_to(here))})
    save(r/'measurements.json',measurements)
    lines=['# Małe watermarki: diagnoza, crop-and-decode i trening PixelSeal','',
        'Aktualne bazowe OOF: PixelSeal0,585655 / TrustMark0,593332. Protokoły: pełne rozwojowe OOF170/metodę, transfer synthetic-only170 oraz szybki screening jednego ustalonego foldu57. Na prośbę użytkownika zrezygnowano z pełnego CV dla nowych prób PixelSeal.','',
        '| Metoda | Wariant | Protokół | mIoU | FG≤1% | FG1–5% | FG5–20% | FG>20% | Clean IoU holes |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for m in measurements:
        g=m['groups'];values=[g['all']['miou'],g['le1']['fg_iou'],g['1to5']['fg_iou'],g['5to20']['fg_iou'],g['gt20']['fg_iou'],g['clean_holes']['clean_iou']]
        lines.append('| '+m['family']+' | '+m['variant']+' | '+m['protocol']+' | '+' | '.join(f'{x:.5f}' for x in values)+' |')
    lines+=['','## Co faktycznie sprawdzono','',
        '- Odtworzono diagnozę najnowszych OOF z zapisanych FP32 map i sprawdzono rozłączność foldów. PS≤1% AUC0,4959/AP0,0057; TM≤1% AUC0,6415/AP0,0361. Oracle próg koszyka daje tylko FG IoU0,0064/0,0195. To reprezentacja/propozycje, nie wyłącznie próg0,5.',
        '- Oficjalny zamrożony NRGMark-Q dostarcza propozycje bez GT, typu edycji i powierzchni GT. Dla PS jawnie testowano transfer detektora TM. Każda propozycja przechodzi ograniczone wyszukiwanie rozmiaru/położenia i odczyt przez NIEZMIENIONY oficjalny decoder konkurencji. Wiadomość to surowe100/256 bitów, bez ECC. Akceptacja>=90% bitów ustalona przed oceną; korekta jest sumą masek baseline i najlepszego prostokąta.',
        '- Skalowanie dotyczy wyłącznie detektora propozycji. Crop decoder czyta zawsze oryginalne piksele512. Skala1536/2560/4096 jest następczym testem dopasowania do treningu NRGMark na regionach rzędu256px; nie jest dowodem niezmienniczości watermarka na resampling.',
        '- NRGMark512+1536+2560: TM poprawił23/170 obrazów, pogorszył0, pozostałe147 bez zmiany. Fold1:0,56318→0,63317; fold2:0,60087→0,63504; fold3:0,61635→0,66781. Random-message correction nie zaakceptowała regionów. Najmniejsze≤1% pozostają niewykryte w tej wersji.',
        '- Trening PS:512 nowych osadzeń train +64dev na dotychczasowych rozłącznych źródłach, dokładne małe prostokąty, brak wzmacniania/JPEG/blur. Trzy warianty, identyczny punkt startowy i200aktualizacji×32widoki. Pomocnicza głowa przewiduje16subpikseli residualuY na komórkę128, zachowując rozdzielczość512. Nie jest wejściem inferencji. Pełna receptura w PLAN.md.',
        '- Pierwszy start PS przekroczył40GB przed pierwszym optimizer step. Poprawiono microbatch16 + akumulacja do32; zgodność gradientów segmentacji i par sprawdzona testem. Nie zmieniono precyzji FP32 ani ekspozycji danych.','',
        '## Ocena researchu i ograniczenia','',
        'Raport RTX3070 dostarcza przydatnych dowodów, ale odczyt z GT-bbox nie jest lokalizacją bez GT. Stride32 nie opisuje całego obecnego modelu: ma on skipy stride4 i lokalny stem stride1. BCE nie ma automatycznie100× mniejszego gradientu na pojedynczym dodatnim pikselu; problem dotyczy łącznej masy klas. Zerowy residual po uint8 nie czyni maski czystą. Lokalny PixelSeal ma dominujący Y, ale chroma po kwantyzacji nie musi być idealnie zerowa.',
        'Nie ma gwarancji0,70 ani pewnego rozstrzygnięcia po jednym pomiarze. Dla25% obrazów poprawa FG IoU o0,48 wnosi około0,06 do wyniku jednej rodziny przy niezmienionej klasie clean; wpływ na wynik trzech rodzin jest odpowiednio mniejszy.',
        'Foldy są wielokrotnie używanymi danymi rozwojowymi. Kolejne skale dobrano adaptacyjnie po poprzednich pomiarach. Nie nazywać wyniku niezależnym testem ani prognozą leaderboardu. Bootstrap paruje obrazy, lecz nie obejmuje ponownego treningu i pełnej zależności źródeł. Próba random-message to kontrola diagnostyczna, nie formalny certyfikat false-positive rate.','',
        '## Źródła i odtworzenie','',
        '- [NRGMark, oryginalny artykuł](https://collomosse.com/pubs/Agarwal-WACV-2026.pdf): region proposals i osobny crop decoder; ograniczenia małych rozdzielczości.',
        '- [Kod Adobe TrustMark](https://github.com/adobe/trustmark/tree/4c60b9fdb4125fca3001f0803e6890a6b55371ac): odizolowany detector, bez aktualizacji competition trustmark0.8.0.',
        '- [TruFor](https://grip-unina.github.io/TruFor/): RGB + Noiseprint++ fusion; ogólny research nie opisuje poprawnie jego architektury jako po prostu SRM+Bayar.',
        'Na klastrze w Slurm: `python -m experiments.small_zoom.proposals --root ROOT --family trustmark --scales 512 1536 2560 --run-name small-zoom-wide`; trening: `python -m experiments.small_zoom.tiny_job`, adaptacja: ten sam moduł z `--adapt`. Lokalnie: `python -m experiments.small_zoom.diagnose`, `evaluate_proposals --runs ... --families trustmark`, `report`.',
        'Artefakty: ROOT/small-zoom-{runs,multiscale,wide,finalscale}, ROOT/tiny-rect-data, ROOT/tiny-runs. Baseline checkpointy i stare predykcje pozostają zachowane.']
    lines+=['','## Wniosek z krótkiego testu residualu PixelSeal','',
        'Po200aktualizacjach synthetic-only: kontrola0,563236; małe prostokąty0,559574; te same dane+residualY0,559587. FG≤1% pozostaje0. Na48 częściowo oznakowanych przykładach tiny-dev FG IoU jest0,00302, mimo ogólnego dev mIoU0,6195 z łatwymi clean controls. Nie jest to udana nauka lokalizacji.',
        'Na niezerowych residualach dev MAE głowy wynosi1,378549 poziomu uint8, wobec1,378558 dla predykcji samych zer. Ta konkretna słaba L1 auxiliary praktycznie zapadła się do zera. Wynik nie dowodzi nieskuteczności wszystkich form nadzoru residualu ani fundamentalnego ograniczenia PixelSeal.','',
        '## Zamrożona receptura nowego zgłoszenia','',
        'PS: sprawdzona syntetyczna kontrola600 +150aktualizacji refitu wszystkich170 z replay. TM: obiektowy checkpoint600 + taki sam refit150, następnie propozycje512/1536/2560/4096 i próg zgodności bitów0,9. SS: identyczne maski jak#2163. PixelSeal z nowej krótkiej próby nie jest automatycznie wdrażany. Każdy refit używa batch32/micro16,24synthetic+8real, LR2e-6/2e-5, progu0,5.']
    lines+=['','## Gdzie dokładnie działa nowa gałąź TrustMark','',
        'W grupie1–5% wszystkie19 niemal prostokątnych GT (fill>0,99) były wcześniej całkowicie pomijane: średnie FG IoU0,0000. Po propozycjach512/1536/2560/4096 i weryfikacji wiadomości: **0,8192**. Dla17 nieregularnych GT wynik pozostaje **0,1517**; nie zaakceptowano tam nowych prostokątów. To analiza następcza według GT, nie reguła wyboru stosowana podczas inference.',
        'Dla≤1% odzyskano1 z8 prostokątów, a jedyny nieregularny przypadek pozostał niewykryty. Wniosek: gałąź poprawia zgodne ze skalą lokalne prostokątne osadzenia; nie rozwiązuje ogólnie pozostałości po transplantacji ani najmniejszych znaków.']
    receipt=r/'submission/server_receipt.json'
    if receipt.exists():
        result=json.loads(receipt.read_text());response=result['response'];response=json.loads(response) if isinstance(response,str) else response
        lines+=['',f"Potwierdzenie serwera: **#{response.get('submission_id')}, status {response.get('status')}, publiczny mIoU {response.get('score')}**. Poprzedni najlepszy: {response.get('previous_best')}. To wynik publicznego podzbioru, nie ukrytego finału.",'','[Receipt](results/submission/server_receipt.json) · [Walidacja artefaktu](results/submission/assembly_validation.json).']
    else:lines+=['','Zgłoszenie nie jest jeszcze potwierdzone. Ostatni potwierdzony publiczny wynik pozostaje#2163=0,6145.']
    provenance=r/'execution_provenance.json'
    if provenance.exists():
        execution=json.loads(provenance.read_text())
        if 'allocated_gpu_seconds' in execution:
            lines+=['','## Koszt wykonania','',f"Slurm: {execution['allocated_gpu_seconds']/3600:.3f} przydzielonych GPU-godzin w tej serii, łącznie z przygotowaniem danych, inferencją, refitami i nieudanymi startami. Scheduler przydzielał całe węzły4×A100 nawet przy żądaniu1–3GPU; faktyczne procesy korzystały z1–3GPU. Nie jest to pomiar czasu aktywności jąder. Szczegóły jobów w execution_provenance.json.",
                'PS: trzy ramiona po200aktualizacji wykonały się równolegle w6min6s przydziału; screening jednego foldu w4min37s. NRGMark cztery skale:110,69s dla170obrazów walidacyjnych; test340z bazowym segmentatorem230,66s. Refit150: TM181,58s, PS229,43s samej pętli. Peak PS micro16:24,45GB; NRGMark cztery skale:7,95GB.']
    (here/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('measured rows',len(measurements))
if __name__=='__main__':main()
