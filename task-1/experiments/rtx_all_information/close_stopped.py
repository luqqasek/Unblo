"""Close the user-stopped experiment using only fully completed comparisons."""
import json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
rows=[];results={};aux=read(OUT/'auxiliary_diagnostics.json')['families']
lines=['# Wszystkie informacje — eksperyment zatrzymany na żądanie użytkownika','','**Brak potwierdzonej poprawy głównej lokalizacji w ukończonych próbach.** PixelSeal i TrustMark zakończyły pełne porównanie 3-fold. StegaStamp zatrzymano podczas przygotowania danych; nie rozpoczął treningu. To wynik dla 340 rzeczywistych obrazów, nie pełne porównanie 510 obrazów.','','| Metoda | N | Kontrola mIoU | Wszystkie informacje | Zmiana | 95% bootstrap zmiany |','|---|---:|---:|---:|---:|---|']
for family in ('pixelseal','trustmark'):
    root=OUT/'runs'/family;r=read(root/'completed.json');results[family]=r
    c=r['control']['all']['miou'];f=r['all_information']['all']['miou'];ci=r['conditional_bootstrap95']
    lines.append(f"| {family} | 170 | {c:.6f} | {f:.6f} | {r['delta']:+.6f} | [{ci[0]:+.6f}; {ci[1]:+.6f}] |")
    for arm in ('control','all_information'):
        rr=read(root/arm/'completed.json');folds=[read(root/arm/f'fold{k}/completed.json') for k in (1,2,3)]
        rows.append({'family':family,'arm':arm,'miou':rr['summary']['all']['miou'],'n':170,'folds':[x['summary']['all']['miou'] for x in folds]})
lines+=['','Oba przedziały obejmują zero. To jeden seed i wielokrotnie używane foldy rozwojowe. Nie ma dowodu poprawy; nie jest to dowód, że żadna architektura nie może skorzystać z dodatkowych informacji.','','## Najmniejsze obszary','','| Metoda | Typ ≤1% | N | Całkowite pominięcia: kontrola | Całkowite pominięcia: pełny wariant |','|---|---|---:|---:|---:|']
for family,r in results.items():
    for g,miss in [('fg_le1','fg_miss'),('clean_le1','clean_miss')]:
        c=r['control'][g];f=r['all_information'][g];lines.append(f"| {family} | {g} | {c['n']} | {c[miss]} | {f[miss]} |")
lines+=['','## Czy dodatkowe sygnały zostały nauczone?','','| Metoda | Trafność full_watermark na realnym OOF | mIoU pomocniczej maski512 | MAE energii na syntetycznym dev | MAE predykcji zerowej |','|---|---:|---:|---:|---:|']
for family,d in aux.items():
    r=d['real_oof'];s=d['synthetic_dev'];lines.append(f"| {family} | {r['regime_correct']:.4f} | {r['aux512']['all']['miou']:.6f} | {s['energy_mae']:.6f} | {s['energy_zero_mae']:.6f} |")
lines+=['','Energię oceniono w skali log1p na 200 syntetycznych obrazach rozwojowych na metodę. Lepsze zadania pomocnicze nie przełożyły się na poprawę głównej maski. Ponowna inferencja checkpointów FP32 odtworzyła główne mIoU dokładnie.','','## Zakres i sprzęt','','Kompaktowy model z zamrożonymi cechami oryginalnego dekodera; nie pełny fine-tuning najlepszego dużego modelu. Każda ukończona metoda: 4000 syntetycznych realizacji treningowych, 200 syntetycznych dev i wszystkie 170 realnych obrazów w trzech foldach. Dwie pełne epoki syntetyczne, 20 epok na części treningowej każdego foldu, seed42, próg0.5. Wszystkie kwalifikujące się realne obrazy odwiedzone w każdej epoce.','','Wykorzystane: RGB512, metoda, wszystkie bity wiadomości, oryginalne cechy dekodera i zgodność bitów, maski128/512, granice, odległości i centra składowych, pokrycie, full_watermark oraz dokładne pary syntetyczne czysty/oznaczony. Nazwy i hashe służą identyfikacji i kontroli podziałów. Etykiety i czyste odpowiedniki nie są wejściami przy predykcji. Nie użyto testu do treningu.','',f"RTX3070: szczyt pamięci tensorów trenera {max(r['peak_gpu_bytes'] for r in results.values())/1e9:.3f} GB. Łączny czas obu ukończonych porównań modelowych {sum(r['seconds'] for r in results.values())/60:.1f} min, bez przygotowania danych. Cechy są cache’owane; ta pamięć nie oznacza kosztu pełnego fine-tuningu dużego modelu ani całej pamięci nvidia-smi.",'','## Pochodzenie danych i zakończenie','','Syntetyczne obrazy odtworzono oryginalnymi wrapperami. Hashe RGB różnią się od historycznych w891/4200 przypadków PixelSeal i3043/4200 TrustMark; źródło, wiadomość i maskę zweryfikowano, a cechy policzono na aktualnych pikselach. Oba warianty otrzymały te same odtworzone obrazy. Nie deklarujemy identyczności z historycznymi cache.','','StegaStamp: przygotowanie przerwane, bez pomiaru jakości ani treningu. Zachowano ukończone checkpointy, predykcje i częściowy cache. Nie wykonano submission. Procesy przygotowania i koordynatora zakończono oraz sprawdzono ich brak.','','Aktualna baza rozwojowa poza tym eksperymentem osiągnęła następnie publiczne0.6677 (#3054). PS/TM pozostały bez zmian względem#2675/#3005; publicznych wyników nie porównujemy bezpośrednio z powyższym CV.','','[Protokół](PROTOCOL.md) · [Pomiary](results/measurements.json) · [Diagnostyka](results/auxiliary_diagnostics.json)','']
(OUT/'measurements.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
status={'status':'stopped_at_user_request','at':datetime.now(timezone.utc).isoformat(),'completed_families':list(results),'evaluated_real_images':340,'stegastamp_training_started':False,'stegastamp_preparation':read(OUT/'data/stegastamp/progress.json')}
(OUT/'stopped.json').write_text(json.dumps(status,indent=2),encoding='utf-8');(HERE/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(status))
subprocess.run([sys.executable,str(HERE/'install_dashboard.py')],check=True)
subprocess.run([sys.executable,str(HERE/'export_results.py')],check=True)
