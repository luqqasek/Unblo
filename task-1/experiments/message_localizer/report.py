"""Build source-backed experiment report from fixed-map diagnostics and paired OOF."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;R=P/'results'
d=json.loads((R/'comparison.json').read_text())
lines=['# PixelSeal i TrustMark: ekstraktor trenowalny i znana wiadomość','',
'Cel: poprawa walidacyjnego mIoU względem modeli użytych w zgłoszeniu #2008. Nie stroimy modeli na leaderboardzie.','',
'## Wyniki na 170 rzeczywistych obrazach na metodę','',
'| Metoda | Dotychczasowy ensemble | Nowy synthetic-only | Średnia 50/50 | Nowy po adaptacji OOF | Adaptacja + stary 50/50 |',
'|---|---:|---:|---:|---:|---:|']
for f,v in d.items():
 def value(k):return f"{v['variants'][k]['miou']:.5f}" if k in v['variants'] else 'w toku'
 lines.append(f"| {f} | {v['incumbent_miou']:.5f} | {value('synthetic_only')} | {value('synthetic_only_fixed_equal_ensemble')} | {value('adapted')} | {value('adapted_fixed_equal_ensemble')} |")
lines+=['','## Niepewność i lokalizacja','',
'| Metoda / wariant | Delta vs stary | Parowany bootstrap 95% | FG IoU | Clean IoU | FG IoU ≤1% | FG IoU 1–5% |',
'|---|---:|---|---:|---:|---:|---:|']
for f,v in d.items():
 for name,r in v['variants'].items():
  g=r['groups'];lo,hi=r['paired_bootstrap95']
  vals=[g['all']['fg_iou']['mean'],g['all']['clean_iou']['mean'],g['fg_le_1pct']['fg_iou']['mean'],g['fg_1_to_5pct']['fg_iou']['mean']]
  lines.append(f"| {f} / {name} | {r['delta_vs_incumbent']:+.5f} | [{lo:+.5f}, {hi:+.5f}] | "+' | '.join(f'{x:.5f}' for x in vals)+' |')
lines+=['','Bootstrap5000 po obrazach, z parowaniem tych samych nazw; warunkowy względem wytrenowanych modeli. Nie obejmuje ponownego treningu ani korekty za wielokrotne używanie danych rozwojowych. Zależność zdjęć źródłowych nie jest w pełni znana.','',
'## Diagnostyka bez treningu','',
'170 realnych obrazów/metodę, właściwe i błędne wiadomości. Macro AUC/AP liczone osobno dla każdego obrazu. Sam ranking nie daje gwarancji mIoU.','',
'| Metoda | Mapa | AUC correct | AUC wrong |','|---|---|---:|---:|']
for f,native in [('pixelseal',256),('trustmark',245)]:
 s=json.loads((R/f'signal/{f}/summary.json').read_text())['maps']
 for name in [f'dense_{native}_cos','dense_512_cos','window_128','window_256','gradient','reembed']:
  lines.append(f"| {f} | {name} | {s[name+'_correct']['macro_auc']:.5f} | {s[name+'_wrong']['macro_auc']:.5f} |")
lines+=['','Bezpośrednie mapy nie osiągnęły AUC0,80+. Gradient×HPF i re-embedding były bliskie losowym; nie włączono ich do nowego modelu. Kontrola zgodności GAP przestrzennych logitów z oryginalnym wrapperem: maksymalny błąd PS4,77e-6, TM5,72e-6.','',
'## Wykonany model i trening','',
'- Oryginalny pretrained extractor PixelSeal / TrustMark; nie wagi ImageNet. PixelSeal używa klasy ConvNeXtV2 z oficjalnego źródła checkpointu. TrustMark: oryginalny ResNet, wszystkie running stats BN zamrożone.',
'- Wspólne wagi ekstraktora dla wejścia512 i skali natywnej: PS256/TM245. Na każdym poziomie po projekcji128ch sumujemy cechy dwóch skal. Decoder progresywny do128.',
'- Mapy dot/cosine/soft-accuracy względem podanych bitów, na obu skalach. Dot ograniczony tanh(dot/4).',
'- Lekka gałąź RGB + Y + high-passRGB + JND. JND z oryginalnej funkcji na obserwowanym obrazie; to nie znana historyczna mapa na oryginalnym cropie sprzed edycji.',
'- Original embedders bez zmian.2500 zapisanych realizacji na rodzinę; źródła i wiadomości rozłączne z dev128. Dev ma64 stałe przykłady.',
'- Od początku nieregularne maski proceduralne, paski i dziury. Bez pełnej mieszanki COCO obiektów, transplantacji lub inpaintingu w tej iteracji.',
'-400aktualizacji×batch32=12800widoków.50kroków zamrożonego ekstraktora,350joint. LR ekstraktora2e-5, nowych modułów3e-4,20warmup,cosine do20%wartości początkowej; AdamW wd0,01; clipping1.',
'- Całość FP32. Per-image0,5BCE+0,5Dice obu klas, ranking par rampowany do0,1. Bez pomocniczego wymuszania wszystkich bitów w małym fragmencie.',
'- Synthetic-only: końcowe surowe wagi po400krokach, próg0,5, bez realnych etykiet w treningu.',
'- Adaptacja: te same3odtworzone oryginalne foldy,150kroków/fold,24synthetic+8real outer-train/batch32; LR2e-6/2e-5. Stały harmonogram i próg0,5, bez doboru na outer-validation.',
'- Ensemble: wcześniej zadeklarowana średnia prawdopodobieństw nowego modelu i starego ensemble trzech seedów; próg to średnia ich progów. Brak wyszukiwania wag na OOF.','',
'## Koszt i weryfikacja','',
'JURECA, jeden węzeł na zadanie, przydział 4×A100 40GB. Diagnostyka 15903994: 2m28s (używane 2GPU); pretraining 15904000: 9m11s; adaptacja 15904020: 10m30s. Wszystkie COMPLETED. Łącznie 22m09s czasu przydziałów, około 1,48 przydzielonej GPU-godziny; bez czasu kolejki i lokalnej analizy. Batch32, peak allocated memory około24,1GB PS i14,1GB TM.',
'',
'Sprawdzono rozłączność train/validation każdego foldu oraz pokrycie dokładnie170 unikalnych obrazów na rodzinę. Połączenie z baseline wymaga identycznych nazw i masek GT. Pełne mapy umożliwiają ponowne przeliczenie metryk.',
'',
'## Interpretacja','',
'To test całej receptury: inicjalizacja, dwie skale, message maps i generator zmieniły się łącznie. Nie identyfikuje przyczynowego wkładu pojedynczego komponentu. Synthetic-only, OOF po adaptacji i publiczny leaderboard to odrębne protokoły. Zbiór170/metodę był wielokrotnie używany rozwojowo, więc wynik nie jest bezstronną prognozą ukrytego testu.','',
'Kontrole AUC nie potwierdzają obietnic wzrostu SNR ani mIoU0,70–0,80. Nie jest to dowód niemożliwości takich wyników. Nie zmieniono działającego StegaStamp; dotychczasowe submission zachowano.','',
'## Artefakty i odtworzenie','',
'`results/comparison.json`, per-image JSON, mapyFP32/GT i konfiguracje w `results/runs/`; źródła, pełne checkpointy i optimizer state na JURECA: `/p/project1/training2666/pszenny1/native-localizer/message-runs/`.',
'','Ze środowiska task-1 na klastrze (w przydziale GPU):',
'```bash','python -m experiments.message_localizer.signal_job','python -m experiments.message_localizer.train_job','python -m experiments.message_localizer.adapt_job','```',
'','Analiza lokalna po pobraniu artefaktów:',
'```bash','PYTHONPATH=task-1:task-1/src task-1/.venv/bin/python -m experiments.message_localizer.compare --runs task-1/experiments/message_localizer/results/runs','task-1/.venv/bin/python task-1/experiments/message_localizer/report.py','```']
receipt_path=R/'submission/upload_receipt.json'
if receipt_path.exists():
 receipt=json.loads(receipt_path.read_text())
 response=receipt['response']
 if isinstance(response,str):response=json.loads(response)
 lines += ['', '## Zgłoszenie testowe', '',
 'Na wyraźne polecenie użytkownika wykonano refit obu nowych modeli na wszystkich170 obrazach/family:150kroków, stałe LR2e-6/2e-5, batch32=24synthetic+8real, próg0,5. Bez zmiany StegaStamp i bez strojenia na testowych etykietach.', '',
 f"Serwer: submission #{response.get('submission_id')}, status {response.get('status')}, public mIoU {response.get('score')}; poprzedni najlepszy0,5993. Jest to publiczny podzbiór, nie ukryty wynik finałowy.", '',
 'Plik `results/submission/submission-message-localizer.pt`:1020 unikalnych nazw, maski uint8[1020,128,128], binarne, poniżej20MB. Hash i potwierdzenie bezpiecznego odczytu w `assembly_validation.json`; źródła checkpointów w per-family inference JSON. Poprzednie zgłoszenie#2008 zachowano.', '',
 'Refit zakończony w zadaniu15904072. Pierwsza inferencja przerwana przez brak modułu submission.py w kopii na klastrze; po skopiowaniu modułu uruchomiono samą inferencję w zadaniu15904087, bez ponownego treningu.', '',
 'Refit i inferencja: `python -m experiments.message_localizer.refit_job` w przydziale4GPU. Łączenie: `PYTHONPATH=task-1:task-1/src task-1/.venv/bin/python -m experiments.message_localizer.assemble_submission`. Upload wykonany pojedynczym żądaniem, z kluczem w stdin; klucza nie zapisano w artefaktach.']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(P/'REPORT.md')
