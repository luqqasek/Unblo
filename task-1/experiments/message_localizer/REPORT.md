# PixelSeal i TrustMark: ekstraktor trenowalny i znana wiadomość

Cel: poprawa walidacyjnego mIoU względem modeli użytych w zgłoszeniu #2008. Nie stroimy modeli na leaderboardzie.

## Wyniki na 170 rzeczywistych obrazach na metodę

| Metoda | Dotychczasowy ensemble | Nowy synthetic-only | Średnia 50/50 | Nowy po adaptacji OOF | Adaptacja + stary 50/50 |
|---|---:|---:|---:|---:|---:|
| pixelseal | 0.55899 | 0.51958 | 0.56286 | 0.57192 | 0.57084 |
| trustmark | 0.54745 | 0.55929 | 0.56592 | 0.57872 | 0.57151 |

## Niepewność i lokalizacja

| Metoda / wariant | Delta vs stary | Parowany bootstrap 95% | FG IoU | Clean IoU | FG IoU ≤1% | FG IoU 1–5% |
|---|---:|---|---:|---:|---:|---:|
| pixelseal / synthetic_only | -0.03941 | [-0.05767, -0.02070] | 0.46877 | 0.57040 | 0.00000 | 0.02024 |
| pixelseal / synthetic_only_fixed_equal_ensemble | +0.00387 | [-0.00416, +0.01183] | 0.53322 | 0.59249 | 0.00000 | 0.02931 |
| pixelseal / adapted | +0.01292 | [-0.00044, +0.02702] | 0.55386 | 0.58997 | 0.00000 | 0.02435 |
| pixelseal / adapted_fixed_equal_ensemble | +0.01185 | [+0.00495, +0.01898] | 0.55151 | 0.59017 | 0.00000 | 0.03886 |
| trustmark / synthetic_only | +0.01183 | [-0.01028, +0.03429] | 0.55257 | 0.56600 | 0.00000 | 0.04518 |
| trustmark / synthetic_only_fixed_equal_ensemble | +0.01847 | [+0.00606, +0.03177] | 0.55404 | 0.57780 | 0.00000 | 0.04601 |
| trustmark / adapted | +0.03127 | [+0.01325, +0.05022] | 0.58062 | 0.57682 | 0.00000 | 0.06082 |
| trustmark / adapted_fixed_equal_ensemble | +0.02405 | [+0.01350, +0.03555] | 0.56726 | 0.57575 | 0.00000 | 0.05779 |

Bootstrap5000 po obrazach, z parowaniem tych samych nazw; warunkowy względem wytrenowanych modeli. Nie obejmuje ponownego treningu ani korekty za wielokrotne używanie danych rozwojowych. Zależność zdjęć źródłowych nie jest w pełni znana.

## Diagnostyka bez treningu

170 realnych obrazów/metodę, właściwe i błędne wiadomości. Macro AUC/AP liczone osobno dla każdego obrazu. Sam ranking nie daje gwarancji mIoU.

| Metoda | Mapa | AUC correct | AUC wrong |
|---|---|---:|---:|
| pixelseal | dense_256_cos | 0.63352 | 0.47581 |
| pixelseal | dense_512_cos | 0.63223 | 0.52695 |
| pixelseal | window_128 | 0.52226 | 0.49224 |
| pixelseal | window_256 | 0.60217 | 0.52829 |
| pixelseal | gradient | 0.49985 | 0.49971 |
| pixelseal | reembed | 0.49812 | 0.50320 |
| trustmark | dense_245_cos | 0.60127 | 0.49571 |
| trustmark | dense_512_cos | 0.51325 | 0.50540 |
| trustmark | window_128 | 0.52951 | 0.50281 |
| trustmark | window_256 | 0.51659 | 0.51492 |
| trustmark | gradient | 0.49953 | 0.50088 |
| trustmark | reembed | 0.51902 | 0.50512 |

Bezpośrednie mapy nie osiągnęły AUC0,80+. Gradient×HPF i re-embedding były bliskie losowym; nie włączono ich do nowego modelu. Kontrola zgodności GAP przestrzennych logitów z oryginalnym wrapperem: maksymalny błąd PS4,77e-6, TM5,72e-6.

## Wykonany model i trening

- Oryginalny pretrained extractor PixelSeal / TrustMark; nie wagi ImageNet. PixelSeal używa klasy ConvNeXtV2 z oficjalnego źródła checkpointu. TrustMark: oryginalny ResNet, wszystkie running stats BN zamrożone.
- Wspólne wagi ekstraktora dla wejścia512 i skali natywnej: PS256/TM245. Na każdym poziomie po projekcji128ch sumujemy cechy dwóch skal. Decoder progresywny do128.
- Mapy dot/cosine/soft-accuracy względem podanych bitów, na obu skalach. Dot ograniczony tanh(dot/4).
- Lekka gałąź RGB + Y + high-passRGB + JND. JND z oryginalnej funkcji na obserwowanym obrazie; to nie znana historyczna mapa na oryginalnym cropie sprzed edycji.
- Original embedders bez zmian.2500 zapisanych realizacji na rodzinę; źródła i wiadomości rozłączne z dev128. Dev ma64 stałe przykłady.
- Od początku nieregularne maski proceduralne, paski i dziury. Bez pełnej mieszanki COCO obiektów, transplantacji lub inpaintingu w tej iteracji.
-400aktualizacji×batch32=12800widoków.50kroków zamrożonego ekstraktora,350joint. LR ekstraktora2e-5, nowych modułów3e-4,20warmup,cosine do20%wartości początkowej; AdamW wd0,01; clipping1.
- Całość FP32. Per-image0,5BCE+0,5Dice obu klas, ranking par rampowany do0,1. Bez pomocniczego wymuszania wszystkich bitów w małym fragmencie.
- Synthetic-only: końcowe surowe wagi po400krokach, próg0,5, bez realnych etykiet w treningu.
- Adaptacja: te same3odtworzone oryginalne foldy,150kroków/fold,24synthetic+8real outer-train/batch32; LR2e-6/2e-5. Stały harmonogram i próg0,5, bez doboru na outer-validation.
- Ensemble: wcześniej zadeklarowana średnia prawdopodobieństw nowego modelu i starego ensemble trzech seedów; próg to średnia ich progów. Brak wyszukiwania wag na OOF.

## Koszt i weryfikacja

JURECA, jeden węzeł na zadanie, przydział 4×A100 40GB. Diagnostyka 15903994: 2m28s (używane 2GPU); pretraining 15904000: 9m11s; adaptacja 15904020: 10m30s. Wszystkie COMPLETED. Łącznie 22m09s czasu przydziałów, około 1,48 przydzielonej GPU-godziny; bez czasu kolejki i lokalnej analizy. Batch32, peak allocated memory około24,1GB PS i14,1GB TM.

Sprawdzono rozłączność train/validation każdego foldu oraz pokrycie dokładnie170 unikalnych obrazów na rodzinę. Połączenie z baseline wymaga identycznych nazw i masek GT. Pełne mapy umożliwiają ponowne przeliczenie metryk.

## Interpretacja

To test całej receptury: inicjalizacja, dwie skale, message maps i generator zmieniły się łącznie. Nie identyfikuje przyczynowego wkładu pojedynczego komponentu. Synthetic-only, OOF po adaptacji i publiczny leaderboard to odrębne protokoły. Zbiór170/metodę był wielokrotnie używany rozwojowo, więc wynik nie jest bezstronną prognozą ukrytego testu.

Kontrole AUC nie potwierdzają obietnic wzrostu SNR ani mIoU0,70–0,80. Nie jest to dowód niemożliwości takich wyników. Nie zmieniono działającego StegaStamp; dotychczasowe submission zachowano.

## Artefakty i odtworzenie

`results/comparison.json`, per-image JSON, mapyFP32/GT i konfiguracje w `results/runs/`; źródła, pełne checkpointy i optimizer state na JURECA: `/p/project1/training2666/pszenny1/native-localizer/message-runs/`.

Ze środowiska task-1 na klastrze (w przydziale GPU):
```bash
python -m experiments.message_localizer.signal_job
python -m experiments.message_localizer.train_job
python -m experiments.message_localizer.adapt_job
```

Analiza lokalna po pobraniu artefaktów:
```bash
PYTHONPATH=task-1:task-1/src task-1/.venv/bin/python -m experiments.message_localizer.compare --runs task-1/experiments/message_localizer/results/runs
task-1/.venv/bin/python task-1/experiments/message_localizer/report.py
```

## Zgłoszenie testowe

Na wyraźne polecenie użytkownika wykonano refit obu nowych modeli na wszystkich170 obrazach/family:150kroków, stałe LR2e-6/2e-5, batch32=24synthetic+8real, próg0,5. Bez zmiany StegaStamp i bez strojenia na testowych etykietach.

Serwer: submission #2163, status success, public mIoU 0.6145; poprzedni najlepszy0,5993. Jest to publiczny podzbiór, nie ukryty wynik finałowy.

Plik `results/submission/submission-message-localizer.pt`:1020 unikalnych nazw, maski uint8[1020,128,128], binarne, poniżej20MB. Hash i potwierdzenie bezpiecznego odczytu w `assembly_validation.json`; źródła checkpointów w per-family inference JSON. Poprzednie zgłoszenie#2008 zachowano.

Refit zakończony w zadaniu15904072. Pierwsza inferencja przerwana przez brak modułu submission.py w kopii na klastrze; po skopiowaniu modułu uruchomiono samą inferencję w zadaniu15904087, bez ponownego treningu.

Refit i inferencja: `python -m experiments.message_localizer.refit_job` w przydziale4GPU. Łączenie: `PYTHONPATH=task-1:task-1/src task-1/.venv/bin/python -m experiments.message_localizer.assemble_submission`. Upload wykonany pojedynczym żądaniem, z kluczem w stdin; klucza nie zapisano w artefaktach.
