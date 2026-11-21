# Zgłoszenie: syntetyki → fine-tuning → test

Po zakończeniu eksperymentu użytkownik polecił przygotować i wysłać predykcje testowe, a następnie jawnie wybrał wariant po fine-tuningu. Wcześniejszy audyt nie korzystał z testu; poniższa operacja jest oddzielnym, późniejszym etapem.

## Model i wybór parametrów

- StegaStamp: zapisany pretrain na 2000 syntetyków, seed 42, 10 epok; świeży fine-tuning na wszystkich 170 rzeczywistych obrazach.
- Fine-tuning: **7 epok**, próg **0.6**; mediany epok 7/5/10 i progów 0.5/0.6/0.6 z trzech foldów seedu 42.
- Niezmieniona głowica, zamrożony ekstraktor, rzeczywista korekta inverse-STN. AdamW LR=0.001, weight decay=0.0001, batch 8, float32 CUDA, TF32 off; optymalizator zresetowany po pretreningu.
- Domyślny seed 42, bez wyboru najlepszego seedu. Test nie wybierał modelu, epoki, progu ani postprocessingu.
- PixelSeal i TrustMark: zachowane 680 masek z najlepszego wcześniejszego zgłoszenia #1202. Nowe są wszystkie 340 predykcji StegaStamp.

Wynik CV wariantu po fine-tuningu: 0.50719 po trzech seedach, 0.51514 dla seedu 42. Synthetic-only osiągał średnio 0.51153, ale nie został wysłany. Obecne zgłoszenie nie jest porównaniem obu wariantów na leaderboardzie.

## Walidacja pliku

**PASS:** 1020 unikalnych wymaganych nazw, tensor uint8 1020×128×128, wyłącznie 0/1; 16,734,635 bajtów. Ponowne wczytanie checkpointu dało identyczne prawdopodobieństwa. Sprawdzono przypisanie masek po nazwie i niezmienność pozostałych 680 masek.

SHA-256 zgłoszenia: `6298caa1853c1431b41fe73c19e7abfdf299ad6d643460a8327badddcf856d84`.

SHA-256 nowego checkpointu: `d69fb0d0520b10d8917e32692dd31ad90523ae22584b7c8766dc28945d261e6c`.

Udział pikseli dodatnich StegaStamp: 0.43853. Zmieniono 473,281 pikseli względem wcześniejszego StegaStamp. To statystyki predykcji, nie miary jakości na nieznanym GT.

## Odpowiedź leaderboardu

**Zaakceptowano #1496: publiczne mIoU 0.5235.** Poprzedni najlepszy wynik 0.5123; zmiana **+0.0112**.

Serwer potwierdził nowy najlepszy wynik.

To wynik publicznego podzbioru. Końcowy ranking zależy od ukrytego podzbioru ujawnianego po zamknięciu zgłoszeń. Wykonano jeden POST; nie dostrajano modelu na podstawie tej odpowiedzi.

[Odpowiedź serwera](results/full_train/upload_receipt.json) · [Walidacja predykcji](results/full_train/prediction_summary.json) · [Konfiguracja pełnego treningu](results/full_train/training_summary.json)

## Odtworzenie predykcji

```powershell
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/deploy.py fit
task-1/.venv/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/deploy.py extract
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/deploy.py predict
```

`fit` i `extract` wznawiają zgodne artefakty; `predict` chroni istniejący plik zgłoszenia przed nadpisaniem. Wysyłka jest osobnym poleceniem i nie stanowi części odtwarzania predykcji. Checkpointy oraz plik submission znajdują się lokalnie w `results/full_train/`.
