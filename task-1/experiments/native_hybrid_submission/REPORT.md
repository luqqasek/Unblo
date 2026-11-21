# Najlepsze modele osobno dla metod — zgłoszenie #2008

**Zaakceptowano zgłoszenie #2008: publiczny mIoU 0,5993; poprzedni najlepszy 0,5235; poprawa +0,0758.** Wysłano jeden plik, bez strojenia po wyniku leaderboardu.

## Walidacja i wybór

| Metoda | Nowy pilot RGB, 170 realnych obrazów | Wybrany model / walidacyjny mIoU |
|---|---:|---|
| PixelSeal | 0,25303 | Stary n4k_s2500, ensemble seedów 42/7/123: **0,55899** |
| TrustMark | 0,50601 | Stary n4k_s5000, ensemble seedów 42/7/123: **0,54745** |
| StegaStamp | **0,69496** | Nowy ConvNeXt-v1 Base + lokalny stem, seed42 |

Nowe piloty: identyczna architektura i zamrożone ustawienia jak StegaStamp: 2500 dokładnie oznakowanych źródeł treningowych, 128 źródeł dev, 200 aktualizacji po 64 widoki, FP32. PixelSeal nie nauczył się lokalizacji: FG IoU=0, przewidywanie clean. TrustMark był słabszy od wcześniejszego pipeline'u. Zachowano checkpointy i pełne wyniki; do zgłoszenia nie użyto tych dwóch nowych pilotów.

Stare warianty wybrano według raportu synthetic_scaling. Średnie pojedynczych seedów wynosiły 0,55360 (PS) i 0,54407 (TM). Średnia prawdopodobieństw trzech seedów, progowana średnią ich wewnętrznie wybranych progów, dała zapisane wyżej wyniki OOF. Wyboru ensemble dokonano przed predykcjami testowymi.

## Końcowy refit

Każdy seed PixelSeal i TrustMark rozpoczął od właściwego syntetycznego checkpointu. Świeży optimizer AdamW lr=0,001, wd=0,0001, batch8 i oryginalny loss/head. Refit na wszystkich 170 obrazach; liczba epok i próg to mediany zapisanych wyborów inner CV z trzech foldów:

| Metoda | Seed42 epoki / próg | Seed7 | Seed123 | Próg średniej prawdopodobieństw |
|---|---|---|---|---|
| PixelSeal | 4 / 0,5 | 11 / 0,6 | 4 / 0,5 | 0,53333 |
| TrustMark | 3 / 0,4 | 6 / 0,5 | 9 / 0,5 | 0,46667 |

Refit sześciu głowic i predykcje z istniejących cech testowych: **50,70 s** na Windows RTX3070. Bez nowego pretrainingu starych głowic. StegaStamp: końcowe surowe wagi pilota 200 kroków, stały próg0,5, natywne RGB512, bez STN/TTA. Jego inferencja wykonała się na A100.

## Artefakt i weryfikacja

`results/submission-best-per-family.pt`: dokładnie 1020 unikalnych nazw, po340 na metodę, uint8 [1020,128,128], wyłącznie0/1, **16 734 779 bajtów**. Sprawdzono mapowanie nazw, pokrycie, zgodność części rodzinnych i bezpieczne ponowne wczytanie.

SHA256: `5483bf5479522469936d7d82a33e8409e1cb472876141fab31c60542c5859fde`.

[Potwierdzenie serwera](results/server_receipt.json) · [Kontrola pliku](results/assembly_validation.json) · [Zamrożony refit](results/frozen_recipe.json) · [Walidacja](results/validation_comparison.json).

## Ograniczenia

Wyniki walidacyjne są rozwojowe i nie stanowią niezależnego potwierdzenia wyboru receptury. PS/TM: modele po realnym fine-tuningu, trzy seedy i foldy; SG: pojedynczy model synthetic-only, bez realnych etykiet w treningu. Ich średnia około0,60047 jest wyłącznie opisowa; nie jest wynikiem jednego jednolitego protokołu CV. Wynik0,5993 dotyczy publicznego splitu, nie ukrytej końcowej części testu.

## Odtworzenie

Na Windowsie z istniejącymi artefaktami synthetic_scaling: `.venv-rgb-cuda/Scripts/python.exe experiments/native_hybrid_submission/deploy_legacy.py` (chroni zapisane predykcje). StegaStamp: moduł `experiments.native_localizer.infer_family` z checkpointem JURECA `runs/stegastamp/pilot-30min/last.pt`. Po skopiowaniu trzech subsetów: `python experiments/native_hybrid_submission/assemble.py`. Upload pozostaje osobnym, jawnym krokiem; klucz API nie jest zapisany w kodzie ani artefaktach.
