# StegaStamp: stabilniejszy wybór epoki i progu

Zmiana: pojedynczy wewnętrzny holdout 80/20 zastąpiono trzema wewnętrznymi foldami, stratyfikowanymi według full_watermark i udziału maski (granice 25%, 50%, 75%). Epokę wybiera średnie per-image mIoU z połączonych predykcji wewnętrznych przy progu 0,5. Próg z [0,3; 0,4; 0,5; 0,6; 0,7] dobiera się z tych samych predykcji dla wybranej epoki. Następnie głowica jest trenowana od nowa na całym zewnętrznym train.

Bez zmian: zamrożone cechy, architektura głowicy, AdamW, loss 0,5 BCE + 0,5 Dice, batch 8, maks. 50 epok, patience 8. Zewnętrzne foldy i seedy 42/7/123 identyczne jak w poprzedniej diagnozie. Nie stosowano minimalnej liczby epok ani wymuszenia 18 epok.

## Wyniki porównania parami

| Fold | Seed | Poprzednio | Nowa selekcja | Zmiana mIoU | Epoki | Próg |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.4656 | 0.4687 | +0.0031 | 32 | 0.5 |
| 1 | 7 | 0.4674 | 0.4697 | +0.0023 | 32 | 0.6 |
| 1 | 123 | 0.4605 | 0.4738 | +0.0133 | 32 | 0.5 |
| 2 | 42 | 0.3418 | 0.4450 | +0.1031 | 23 | 0.6 |
| 2 | 7 | 0.4604 | 0.4496 | -0.0108 | 13 | 0.6 |
| 2 | 123 | 0.4485 | 0.4429 | -0.0056 | 15 | 0.4 |
| 3 | 42 | 0.4244 | 0.4550 | +0.0306 | 29 | 0.5 |
| 3 | 7 | 0.4566 | 0.4738 | +0.0172 | 30 | 0.5 |
| 3 | 123 | 0.4842 | 0.4703 | -0.0138 | 26 | 0.5 |

| Seed | Poprzednia średnia foldów | Nowa średnia | Zmiana | Poprzednie SD foldów | Nowe SD foldów |
|---|---:|---:|---:|---:|---:|
| 42 | 0.4106 | 0.4562 | +0.0456 | 0.0630 | 0.0119 |
| 7 | 0.4615 | 0.4644 | +0.0029 | 0.0055 | 0.0130 |
| 123 | 0.4644 | 0.4623 | -0.0021 | 0.0181 | 0.0169 |

| Fold | Poprzednia średnia seedów | Nowa średnia | Poprzednie SD seedów | Nowe SD seedów |
|---|---:|---:|---:|---:|
| 1 | 0.4645 | 0.4707 | 0.0036 | 0.0027 |
| 2 | 0.4169 | 0.4458 | 0.0653 | 0.0034 |
| 3 | 0.4551 | 0.4664 | 0.0299 | 0.0100 |

Średnia po 3 foldach × 3 seedach: **0.4455 → 0.4610**, zmiana **+0.0155 mIoU** (+1.55 punktu procentowego). Poprawa w 6/9 porównań fold–seed.

Średnia foldów waży foldy równo; wynik OOF zapisany w summary.json waży obrazy równo (57/57/56). Powtórzone seedy używają tych samych obrazów, więc 9 przebiegów nie jest 9 niezależnymi próbami. Nie wyliczamy na tej podstawie pozornego przedziału ufności.

## Wniosek

Nowa procedura ogranicza załamanie oryginalnego seedu 42 i poprawia średnią wszystkich porównań. Nie poprawia każdego przebiegu: seed 123 ma nieznacznie niższą średnią. Największy zysk dotyczy stabilności drugiego foldu, a nie podniesienia najlepszego osiągniętego wyniku. Dla dalszych eksperymentów StegaStamp rekomendujemy tę procedurę jako lepiej uzasadniony sposób selekcji, zachowując raportowanie wszystkich trzech seedów. Kosztem jest trening trzech modeli wewnętrznych zamiast jednego. Poprzedni benchmark pozostaje zachowany do porównań; nie zmieniono go wstecznie.

Kontrola wizualna czterech najgorszych oryginalnych przykładów foldu 2 pokazuje, że nie wszystkie błędy lokalizacji zniknęły. Na 53934df0429a i 5b7127ca0a3a nadal przewidywana jest niemal pełna maska; b00270778e48 pogorszył się z 0,080 do 0,032. Poprawa średniej nie oznacza rozwiązania problemu małych obszarów.

## Ograniczenia i odtworzenie

Poprawka powstała po obejrzeniu błędów tych zewnętrznych foldów: jest to porównanie rozwojowe, nie nowy nietknięty test. Żadna zewnętrzna etykieta nie wybiera epoki ani progu w nowym kodzie. Stratyfikacja korzysta wyłącznie z masek zewnętrznego train. Zmiana obejmuje naraz rozmiar wewnętrznego train, stratyfikację i agregację trzech foldów; ich oddzielnych efektów nie zidentyfikowano. Nie użyto testu konkursowego ani leaderboardu.

Pełny przebieg (wznawia ukończone pary fold–seed):
```powershell
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stable_selection/run.py
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stable_selection/verify.py
python task-1/dashboard/build.py
```

Pliki results/: konfiguracja, manifest podziałów, historie walidacji, 9 checkpointów, prawdopodobieństwa, metryki i wykres porównania.

## Trening końcowy i leaderboard

Po zakończeniu CV wytrenowano StegaStamp na wszystkich 170 obrazach, seed 42, 29 epok, próg 0.5. Parametry są medianami nowego CV dla seedu 42. Nie wybierano najlepszego seedu.

Zmieniono 340 masek StegaStamp; 680 masek PixelSeal i TrustMark pochodzi bez zmian z wcześniejszego submission #1153. Zweryfikowano checkpoint, zgodność predykcji, unikalne nazwy, komplet 1020 binarnych masek 128×128 i limit rozmiaru.

Zgłoszenie **#1202**, status **success**, publiczne mIoU **0.5123**, poprzedni najlepszy wynik **0.5066**, poprawa według serwera **+0.0056**. Różnica zaokrąglonych wyświetlanych wyników może odbiegać od delty serwera. Wynik dotyczy publicznej części testu; wynik prywatny pozostaje nieznany.

Submission: `submission.pt`, 16734635 bajtów; SHA256 `bf48d422a10f15e598a2b1f39dfa5dece542f936e6f6766e3aeaeeb11139e01b`.

Odtworzenie pliku (nie wysyła automatycznie):
```powershell
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_stable_selection/train_full_predict.py
```
