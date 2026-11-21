# Porównanie BCE, równoważenia klas i focal loss

Głowica 32×32 i zamrożone ekstraktory bez zmian. Seed 42, batch 8, AdamW 0.001, maksymalnie 50 epok i patience 8. Epoka i próg dobierane na zapisanym wewnętrznym podziale; potem trening od nowa na całej części uczącej foldu.
Screening: ten sam pierwszy fold co poprzednio, po 57 obrazów walidacyjnych na metodę. CV tylko przy poprawie średniego mIoU >= 0.005 i poprawie w co najmniej dwóch metodach. CV obejmuje fold screeningowy, więc nie jest niezależnym testem wyboru wariantu.
Focal nie używa wag alpha; gamma wynosi 1 lub 2. Zmniejsza też skalę składnika pikselowego względem Dice: poprawy nie można przypisać wyłącznie wyborowi trudnych pikseli.
Równoważenie per obraz: osobne średnie BCE dla każdej obecnej klasy, potem średnia klas; wynik mieszany 50/50 ze zwykłym BCE. Małe obszary nadal mogą dostać dużą wagę.

Stan: complete

| Wariant | PixelSeal | TrustMark | StegaStamp | Łącznie | Zmiana | CV |
|---|---:|---:|---:|---:|---:|---|
| Bazowy BCE + Dice | 0.519131 | 0.512931 | 0.465554 | 0.499205 | — | już wykonane |
| bce_only | 0.518901 | 0.499171 | 0.426996 | 0.481689 | -0.017516 | pominięte |
| bce75_dice25 | 0.524414 | 0.461119 | 0.435328 | 0.473620 | -0.025585 | pominięte |
| balanced_bce_mix | 0.519798 | 0.494723 | 0.458781 | 0.491101 | -0.008105 | pominięte |
| focal1_dice | 0.513546 | 0.478842 | 0.385267 | 0.459218 | -0.039987 | pominięte |
| focal2_dice | 0.520707 | 0.518328 | 0.448766 | 0.495933 | -0.003272 | pominięte |

## bce_only

BCE

| Metoda | IoU czyste | IoU watermark | FPR | FNR | Udział predykcji | Udział GT | Epoka | Próg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pixelseal | 0.5671 | 0.4707 | 0.0613 | 0.1581 | 0.4293 | 0.4714 | 16 | 0.6 |
| trustmark | 0.5666 | 0.4317 | 0.0600 | 0.1052 | 0.4400 | 0.4552 | 7 | 0.6 |
| stegastamp | 0.4509 | 0.4031 | 0.0537 | 0.2943 | 0.4161 | 0.5559 | 19 | 0.5 |

## bce75_dice25

0.75 BCE + 0.25 foreground Dice

| Metoda | IoU czyste | IoU watermark | FPR | FNR | Udział predykcji | Udział GT | Epoka | Próg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pixelseal | 0.5597 | 0.4891 | 0.0709 | 0.1318 | 0.4468 | 0.4714 | 11 | 0.5 |
| trustmark | 0.4558 | 0.4664 | 0.2037 | 0.0510 | 0.5430 | 0.4552 | 11 | 0.6 |
| stegastamp | 0.4561 | 0.4145 | 0.0551 | 0.2759 | 0.4270 | 0.5559 | 21 | 0.6 |

## balanced_bce_mix

0.5 pixel-mean BCE + 0.5 per-image class-mean BCE

| Metoda | IoU czyste | IoU watermark | FPR | FNR | Udział predykcji | Udział GT | Epoka | Próg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pixelseal | 0.5585 | 0.4811 | 0.0869 | 0.1539 | 0.4448 | 0.4714 | 19 | 0.4 |
| trustmark | 0.5710 | 0.4185 | 0.0942 | 0.1968 | 0.4169 | 0.4552 | 10 | 0.6 |
| stegastamp | 0.4364 | 0.4812 | 0.1509 | 0.2174 | 0.5021 | 0.5559 | 26 | 0.6 |

## focal1_dice

0.5 focal(gamma=1, no alpha weighting) + 0.5 foreground Dice

| Metoda | IoU czyste | IoU watermark | FPR | FNR | Udział predykcji | Udział GT | Epoka | Próg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pixelseal | 0.5369 | 0.4901 | 0.1050 | 0.1143 | 0.4730 | 0.4714 | 15 | 0.6 |
| trustmark | 0.4910 | 0.4667 | 0.1747 | 0.0646 | 0.5210 | 0.4552 | 8 | 0.5 |
| stegastamp | 0.4567 | 0.3138 | 0.0331 | 0.4596 | 0.3152 | 0.5559 | 18 | 0.6 |

## focal2_dice

0.5 focal(gamma=2, no alpha weighting) + 0.5 foreground Dice

| Metoda | IoU czyste | IoU watermark | FPR | FNR | Udział predykcji | Udział GT | Epoka | Próg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pixelseal | 0.5626 | 0.4788 | 0.0693 | 0.1542 | 0.4353 | 0.4714 | 15 | 0.6 |
| trustmark | 0.5801 | 0.4566 | 0.0745 | 0.0957 | 0.4522 | 0.4552 | 12 | 0.5 |
| stegastamp | 0.3200 | 0.5775 | 0.3513 | 0.0434 | 0.6878 | 0.5559 | 24 | 0.5 |

IoU klas: średnia po obrazach. FPR=FP/liczba czystych pikseli; FNR=FN/liczba oznaczonych pikseli (agregacja pikselowa). Pełne liczniki FP/FN i wyniki według powierzchni masek w results/summary.json.
Checkpointy odtwarzają predykcje z tolerancją 1e-7; metryki sprawdzono z zapisanych prawdopodobieństw. Nie użyto danych testowych ani leaderboardu.
Odtworzenie: `task-1/.venv/Scripts/python.exe task-1/experiments/loss_screen/run.py`. Po każdej ukończonej próbie skrypt aktualizuje raport i dashboard PL/EN.
