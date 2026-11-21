# Symetryczny Dice i dekoder 128×128

Dwa niezależne eksperymenty: BCE + Dice obu klas przy dotychczasowej głowicy 32×32 oraz oryginalny BCE + Dice watermarku z fuzją i konwolucjami w 128×128.
Wariant 128 korzysta z ponownie wyciągniętych cech: zachowuje natywne rozmiary do 128×128. Nie powiększa starego cache 32×32. PixelSeal/TrustMark nie mają natywnych cech 128×128 w wybranych warstwach; są interpolowane po projekcji.
Seed 42, AdamW 0.001, batch 8, maksymalnie 50 epok, patience 8. Te same podziały i dobór epoki/progu wewnątrz treningu co poprzednie CV.
Screening: pierwszy fold, 57 obrazów walidacyjnych na metodę. Próg wejścia do CV ustalony przed wynikami: średnia poprawa >= 0.005 oraz poprawa w co najmniej 2 metodach.

| Wariant | Metoda | Bazowy fold 1 | Nowy fold 1 | Zmiana | Epoka | Próg |
|---|---|---:|---:|---:|---:|---:|
| symmetric_dice | pixelseal | 0.519131 | 0.514614 | -0.004517 | 15 | 0.5 |
| symmetric_dice | trustmark | 0.512931 | 0.484934 | -0.027997 | 4 | 0.4 |
| symmetric_dice | stegastamp | 0.465554 | 0.470989 | +0.005435 | 17 | 0.5 |

symmetric_dice: zmiana łącznego wyniku pierwszego foldu -0.009026; próg CV niespełniony.

| Wariant | Metoda | Bazowy fold 1 | Nowy fold 1 | Zmiana | Epoka | Próg |
|---|---|---:|---:|---:|---:|---:|
| decoder128 | pixelseal | 0.519131 | 0.504900 | -0.014231 | 14 | 0.7 |
| decoder128 | trustmark | 0.512931 | 0.486883 | -0.026048 | 4 | 0.7 |
| decoder128 | stegastamp | 0.465554 | 0.381171 | -0.084383 | 18 | 0.6 |

decoder128: zmiana łącznego wyniku pierwszego foldu -0.041554; próg CV niespełniony.

Screening służy wyborowi wariantu, nie stanowi niezależnego testu. Ewentualne CV obejmuje również fold użyty do screeningu; wynik ma obciążenie selekcją. Nie zmieniano podziałów w celu poprawy wyniku. Nie używano zbioru testowego ani leaderboardu.
Zapisane checkpointy odtworzyły predykcje z dokładnością 1e-7; metryki przeliczono z zapisanych prawdopodobieństw.
Odtworzenie: `task-1/.venv/Scripts/python.exe task-1/experiments/head_ablation/run.py`, następnie `report.py`. Zakończone dopasowania są wznawiane z plików wynikowych.
