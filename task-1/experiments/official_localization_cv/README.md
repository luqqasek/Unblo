# Pięć foldów CV — wpływ pretrained lokalizacji

## Wyniki — 2026-11-21

| Wariant | Baseline OOF | Detektor | Zmiana | SD zmiany między foldami | Dodatnie foldy |
| --- | ---: | ---: | ---: | ---: | ---: |
| TrustMark top1 | 0.284897 | 0.355219 | **+0.070322** | 0.017479 | 5/5 |
| TrustMark suma bbox | 0.284897 | 0.346260 | **+0.061363** | 0.017735 | 5/5 |
| StegaStamp segmentacja z mirrora | 0.275392 | 0.278603 | +0.003211 | 0.023196 | 2/5 |

Zmiany mIoU w kolejnych foldach:

| Fold | TrustMark top1 | TrustMark suma bbox | StegaStamp |
| --- | ---: | ---: | ---: |
| 1 | +0.051371 | +0.037411 | -0.016900 |
| 2 | +0.061645 | +0.058770 | -0.015108 |
| 3 | +0.081852 | +0.068454 | +0.012975 |
| 4 | +0.062167 | +0.056286 | +0.038829 |
| 5 | +0.094573 | +0.085893 | -0.003739 |

Orientacyjne, warunkowe 95% przedziały sparowanego bootstrapu dla zmiany:
TrustMark top1 [0.038152, 0.102412], suma bbox [0.029534, 0.092466],
StegaStamp [-0.024993, 0.030949]. Ograniczenia przedziałów opisano niżej.
Top1 poprawia wynik na 112/170 obrazach, suma bbox na 110/170, StegaStamp na 83/170.

Całe 510 obrazów OOF (PixelSeal z priory we wszystkich wariantach):

| Zastąpienie baseline'u detektorem | mIoU | Zmiana |
| --- | ---: | ---: |
| Żadna metoda | 0.270626 | — |
| Tylko TrustMark top1 | **0.294066** | +0.023441 |
| TrustMark top1 oraz StegaStamp | 0.295137 | +0.024511 |

Wniosek: TrustMark top1 daje powtarzalną poprawę względem priory. Dodatkowy wpływ
StegaStamp na łączny wynik to tylko +0.001070, a jego efekt per metoda zmienia
znak między foldami. Rozsądny wariant do następnego eksperymentu to TrustMark
top1 i dotychczasowy baseline dla pozostałych metod. Te wyniki są lokalne;
nie porównujemy ich bezpośrednio z wynikiem leaderboardu 0.2934.

Sprawdzono 340 predykcji detektorów, 510 predykcji OOF baseline'u i zapisano
pełne artefakty. Jedenaście testów repozytorium i adapterów/CV przeszło.

## Protokół

Porównanie z `../official_localization/`, rozszerzone na wszystkie 170 obrazów
TrustMark i 170 StegaStamp. Pięć rozłącznych foldów, seed 42, po 34 obrazy
każdej metody do oceny i 136 do wyznaczenia priory. Pierwszy fold odpowiada
wcześniejszej walidacji. Każdy obraz jest oceniany dokładnie raz poza swoim
foldem uczącym. To 5-fold CV, nie pięć powtórzeń losowego holdoutu.

Detektory są zamrożone, bez treningu i strojenia progów: ich predykcje wystarczy
policzyć raz dla każdego obrazu. Baseline częstości jest dopasowywany od nowa
w każdym foldzie, wyłącznie do masek uczących. Porównujemy predykcje na tych
samych obrazach; zmiana mIoU to `detektor - baseline`.

Sprawdzamy TrustMark top1, sumę bbox oraz segmentację StegaStamp. StegaStamp
pochodzi z mirrora społecznościowego: sumy plików są sprawdzone, ale identyczności
z niedostępnymi oryginalnymi wagami autora nie potwierdzono. Szczegóły i źródła:
`../official_localization/RESEARCH.md`.

Raport zawiera średnie, wyniki i różnice w każdym foldzie, odchylenie standardowe
pięciu różnic i liczbę dodatnich foldów. Dodatkowo zapisujemy 95% przedział
percentylowy z 10 000 sparowanych bootstrapów obrazów (seed 42). Ten przedział
jest warunkowy względem gotowych predykcji OOF: nie uwzględnia ponownego uczenia,
zależności nakładających się zbiorów uczących ani ewentualnych wspólnych obrazów
źródłowych. Nie jest formalną gwarancją istotności ani wyniku leaderboardu.

Całościowy wynik dla 510 obrazów porównuje priory we wszystkich trzech metodach
z zastąpieniem TrustMark przez top1 oraz dodatkowo StegaStamp przez segmentację.
PixelSeal zawsze zachowuje priory wyznaczoną w danym foldzie. Wynik nie jest
estymacją wyuczonego wyboru wariantów: raportujemy wcześniej określone warianty.

## Uruchomienie

Z katalogu repozytorium, po przygotowaniu modeli z poprzedniego eksperymentu:

```powershell
.\task-1\.venv\Scripts\python.exe task-1/experiments/official_localization/evaluate.py --method trustmark --all-images --out task-1/experiments/official_localization_cv/results/cache
.\task-1\.cache\stegastamp-env\Scripts\python.exe task-1/experiments/official_localization/evaluate.py --method stegastamp --all-images --out task-1/experiments/official_localization_cv/results/cache
.\task-1\.venv\Scripts\python.exe task-1/experiments/official_localization_cv/cross_validate.py
.\task-1\.venv\Scripts\python.exe task-1/experiments/official_localization_cv/test_cv.py
```

Artefakty w ignorowanym `results/`: `cache/` z predykcjami detektorów,
`folds.json` z dokładnym podziałem, `cv_metrics.json` z porównaniem i `oof.npz`
z maskami OOF baseline'u oraz obu wariantów łączonych. Żaden submission nie jest
wysyłany. Testy sprawdzają pokrycie, rozłączność, zgodność pierwszego folda,
brak wykorzystania masek walidacyjnych w priorze i kierunek sparowanej różnicy.
