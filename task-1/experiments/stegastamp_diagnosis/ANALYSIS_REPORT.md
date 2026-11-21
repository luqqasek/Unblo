# Wnioski z diagnozy StegaStamp

Najsilniej potwierdzoną przyczyną załamania drugiego foldu jest niestabilność
treningu i zbyt wczesne zakończenie oryginalnego refitu. Skład danych ma znaczenie,
ale nie wyjaśnia sam spadku do 0,3418. Nie stwierdzono błędu przyporządkowania masek,
checkpointów ani obliczania oficjalnej metryki.

## Kontrolowana zmiana długości treningu

Zachowano drugi fold, seed 42, inicjalizację, kolejność batchy, optimizer i próg 0,5.
Zmieniła się wyłącznie liczba epok. Predykcje po 10 epokach odtworzyły oryginał
z tolerancją 1e-7. Po 18 epokach:

| Epoki | mIoU | Średni udział przewidzianego watermarku |
|---|---:|---:|
| 10 | 0,3418 | 80,1% |
| 18 | 0,4552 | 52,6% |

Prawdziwy udział watermarku wynosi 46,7%. Poprawa o **0,1134 mIoU** na dokładnie
tych samych obrazach pokazuje, że duża część problemu jest zależna od treningu.
18 epok pochodzi z wewnętrznej walidacji seedu 7; kontrola jest jednak analizą
post-hoc, a nie nowym niezależnym benchmarkiem. Nie dowodzi optymalności 18 epok
ani tego, że dłuższy trening zawsze pomaga. Dobór epoki na małym wewnętrznym
zbiorze (23 obrazy) nie przeniósł się dobrze na refit oryginalnego seedu.

## Losowość i dobór parametrów

Na foldzie 2 przy stałych 10 epokach i progu 0,5: seed 42 = 0,3418,
seed 7 = 0,4123, seed 123 = 0,4022. Zatem zmienność treningu istnieje nawet
bez zmiany reguły selekcji. Przy ponownym doborze parametrów wewnątrz treningu
nowe seedy osiągnęły 0,4604 i 0,4485 (18 i 19 epok).

Średnie trzech foldów dla całej procedury nested:

| Seed | Średnia mIoU | SD między foldami |
|---|---:|---:|
| 42 | 0,4106 | 0,0630 |
| 7 | 0,4615 | 0,0055 |
| 123 | 0,4644 | 0,0181 |

Nie zastępujemy benchmarku najlepszym seedem. Trzy seedy to ograniczona diagnoza,
nie precyzyjne oszacowanie rozkładu wyników. Inicjalizację i kolejność batchy
zmieniano razem, więc ich osobne udziały pozostają nieustalone.

## Czy drugi fold jest trudniejszy?

- Ma 30/57 regionów zajmujących mniej niż 25% obrazu, wobec 22/57 w pierwszym.
  Trzeci fold ma jednak podobnie: 29/56.
- Zmiana proporcji `full_watermark` nie tłumaczy spadku: po standaryzacji do
  wspólnych proporcji kategorii fold 2 nadal ma tylko 0,3478.
- Średni udział otworów to 1,09% (fold 1: 1,47%; fold 3: 0,99%). Nie jest to
  wyjątkowo duży udział. Krawędzie są nieco liczniejsze, co samo nie dowodzi
  większej trudności.
- Globalna trafność bitów w drugim foldzie to 0,7351, wyższa niż 0,7180 w trzecim.
  Słabe globalne dekodowanie nie wyjaśnia zatem samo minimum lokalizacji.
- Maska przestrzenna daje w drugim foldzie 0,2905, więcej niż w pierwszym i trzecim.
  To kolejny argument przeciw ogólnej tezie, że jest on po prostu najtrudniejszy.

Brak metadanych czterech konkretnych typów edycji. Dostępna jest tylko flaga
`full_watermark`; nie utożsamiamy jej z outpaintingiem czy transplantacją.
Pozostaje możliwy wpływ treści obrazów, geometrii i składu zbioru treningowego.

## Obejrzane błędy

W drugim foldzie dominują false positives: średnio 34,3% powierzchni obrazu,
przy 0,8% false negatives. Obraz `53934df0429a` z pluszakami ma mały dodatni
prostokąt, a predykcja wypełnia cały obraz (mIoU 0,003). `5b7127ca0a3a`
z samolotem ma niewielki prostokąt GT, ale niemal pełną predykcję (0,030).
`b00270778e48` ma watermark na sylwetce samolotu, podczas gdy model zaznacza
większość tła (0,080).

Trzeci fold pokazuje przeciwną skłonność: przewidywane 30,1% wobec 45,3% GT,
średnio 18,4% powierzchni to false negatives. Na `a7ee54c2d1f7` oraz
`62042cbd008e` model wycina duże plamy mimo niemal pełnego watermarku.
Nie jest to wyłącznie niedokładność na granicy ani odwrócenie etykiet.

## Dlaczego nadal słabiej niż PixelSeal i TrustMark?

Nowe przebiegi zmniejszają różnicę, ale nie wyjaśniają całej: około 0,46 pozostaje
poniżej wcześniejszych 0,508 i 0,496. To porównanie różnych obrazów i ekstraktorów,
przy czym PixelSeal i TrustMark nie powtarzano tu dla dodatkowych seedów.
Nie można z niego wywnioskować, że same cechy StegaStamp są gorsze.

Możliwe dalsze ograniczenia to redukcja wczesnych cech StegaStamp z 200×200
i 100×100 do maksymalnie 32×32, mały zbiór treningowy i strata BCE + Dice
różna od ocenianego dwuklasowego mIoU. To hipotezy: nie wykonano ablacjii
rozdzielczości, lossu ani porównania z RGB. Utrata rozdzielczości nie tłumaczy
sama poprawy po dodatkowych epokach przy niezmienionych cechach.

## Zalecany następny eksperyment

Najpierw ustabilizować dobór długości treningu: wewnętrzne podziały
stratyfikowane według `full_watermark` i przedziałów powierzchni maski, wybór
epoki na kilku wewnętrznych foldach, a następnie refit i ocena na niezmienionym
zewnętrznym foldzie. Zachować z góry ustalone seedy 42, 7 i 123 i raportować
wszystkie wyniki, zamiast wybierać najlepszy po zewnętrznym mIoU.

Dopiero potem porównać większą rozdzielczość cech oraz RGB-only na tych samych
podziałach. Wyniki nie uzasadniają jeszcze dodawania syntetycznych danych.

## Weryfikacja i odtworzenie

`verify.py` odtworzył predykcje wszystkich sześciu nowych checkpointów nested
i potwierdził ich oficjalne mIoU. `epoch_probe.py` odtworzył oryginalne predykcje
seedu 42 po 10 epokach, a następnie wykonał kontrolę 18 epok.
Surowe liczby znajdują się w `results/metrics.json`, `results/epoch_probe.json`
i `results/per_image.csv`. Polecenia podstawowej diagnozy są w `REPORT.md`;
kontrola dodatkowa:

```powershell
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_diagnosis/epoch_probe.py
task-1/.venv/Scripts/python.exe task-1/experiments/stegastamp_diagnosis/verify.py
```
