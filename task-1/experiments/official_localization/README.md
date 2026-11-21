# Pretrained lokalizacja TrustMark i StegaStamp

Sprawdzone 2026-11-21. Bez treningu i strojenia progów. Ten sam seed 42 i te same
34 obrazy walidacyjne na metodę co w `../baseline/`. PixelSeal poza zakresem tego
eksperymentu. Metryka: mIoU dwóch klas per obraz, potem średnia po obrazach.

| Metoda | Baseline częstości | Oficjalny lokalizator | Różnica |
| --- | ---: | ---: | ---: |
| TrustMark Q: najlepszy bbox | 0.272808 | **0.324179** | +0.051371 |
| TrustMark Q: suma wszystkich bbox | 0.272808 | **0.310219** | +0.037411 |
| StegaStamp: segmentacja z mirrora | **0.283805** | 0.266904 | -0.016900 |

Są to wyniki lokalnej walidacji, nie leaderboardu. Nie przygotowano ani nie
wysłano nowego submission. Przy 34 obrazach na metodę wyniki są orientacyjne.
Top1 i suma bbox to dwa raportowane warianty, nie niezależne potwierdzenie
wyboru najlepszego wariantu na nowej walidacji.

## Jak działa integracja

**TrustMark:** oficjalny kod Adobe z przypiętego commitu
`4c60b9fdb4125fca3001f0803e6890a6b55371ac` (wersja 0.9.2), Q bbox detector,
`loadBBoxDetector=True`, `loadRemover=False`. Używamy `localize()` na RGB PIL.
Wynik to znormalizowany prostokąt xyxy; wypełniamy go na masce 512 × 512,
potem wykonujemy oficjalny downsampling z progiem >= 50%. Nie zmieniamy
wewnętrznych progów detektora. `return_all=True` daje drugi wariant: sumę bbox.
Nowy kod jest importowany z checkoutu w `checkpoints/`, więc główne środowisko
z TrustMark 0.8.0 pozostaje zgodne ze starter kitem.

Detektor używa Faster R-CNN, mimo nazwy pola `yolo` w implementacji.
Oficjalny wrapper przy braku detekcji może zwrócić cały obraz z wynikiem 0,
a `localize()` nie przekazuje confidence. Uzyskaliśmy 13 bbox dokładnie równych
pełnemu obrazowi i 15 pełnych masek po downsamplingu. Bbox nie może odwzorować
dziur i nieregularnych granic watermarku. Czas CPU: około 37.5 s łącznie
z ładowaniem, przy dwóch wywołaniach detektora na obraz i 4 wątkach.

**StegaStamp:** uruchomiony stary TensorFlow SavedModel w `tf.compat.v1`
(TensorFlow CPU 2.15.1), osobne środowisko. Oficjalny preprocessing: RGB,
resize 1024 × 1024, NHWC float32 / 255. Argmax dwóch klas; **klasa 0 to watermark**.
Maskę foreground zmniejszamy biliniowo do 512, binaryzujemy >= 0.5 i stosujemy
oficjalny downsampling do 128. Bierzemy segmentację, bez heurystyk dopasowania
czworokątów, dekodowania wiadomości i ECC z demonstracji wideo.

Oryginalne wagi autora są niedostępne (404; potwierdzenie autora w issue 59).
Użyto kopii z Gu-f/StegaStamp, commit
`d95ea307c73410c96557200656d2e0a59574eedb`, ze sprawdzonymi SHA-256 wszystkich
trzech plików. To potwierdza zgodność z mirrorem, **nie identyczność z utraconym
oryginalnym wydaniem**. Wynik należy traktować jako ocenę tej kopii modelu.

StegaStamp wykrywa średnio 8.56% powierzchni; 3 maski są puste. Segmentacja jest
fragmentaryczna i nie poprawia baseline'u. Możliwą przyczyną jest różnica domen:
oryginalny detektor trenowano na warped StegaStampach w większych kadrach,
a dane konkursowe zawierają również duże obszary i nieregularne ubytki.
To interpretacja wyników, nie potwierdzona przyczyna. Czas CPU: około 27 s
z ładowaniem, 4 wątki.

Źródła, szczegóły wejść i pochodzenia wag: [RESEARCH.md](RESEARCH.md).

## Odtworzenie na Windows

Z katalogu głównego repozytorium, po standardowym setupie task-1:

```powershell
.\task-1\.venv\Scripts\python.exe task-1/experiments/official_localization/prepare.py
.\task-1\.venv\Scripts\python.exe task-1/experiments/official_localization/evaluate.py --method trustmark

.\task-1\.cache\bootstrap\bin\uv.exe venv task-1/.cache/stegastamp-env --python task-1/.venv/Scripts/python.exe --seed
.\task-1\.cache\stegastamp-env\Scripts\python.exe -m pip install -r task-1/experiments/official_localization/requirements-stegastamp.txt
.\task-1\.cache\stegastamp-env\Scripts\python.exe task-1/experiments/official_localization/evaluate.py --method stegastamp

.\task-1\.venv\Scripts\python.exe task-1/experiments/official_localization/test_adapters.py
.\task-1\.venv\Scripts\python.exe task-1/experiments/official_localization/summarize.py
```

Trzy testy adaptera przeszły. `prepare.py` weryfikuje sumy wag przed oceną.
Ewaluacja zapisuje dokładne nazwy walidacji, wyniki per obraz i predykcje NPZ
w ignorowanym `results/`. `summarize.py` dodatkowo sprawdza zgodność nazw,
binarność, rozmiary i ponownie oblicza mIoU z zapisanych predykcji.

Przykłady: `results/trustmark_examples.png`, `results/stegastamp_examples.png`.
Wybrane deterministycznie według wyniku: najgorszy, 1/3, 2/3 i najlepszy obraz.
Białe = watermark, czarne = czysty. Wyniki: `results/summary.json`,
`results/*_metrics.json`, `results/*_masks.npz`, `results/trustmark_boxes.json`.
