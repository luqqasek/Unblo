# TrustMark: więcej bboxów i patche 2×2

## Wyniki — 2026-11-21

170 obrazów TrustMark, pięć foldów po 34. Baseline częstości OOF: 0.284897.

| Wariant | mIoU | Zmiana względem oryginalnego top1 |
| --- | ---: | ---: |
| Cały obraz, top1, oficjalny fallback | 0.355219 | — |
| Cały obraz, suma do 4 bboxów, fallback | 0.346260 | -0.008959 |
| Cały obraz, suma do 20 bboxów, fallback | 0.346260 | -0.008959 |
| Cały obraz, top1, pusto przy braku detekcji | **0.475915** | **+0.120696** |
| Cały obraz, suma do 4 lub 20, pusto przy braku detekcji | 0.466956 | +0.111737 |
| Patche 2×2, top1 lub suma do 4/20, fallback | 0.249826 | -0.105393 |
| Patche 2×2, top1 lub suma do 4/20, pusto przy braku detekcji | 0.257680 | -0.097538 |

Top1 bez fallbacku uzyskuje w kolejnych foldach: 0.472191, 0.465821, 0.487155,
0.501555, 0.452851. Poprawa względem oryginalnego top1 występuje w 5/5 foldach;
średnia zmiana +0.120696, SD między foldami 0.026345. Warunkowy sparowany
bootstrap 95% dla zmiany: [0.090485, 0.151519]. To opisowa ocena wypróbowanych
wariantów na znanym CV, nie niezależny test po wyborze najlepszego.

Zmiana fallbacku top1 poprawia wynik na 46 obrazach i nie zmienia go na
pozostałych 124. Surowy detektor zwraca zero bboxów na dokładnie 46/170 obrazów.
Usunięcie zastępczego pełnego prostokąta pomaga we wszystkich tych przypadkach.
Nie oznacza to, że brak detekcji zawsze dowodzi braku watermarku poza tym zbiorem.

Limit 20 ujawnił piąty bbox tylko na jednym obrazie; największa liczba detekcji
wyniosła 5. Suma masek z limitami 4 i 20 pozostaje identyczna na wszystkich
obrazach. Dla patchy brak detekcji występuje na **634/680 wycinków** (93.2%).
Taki podział bez zmiany skali i ponownego treningu silnie pogarsza detekcję.

Odtworzony pełnoobrazowy top1 i all4 są pikselowo identyczne z poprzednimi
predykcjami oficjalnego API na **170/170 obrazach**, co potwierdza poprawność
przejścia na niższy interfejs. Dwa testy adaptera przeszły; łącznie 13 testów.

Rekomendacja: pełny obraz + top1, z pustą maską przy braku rzeczywistej detekcji.
Zwiększanie limitu i patche 2×2 w tej konfiguracji nie dają poprawy.
## Submission — 2026-11-21

Wysłano pełne 1020 masek: TrustMark full-image top1 bez fallbacku, PixelSeal
i StegaStamp z priory wyznaczoną na całym train. W 65/340 obrazów testowych
TrustMark detektor nie zwrócił bboxów; zapisano dla nich puste maski.
Plik 16 734 635 bajtów przeszedł lokalną walidację oraz walidację serwera.

ID zgłoszenia **1096**, publiczne mIoU **0.3416**, poprzedni wynik **0.2934**,
poprawa **+0.0482**. Serwer potwierdził nowy najlepszy wynik zespołu dla zadania.
Wynik prywatny pozostaje ukryty. Artefakty i odpowiedź serwera:
`results/test_top1_empty/`. Generator: `predict_test.py`.

## Implementacja

Eksperyment zmienia postprocessing zamrożonego detektora Q i rozmiar jego wejścia.
Źródło modelu, pochodzenie wag i przygotowanie środowiska opisano w
`../official_localization/`. Limit 4 w oficjalnym wrapperze to
`model.detector.yolo.yolo_model.roi_heads.detections_per_img`, ustawiony przez
`box_detections_per_img=4`; nie jest ograniczeniem architektury wyuczonej sieci.
Zwiększamy go do 20. Próg score pozostaje 0.05, NMS pozostaje 0.5.
Sam wyższy limit nie wymusza większej liczby detekcji ponad progiem i po NMS.

Patche **2×2 oznaczają cztery obrazy 256×256**, bez overlapu, dodatkowego
skalowania i dopełniania. Nie oznaczają wycinków 2×2 piksele. Detektor przetwarza
cztery wycinki w batchu, lokalne współrzędne bboxów przesuwamy do globalnych
współrzędnych obrazu. Wypełniamy prostokąty na siatce 512×512, a dopiero potem
stosujemy oficjalną regułę downsamplingu >=50% do **128×128**. Granice patchy
256 są podzielne przez 4, więc są zgodne z granicami komórek ewaluacji.
Jeden patch odpowiada części maski 64×64. Oceniana maska nie ma rozmiaru 28×28.

Korzystamy bezpośrednio z oficjalnego Faster R-CNN zamiast `localize()`:
to daje confidence i zachowuje puste detekcje. Wejście do tego niższego poziomu
to lista tensorów RGB CHW [0,1] (wrapper `yolo` właśnie do takiego zakresu
przelicza wejście [-1,1]). Warianty `default` odtwarzają zastępowanie braku
detekcji całym obrazem/patchem, a `empty` pozostawiają ten obszar czysty.
Warianty `empty` odrzucają też zdegenerowane bboxy. Nie filtrujemy dodatkowo klas.

Badamy top1 oraz sumy maksymalnie 4 i 20 bboxów. Dla patchy limity obowiązują
**na patch**, czyli suma może zawierać do 16 lub 80 prostokątów przed ich
połączeniem. Więcej prostokątów może zwiększyć false positives, a wycinanie
może usuwać kontekst i zmieniać skuteczność detektora.

## Ocena

Wszystkie 170 obrazów TrustMark, poprzedni 5-fold CV z seedem 42. Baseline
wykorzystuje zapisane predykcje OOF, z priory dopasowaną wyłącznie na pozostałych
136 obrazach w danym foldzie. Porównujemy też każdy wariant z pełnoobrazowym
top1. Wiele wariantów oceniono na tych samych foldach: wybór najlepszego nie
stanowi nowej, niezależnej walidacji. Bootstrap to ten sam warunkowy estymator
co w poprzednim eksperymencie; pełne ograniczenia w jego README.

Sprawdzamy również zgodność odtworzonych wariantów pełnoobrazowych top1/all4
z maskami zapisanymi z oryginalnego API. Zmiana limitu może ujawnić dodatkowe
prostokąty, które wcześniej były obcięte. Wagi modelu pozostają identyczne.

## Uruchomienie

Po poprzednim CV, z katalogu repozytorium:

```powershell
.\task-1\.venv\Scripts\python.exe task-1/experiments/trustmark_patches/run.py
.\task-1\.venv\Scripts\python.exe task-1/experiments/trustmark_patches/test_patches.py
```

`results/predictions.npz` przechowuje maski z nazwami, `detections.json` surowe
bboxy i confidence, `cv_metrics.json` wyniki per fold i sparowane różnice.
Ponowne uruchomienie używa cache predykcji. Skrypt niczego nie wysyła.
Testy sprawdzają globalne przesunięcia patchy, układ osi, fallback ograniczony
do pojedynczego patcha, kolejność confidence i limit liczby prostokątów.
