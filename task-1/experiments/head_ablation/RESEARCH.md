# Jak uczyć lokalizacji watermarku i czystych pikseli

Przegląd źródeł pierwotnych, 2026-11-21. Analiza dotyczy obecności watermarku w pikselu; odróżnia ją od odczytu bitów wiadomości i wykrywania manipulacji. Nie uruchamiano nowego treningu.

**Obecny BCE już uczy obu klas. Najlepiej uzasadniony kolejny krok to kontrolowane porównanie zwykłego BCE, łagodnego równoważenia klas na obraz i focal loss, przy zachowaniu tej samej głowicy i podziałów.** Literatura nie uzasadnia tezy, że do uczenia czystych pikseli trzeba dodać Dice tła ani że focal jest standardem lokalizacji niewidocznych watermarków.

## Co wynika z naszego eksperymentu

W [kodzie treningu](run.py) podstawowy cel to `0.5 BCE + 0.5 Dice_watermark`. Symetryczny wariant zmienił go na `0.5 BCE + 0.25 Dice_watermark + 0.25 Dice_clean`: jednocześnie dodał cel tła i zmniejszył o połowę wagę Dice watermarku. Spadek pierwszego foldu z około 0.4992 do 0.4902 nie izoluje więc samego wpływu uczenia tła. Wariant większej rozdzielczości osiągnął około 0.4577; oba nie przeszły ustalonej bramki. To screening jednego foldu, nie dowód ogólnej wyższości funkcji straty. [Wyniki i protokół](REPORT.md).

Dla `p=sigmoid(z)` BCE wynosi `-y log(p) -(1-y) log(1-p)`. Dla czystego piksela (`y=0`) pozostaje `-log(1-p)`, a pochodna po logicie wynosi `p`; fałszywie dodatni piksel dostaje silny sygnał obniżenia predykcji. Dla watermarku pochodna to `p-1`. Także foreground Dice karze fałszywe dodatnie przez sumę predykcji w mianowniku. To bezpośrednie własności wzorów użytych w [run.py](run.py), nie hipoteza o wyniku treningu.

[Pomiar masek treningowych 128×128](results/class_balance.json) obejmuje 170 obrazów każdej metody:

| Metoda | Udział pikseli watermarku | Obrazy z <10% watermarku | Obrazy z >90% watermarku |
|---|---:|---:|---:|
| PixelSeal | 49.3941% | 54 | 53 |
| TrustMark | 50.3260% | 61 | 55 |
| StegaStamp | 49.2061% | 65 | 60 |

Nie ma masek całkowicie pustych ani pełnych. **Globalnie klasy są zrównoważone, lecz wiele pojedynczych obrazów ma skrajną proporcję klas.** Wniosek dla tego zbioru: duże globalne `pos_weight` nie ma uzasadnienia w częstości klas; ewentualne równoważenie należy rozważać na poziomie obrazu. Statystyki służą diagnozie; parametry dobieramy wyłącznie na odpowiednim podzbiorze treningowym.

## Co rzeczywiście robią metody watermarkingu

**Watermark Anything / WAM to najbardziej bezpośredni punkt odniesienia.** Przewiduje osobno obecność watermarku i wiadomość w każdym pikselu. Strata detekcji to średni BCE po wszystkich pikselach, a strata wiadomości to BCE po bitach tylko wewnątrz watermarku. Artykuł podaje `lambda_det=1`, `lambda_dec=10` dla pretreningu. Maska powstaje przez mieszanie obrazu oryginalnego z watermarked według losowej maski; transformacje geometryczne obejmują również etykietę. Inwersja maski ma prawdopodobieństwo 0.5. To uczenie dodatnich i ujemnych przykładów poprzez dane, bez Dice tła. [WAM, §4.2, równania 4–5 i §5.1](https://arxiv.org/html/2411.07231v2#S4.SS2).

W oficjalnej konfiguracji maski prostokątne, nieregularne, pełne i segmentacyjne mają po 0.25 prawdopodobieństwa wyboru; `invert_proba=0.5`. Inwersja jest operacją podczas syntezy wejścia, więc nie wolno jej kopiować jako odwrócenia samej etykiety już gotowego obrazu. [Konfiguracja augmentacji](https://github.com/facebookresearch/watermark-anything/blob/main/configs/all_augs.yaml). Przykładowe komendy repo podają stosunek detekcja/wiadomość 1/10 w pretreningu i 1/6 w finetuningu. `balanced=True` w skrypcie oraz `calculate_adaptive_weights` skalują składniki według norm gradientów przy ostatniej warstwie embeddera: **to równoważenie różnych celów treningowych, nie wag czystej i oznaczonej klasy**. Nie należy interpretować 1/10 jako proporcji klas. [Skrypt](https://github.com/facebookresearch/watermark-anything/blob/main/train.py), [implementacja BCE i wag](https://github.com/facebookresearch/watermark-anything/blob/main/watermark_anything/losses/detperceptual.py).

**PixelSeal / VideoSeal nie dostarczają równoważnego dowodu dla naszej głowicy segmentacyjnej.** PixelSeal opisuje BCE rekonstrukcji wiadomości oraz cel adversarialny dla niewidoczności; „adversarial-only” oznacza rezygnację z dodatkowych strat percepcyjnych, nie rezygnację z BCE wiadomości. W pierwszej fazie uczy samą wiadomość, potem dodaje dyskryminator. Celem jest odczyt wiadomości z obrazu, nie maska obecności watermarku. [PixelSeal, §3.2–4.2](https://arxiv.org/html/2512.16874v1#S4.SS1).

Wspólny kod VideoSeal zawiera zarówno BCE detekcji, jak i BCE dekodowania, ale skrypt domyślnie ma `lambda_det=0`, `lambda_dec=1`, `balanced=False`. Sama obecność `detection_loss` w bibliotece nie dowodzi użycia tego celu przez wydany checkpoint. Przykład treningu obrazowego ustawia wagi wiadomość/dyskryminator/percepcyjna na 1/0.1/0.1; są to wagi celów, nie klas. [Skrypt treningowy](https://github.com/facebookresearch/videoseal/blob/main/train.py), [straty](https://github.com/facebookresearch/videoseal/blob/main/videoseal/losses/videosealloss.py), [przykładowa receptura VideoSeal](https://github.com/facebookresearch/videoseal/blob/main/docs/training.md). Nasza lokalizacja na zamrożonych cechach tych modeli jest dodatkowym zadaniem; nie dziedziczy automatycznie ich jakości odczytu bitów.

**TrustMark również wymaga rozróżnienia zadań.** API zwraca wiadomość i flagę obecności, a obecna dodatkowa funkcja `localize()` uruchamia detektor YOLO i zwraca prostokąty. To nie gęsta maska pikselowa. [Oficjalny kod `localize` i `subimage_decode`](https://github.com/adobe/trustmark/blob/main/python/trustmark/trustmark.py). Wzmianka o *focal frequency loss* w dokumentacji dotyczy innego celu niż klasyfikacyjny focal loss Lin et al.; nie jest dowodem używania focal do maski. [FAQ Adobe](https://github.com/adobe/trustmark/blob/main/FAQ.md).

**OmniGuard i EditGuard lokalizują manipulacje.** OmniGuard uczy ekstraktor maski manipulacji przez `BCE + 20 × edge_loss` po treningu watermarkingu, na 20 tys. skonstruowanych trójek; trening manipulacji wykorzystuje Stable Diffusion Inpaint. To rzeczywisty przykład gęstej predykcji z BCE, ale etykieta oznacza edycję, a wejście korzysta również z odzyskanego watermarku lokalizacyjnego. [OmniGuard, CVPR 2025, §3.6, równanie 10 i §4.1](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_OmniGuard_Hybrid_Manipulation_Localization_via_Augmented_Versatile_Deep_Image_Watermarking_CVPR_2025_paper.pdf).

EditGuard uczy odtwarzania obrazowego watermarku z L1/L2; maskę manipulacji uzyskuje przez progowanie błędu rekonstrukcji (`tau=0.2`). Jego faza lokalizacji używa wag `alpha=100`, `beta=1` w równaniu 6, bez tamperowanych próbek treningowych. To nie receptura BCE/Dice dla maski obecności. [EditGuard, CVPR 2024, §4.5, równania 6–7](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_EditGuard_Versatile_Image_Watermarking_for_Tamper_Localization_and_Copyright_Protection_CVPR_2024_paper.pdf). Brak watermarku nie oznacza ogólnie manipulacji: obraz mógł nigdy nie być oznaczony; po edycji część sygnału może też pozostać. Dlatego wyników tych zadań nie przenosimy wprost na naszą metrykę.

## Focal, Dice i Lovász: co warto sprawdzić

Focal loss ma postać `-(1-p_t)^gamma log(p_t)` z opcjonalną wagą klasy. Zmniejsza udział łatwych pikseli **obu klas**, koncentrując gradient na trudnych przykładach. Oryginalny dowód dotyczy RetinaNet i silnej nierównowagi łatwych negatywów w detekcji obiektów, nie watermarków. [Lin et al., ICCV 2017, §3 i §5](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf). Nie znaleziono w sprawdzonych powyżej celach treningowych potwierdzenia używania klasyfikacyjnego focal jako standardowej straty maski watermarku. Jest uzasadnioną hipotezą eksperymentalną, bez obietnicy poprawy.

W torchvision `alpha=0.25` waży dodatnie piksele przez 0.25, ujemne przez 0.75; nie jest neutralnym ustawieniem. Dla naszych globalnie zrównoważonych danych proponuję początkowo `alpha=-1` (bez wag klas), `gamma=1` i dopiero `gamma=2`, redukcja `mean`. `alpha=0.5` jest symetryczne, ale dodatkowo mnoży cały focal przez 0.5, co zmienia jego relację do Dice. [Implementacja torchvision](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html). Przy podmianie BCE przez focal trzeba mierzyć skalę i gradienty składników; identyczne współczynniki nie gwarantują identycznego ich udziału.

Lovász jest przybliżeniem celu IoU; autorzy udostępniają binarny Lovász-hinge i wieloklasowy Lovász-softmax. To inna motywacja niż dobór trudnych pikseli. [Artykuł](https://arxiv.org/abs/1705.08790), [kod autorów](https://github.com/bermanmaxim/LovaszSoftmax). Rozważyłbym go dopiero po prostszych kontrolach. Nie ma tu dowodu, że pokona BCE albo naprawi ograniczenia zamrożonych cech.

Proponowana kolejność, bez uruchamiania tych prób w ramach tego przeglądu:

1. **Kontrole BCE:** czysty BCE oraz `0.75 BCE + 0.25 Dice_watermark`, obok obecnego `0.5/0.5`. Sprawdzają, czy dodatkowa optymalizacja overlapu pomaga w tym zbiorze.
2. **Łagodne równoważenie na obraz:** dla obrazu policzyć osobno średni BCE watermarku i czystego obszaru, a następnie ich średnią. Aby mała wyspa nie dominowała, rozpocząć od mieszanki `0.5 BCE_pixel + 0.5 BCE_class_per_image`, ewentualnie ograniczyć wagi odwrotnej częstości. To nasza hipoteza wynikająca z rozkładu masek, nie receptura WAM. Przy nieobecnej klasie liczyć tylko klasy obecne; próg wygładzania ustalić z części treningowej.
3. **Focal:** podmiana składnika BCE przy stałej architekturze; najpierw `gamma=1`, następnie `2`, `alpha=-1`. Porównać również wartości i normy gradientów focal/Dice, żeby wiedzieć, czy poprawę daje wybór trudnych pikseli, czy zmniejszona siła składnika pikselowego.
4. Raportować osobne IoU czystych i oznaczonych pikseli, FP/FN, przewidywany udział watermarku oraz wyniki w koszykach powierzchni maski, obok oficjalnej metryki. Próg i epokę dobierać wewnątrz treningu; te same podziały i seedy. Jeżeli żaden wariant nie poprawia rankingu pikseli, zbadać cechy i dystrybucję danych zamiast rozbudowywać stratę.

Nie ustalono jeszcze, czy główny problem to nadmiar FP, FN, błędy granic, kalibracja czy brak informacji przestrzennej w cechach. Wynik jednego foldu ani porównanie różnych pełnych systemów watermarkingu tego nie rozstrzyga. Wnioski o maskach syntetycznych WAM uzasadniają osobną przyszłą próbę syntezy par czysty/oznaczony i różnorodnych obszarów, jeżeli dostępne będą pasujące obrazy źródłowe; nie uzasadniają zmiany samych istniejących etykiet.
