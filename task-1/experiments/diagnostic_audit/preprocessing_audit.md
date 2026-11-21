# Audyt danych, metryki i geometrii

## Zweryfikowano

- 510 unikalnych hashy obrazów RGB, 170 obrazów/metodę; każdy 512×512×3. Brak identycznych obrazów; nie dowodzi to braku podobnych obrazów z tego samego źródła.
- Każdą maskę 512×512 sprawdzono względem oficjalnego `masks_128.npz`: **510/510 dokładnie zgodnych** po podziale na bloki 4×4 i progu udziału >=0,5.
- Piksel dodatni to PNG >127, czyli watermark obecny. Obrazy i maski łączone po image_name, nie po pozycji w przypadkowo posortowanym katalogu.
- Żadna maska GT nie jest całkowicie pusta; wszystkie per-image AUC/AP w tym audycie mają obie klasy. Kod obsługuje jednak przypadki jednorodne jako NaN z powodem.
- Syntetyczna geometria: prostokąt o znanym położeniu, dokładny próg połowy komórki, test kanałów RGB, odwracalność affine na punktach. To testy współrzędnych, bez generowania treningowych watermarków.
- Wszystkie zapisane targety OOF są identyczne z oficjalnymi maskami, nazwy unikalne, pokrycie 170/170 i train/validation rozłączne.
- Nie ma crop, letterbox ani zmiany proporcji, bo źródła i wejścia są kwadratowe. Maski nie są interpolowane bilinearnie w treningu głowicy: używa się oficjalnego targetu 128×128. Bilinear dotyczy logitów/cech.
- Galeria pokazuje RGB, GT, probabilities, predykcję, overlay oraz FP/FN na wspólnej siatce. Sprawdzenie wizualne geometrii jest uzupełnieniem równości masek; niewidocznego watermarku nie można potwierdzić wzrokiem z samego RGB.

## Oficjalna metryka

Import bezpośrednio z `starter-kit/task-1/wmloc/utils.py`: `miou_image`. Dla każdego obrazu liczymy IoU klasy 0 i 1, a następnie ich średnią. Pusta unia daje 1, nie 0. Wynik oficjalny jest średnią obrazów; pooled pixel IoU ma osobną nazwę i nie zastępuje wyniku konkursowego.

Przy GT foreground f>0 pusta predykcja daje (1−f)/2. Dla GT całkowicie czystego daje 1. Odpowiednie testy graniczne są w `verify.py`.

## Zgodność preprocessingów

Kod używa wrapperów lokalnego startera; szczegóły są w audycie architektury. TrustMark API upstream stosuje PIL bilinear + konwersję uint8, podczas gdy wrapper konkursowy używa torch bilinear antialias na float. Nie są to gwarantowanie bit-identyczne ścieżki, ale **trening, OOF i finalne inferencje w repo używają konsekwentnie tej samej ścieżki wrappera**. Nie zmieniono jej podczas audytu. Nie zakładamy, że błędy dekodowania wynikają z samej różnicy implementacji resize.

StegaStamp otrzymuje RGB [0,1] NHWC; odjęcie 0,5 występuje wewnątrz wykonanego grafu przed convolutional decoderem. Istotna geometria STN została odtworzona z węzłów i stałych ONNX, zamiast automatycznie przenosić konwencje z aktualnego repo autorów.

## Definicje analiz dodatkowych

- ROC-AUC i AP: dokładne sklearn, macro po obrazach z obiema klasami oraz pooled po wszystkich pikselach. AP to Average Precision, nie trapezowa PR-AUC. Prevalence jest odniesieniem losowym. Duże maski mogą dawać wysokie AP bez dobrego wydzielania czystych dziur.
- Top-K: K z liczby dodatnich pikseli GT, sortowanie stabilne przy remisach. **Oracle-area**, nie reguła inferencji. Precision@K=Recall@K. Raportowany rozmiar remisu przy odcięciu.
- Progi oracle: 0,00–1,00 co 0,01 oraz 1,000001 jako jawna pusta maska. Dla każdego obrazu maksimum; żadnej takiej predykcji nie wysłano.
- Globalny próg wybrany na tym samym OOF jest resubstitution i zawyża ocenę. Także cross-OOF próg z pozostałych foldów nie jest nową bezstronną oceną: modele tych foldów uczyły się na obrazach bieżącego foldu. Tylko zapisane progi operacyjne mają oryginalny nested protokół.
- ECE: 15 równych przedziałów probabilities, pooled pixels. BCE/Brier: per-image mean, potem macro. Nie wykonywano temperature scaling ani isotonic calibration.
- Komponenty: 8-connectivity; holes: binary_fill_holes, komponenty otworów według domyślnego 4-connectivity scipy. Bbox w przedziale półotwartym; centroid w komórkach 128×128. Aspect ratio nie identyfikuje typu edycji.
- Boundary F1: odległość euklidesowa 1/2/4 komórek. Wariant podstawowy obejmuje ramkę obrazu. **Dodatkowy internal F1 wyłącza ramkę**, by pełna predykcja nie dostawała punktów za brzegi obrazu mimo pominięcia otworów. Jeśli jedna granica pusta, F1=0; jeśli obie puste, NaN z przyczyną.
- Grubość: dwa razy distance transform na lokalnych maksimach 3×3, minimum i mediana. Jest to przybliżenie, nie dokładny medial-axis thickness.
- Bootstrap: 2000 losowań obrazów, stały seed 20261121, percentile 95%; różnice parowane po image_id. Warunkowe na zapisanych OOF modelach; nie obejmują niepewności ponownego trenowania i wyboru eksperymentów.
- Bin n<10 oznaczony jako niewiarygodny liczebnościowo. Brakujące metryki w CSV puste/NaN, JSON null; nie zastępujemy ich zerem.

## Ograniczenia

To audyt rozwojowy tych samych 510 obrazów. Nie jest nowym niezależnym testem. Nie inferujemy rodzaju edycji z kształtu maski. Nie potwierdzono grouping po pierwotnym obrazie źródłowym, bo metadata nie udostępnia takiej tożsamości. Hashe cache zapisane dopiero teraz nie stanowią historycznego dowodu, dlatego wykonano dodatkowe odtworzenie forwardów i logits.
