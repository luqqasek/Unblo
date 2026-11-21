# Lokalna rekonstrukcja wiadomości jako zadanie pomocnicze

Stan źródeł: 2026-11-21. Rekomendacja do walidacji, nie wynik eksperymentu.

**Wybrać maskowaną rekonstrukcję bitów ze wspólnego trzonu istniejącej głowicy segmentacji.** Wiadomość z metadanych jest wyłącznie etykietą. Zacząć od stałych wag `lambda=0, 0.1, 1`, zachowując segmentacyjne `0.5 BCE + 0.5 Dice`, zamrożone ekstraktory i dotychczasowy podział. Korzyść dla mIoU pozostaje hipotezą.

## Co dokładnie wynika z WAM

WAM wspólnie trenuje embedder i ekstraktor. Ekstraktor przewiduje obecność watermarku oraz K bitów w każdym pikselu. Równanie (5) dla pojedynczego obrazu normalizuje sumę BCE przez **K razy liczbę oznaczonych pikseli**:

`L_message = sum_i M_i * sum_k BCEWithLogits(z_ik, m_k) / (K * sum_i M_i)`.

Strata detekcji obejmuje wszystkie piksele. Bity czystego tła nie otrzymują etykiet. Artykuł stosuje K=32, rozdzielczość ekstraktora 256×256 oraz w pretreningu wagi detekcja=1, dekodowanie=10. Sam artykuł ostrzega, że trening jednej wiadomości sprzyja przewidywaniu stałej wiadomości w różnych regionach; rozróżnianie wielu wiadomości wymaga dodatkowego treningu. To ogranicza interpretację lokalności naszej próby. [Artykuł, §3–5.1, równania 4–5](https://arxiv.org/html/2411.07231v2#S4.SS2).

Oficjalny `PixelDecoder` ma wspólne warstwy upsamplingu oraz jedną końcową konwolucję 1×1 do K+1 kanałów. Rozdzielenie jej na dwie równoległe konwolucje 1×1 do 1 i K kanałów zachowuje ten rodzaj współdzielenia reprezentacji. [Kod PixelDecoder, linie 39–48 i 70–71](https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/watermark_anything/modules/pixel_decoder.py#L39).

**Różnice między równaniem a kodem:** implementacja wybiera maskowane elementy całego batcha dla każdej wiadomości, łączy je i stosuje `mean()`. Jest to średnia ważona liczbą dodatnich pikseli, a nie średnia strat znormalizowanych osobno na obraz. Domyślnie aktywne równoważenie ustala efektywne mnożniki `s_j = (w_j / sum(w)) * G / (eps + ||grad_j||)`, gdzie przy domyślnych ustawieniach G jest normą gradientu straty detekcji. Przy wyłączonym równoważeniu mnożniki są stałymi wagami. Puste maski nie dodają elementów do straty wiadomości. [Implementacja strat, linie 58–87 i 108–139](https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/watermark_anything/losses/detperceptual.py#L58).

Skrypt treningowy ustawia `balanced=True`, przekazuje **ostatnią warstwę embeddera** do obliczania norm gradientów i optymalizuje łącznie embedder oraz ekstraktor. Przykład pretreningu podaje `lambda_w2=10`, lecz przy domyślnym równoważeniu nie oznacza to stałego mnożnika 10. Nasz zamrożony ekstraktor nie ma takiego embeddera w grafie treningu. Dlatego nie kopiujemy ani wartości 10, ani mechanizmu adaptacyjnego. [Oficjalny train.py](https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/train.py).

## Dane i ryzyko skrótów

Odczyt [metadata.json](../../data/train/metadata.json) wykazał po 170 obrazów i 170 różnych wiadomości na metodę:

| Metoda | K | Udział jedynek |
|---|---:|---:|
| PixelSeal | 256 | 0.502068 |
| TrustMark | 100 | 0.493471 |
| StegaStamp | 100 | 0.499882 |

To opis próby, nie dowód niezależnego losowania bitów. [Wrappery](../../wmloc/watermarks.py) przekazują PixelSeal surowe bity jako `msgs`, TrustMark jawnie używa `use_ECC=False`, a StegaStamp dostaje 100 bitów bezpośrednio jako tensor `secret`. W tych ścieżkach nie ma dodatkowego kodowania ECC; nie przenosimy założeń o payloadzie/ECC z demonstracji innych wersji modeli. Dla diagnostyki punkt odniesienia majority-bit należy policzyć wyłącznie na części treningowej, osobno dla każdej pozycji bitu.

## Porównanie trzech możliwości

| Wariant | Potencjalna korzyść | Ograniczenie i decyzja |
|---|---|---|
| Maskowana rekonstrukcja bitów | Gradient wiadomości zmienia tę samą reprezentację, z której powstaje maska | Najmniejsza kontrolowana zmiana; wybrana do pierwszej próby |
| Warunkowanie cech znaną wiadomością | Może nauczyć zgodności obrazu z oczekiwaną wiadomością | Przy 170 przykładach łatwo o skróty; wiadomość podana także dekoderowi umożliwiałaby skopiowanie etykiety. Osobny przyszły eksperyment segmentacyjny |
| Lokalna korelacja z wiadomością | Bezpośredni sygnał zgodności `mean_k((2m_k-1)*z_ik)` | Wymaga wiarygodnych lokalnych logitów. Globalnie trenowany dekoder i duże pola recepcyjne nie gwarantują lokalności; nie mieszać jej teraz ze zmianą celu treningowego |

To porównanie jest wnioskowaniem projektowym na podstawie naszej architektury, nie rankingiem opublikowanych wyników. Wrappery PixelSeal i TrustMark udostępniają projekcje przestrzenne globalnych dekoderów, lecz istniejący [cache eksperymentu](../frozen_extractor/run.py) zawiera cztery skale aktywacji, nie dowód lokalnego odczytu wiadomości.

## Minimalna implementacja i uczciwa kontrola

Zachować projekcje czterech skal do 32 kanałów oraz dotychczasowy trzon `Conv32/GN/ReLU → Conv32/ReLU`. Segmentacyjny `Conv1x1(32,1)` i interpolacja do 128×128 pozostają identyczne. Dopiąć `Conv1x1(32,K)` na wspólnej mapie 32×32. Nie odłączać jej gradientu od trzonu. Metadane nie trafiają na wejście żadnej części dekodera.

Do nadzoru bitów użyć `a = avg_pool2d(mask128, 4)` i liczyć per obraz:

`L_b = sum_u a_bu * mean_k BCEWithLogits(z_bku, m_bk) / max(sum_u a_bu, eps)`.

Uśredniać po obrazach z dodatnią masą; gdy wszystkie są puste, zwrócić różniczkowalne zero. Nie mnożyć logitów ani targetów przez maskę przed BCE. Nie używać przewidzianej maski do ważenia straty, bo model mógłby obniżać koszt przez jej wygaszanie. Całkiem czysta komórka ma zerowy gradient bitów; komórka graniczna dostaje wagę proporcjonalną do pokrycia. Nie wymuszać na tle zer, przeciwnej wiadomości ani prawdopodobieństwa 0.5.

**Rozdzielczość 32×32 jest naszą adaptacją obliczeniową.** Ogranicza tensor bitów 16-krotnie względem 128×128; odpowiada rozdzielczości wspólnych cech, ale nie odtwarza pełnej przestrzennej straty WAM. Miękkie pokrycie zachowuje udział małych obszarów, który ginąłby przy progowaniu. Dla logitu stałego w komórce wzór jest równoważny ważeniu wszystkich jej dodatnich pikseli; nie jest równoważny BCE po biliniowym upsamplingu logitów. Nie twierdzimy, że interpolacja odtwarza utraconą informację lokalną.

Kontrola `lambda=0` ma mieć identyczne początkowe wagi segmentacji i trzonu, kolejność minibatchy, optymalizator i harmonogram. Najprościej utworzyć pełny model w tej samej kolejności dla wszystkich wag i porównać jego stan początkowy; inicjalizacja dodatkowego modułu nie może zmienić losowania bazowych wag ani kolejności danych. Świeża kontrola powinna odtworzyć pierwszy zewnętrzny fold: PixelSeal 0.519131, TrustMark 0.512931, StegaStamp 0.465554 (wartości referencyjne istniejącego protokołu przekazane do tego badania).

Epokę i próg nadal wybiera wyłącznie mIoU segmentacji na wewnętrznych 20%; bity są diagnostyką. Porównać accuracy na dodatnich komórkach z majority-bit i przetasowanymi wiadomościami; accuracy na tle tylko opisać. Poprawa bitów bez poprawy maski nie spełnia celu. Wynik pierwszego zewnętrznego foldu służy screeningowi; po wyborze wariantu nie przedstawiać go jako niezależnego potwierdzenia przewagi. Kontynuacja CV powinna przestrzegać istniejącej bramki eksperymentalnej.

Zamrożone cechy mogą nie zawierać odzyskiwalnych lokalnych bitów; przy zaledwie 170 wiadomościach model może też zapamiętać zbiór albo rozgłaszać globalnie odczytane bity. Nawet przewaga nad majority-bit nie dowodzi lokalności. To uzasadnia małe wagi i decyzję opartą o segmentację, a nie zakładanie sukcesu na podstawie WAM.
