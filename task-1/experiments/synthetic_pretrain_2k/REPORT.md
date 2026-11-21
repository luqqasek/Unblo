# Czy 2000 syntetyków pomaga? StegaStamp

W tej recepturze syntetyczny pretrening poprawił średnie OOF mIoU.

**Kontrola real-only: 0.46129; synthetic→real: 0.50719; zmiana +0.04590 mIoU.** Bootstrap 95% CI parowanej zmiany: [+0.03495, +0.05725].

Poprawa wystąpiła w **9/9 par fold–seed**. Najważniejsze zastrzeżenie: wzrosło głównie IoU klasy czystej (0.4343 → 0.5337), a foreground IoU nie wzrosło (0.4883 → 0.4806). Ranking pikseli jest jednak lepszy: AUC 0.6713 → 0.7435, AP 0.5745 → 0.6022. To poprawa sygnału i redukcja fałszywych alarmów, a nie rozwiązanie lokalizacji małych watermarków.

**Sam pretrening bez fine-tuningu: 0.51153 mIoU.** Fine-tuning nie dał dodatkowej potwierdzonej przewagi w mIoU: parowany CI zmiany obejmuje zero. Jednocześnie poprawił AP/AUC, więc nie należy sprowadzać tego do całkowitego zapomnienia syntetyków.

## Co wykonano

Wygenerowano **2000 przykładów treningowych z 2000 różnych zdjęć COCO** oraz 200 przykładów syntetycznej walidacji z innych zdjęć. Oznakowanie prawdziwym encoderem StegaStamp ze starter kitu, 100 losowych surowych bitów. Zamrożony ekstraktor, korekta inverse-STN i niezmieniona głowica. Nie trenowano enkodera ani ekstraktora.

Każdy z seedów 42/7/123 dostał 10 ustalonych epok pretreningu. Następnie wykonano 3 zewnętrzne foldy × 3 seedy fine-tuningu oraz 9 dopasowanych kontroli real-only. Długość fine-tuningu i próg wybrano wyłącznie na trzech wewnętrznych foldach. Świeży refit zaczynał od tego samego syntetycznego checkpointu, z nowym optymalizatorem. Walidacja syntetyczna nie wybierała checkpointu. Obrazy testowe i leaderboard nie były użyte.

Plan zapisany przed wynikami: [README.md](README.md). Przegląd źródeł: [RESEARCH.md](RESEARCH.md). Badamy jedną rodzinę znaków, nie wszystkie trzy metody konkursu.

## Literatura i wybór generatora

[Watermark Anything (ICLR 2025)](https://arxiv.org/html/2411.07231v2) uzasadnia mieszanie obszarów czystych i oznakowanych z maskami różnego kształtu. [MaskWM (NeurIPS 2025)](https://arxiv.org/abs/2504.12739) rozwija uczenie odporności lokalnej, ale oba podejścia uczą także własne systemy znakowania: nie są gotowymi lokalizatorami znaków StegaStamp. Ten eksperyment stosuje zasady konstrukcji danych, nie deklaruje reprodukcji SOTA.

Źródło zdjęć: [oficjalne COCO](https://cocodataset.org/#download), archiwum val2017. Ta nazwa nie oznacza użycia walidacji konkursowej. Generator bierze środkowy kwadrat i skaluje do 512×512 **przed** dodaniem watermarku. Obiektowe maski pochodzą z adnotacji COCO, nie z masek konkursowych.

45% przykładów: znakowanie rzeczywistego cropa, potem wklejenie oznakowanych pikseli do tego samego czystego zdjęcia. 45%: znakowanie całego obrazu, potem przywrócenie czystego regionu. Po 5% całkiem czystych i całkiem oznakowanych. Maski: prostokąty, elipsy i obiekty COCO; brak JPEG po znakowaniu i brak generatywnego inpaintingu. Proporcje są hipotezą inżynierską, nie twierdzeniem z publikacji.

## Wyniki OOF

| Miara (średnia po obrazach i seedach) | Real-only | Synthetic→real | Zmiana | 95% CI zmiany |
|---|---:|---:|---:|---|
| miou | 0.46129 | 0.50719 | +0.04590 | [+0.03495, +0.05725] |
| fg_iou | 0.48833 | 0.48063 | -0.00770 | [-0.01834, +0.00294] |
| clean_iou | 0.43426 | 0.53375 | +0.09949 | [+0.08207, +0.11770] |
| auc | 0.67133 | 0.74350 | +0.07218 | [+0.05169, +0.09228] |
| ap | 0.57450 | 0.60215 | +0.02765 | [+0.01507, +0.04095] |
| topk | 0.55575 | 0.57947 | +0.02372 | [+0.01112, +0.03673] |
| delta_empty | 0.20732 | 0.25322 | +0.04590 | [+0.03495, +0.05725] |
| probability_separation | 0.09527 | 0.15572 | +0.06045 | [+0.04125, +0.07967] |
| brier | 0.11690 | 0.08014 | -0.03675 | [-0.04982, -0.02533] |

mIoU: oficjalna średnia per obraz, z oficjalną regułą pustej unii. AP i ROC-AUC to średnie per obraz, nie pooled pixel scores. Top-K wykorzystuje liczbę pikseli GT i jest wyłącznie diagnostyką; remisy rozstrzygnięto przez wartość oczekiwaną losowego wyboru. Wszystkie rzeczywiste maski zawierają obie klasy. Brier: niżej lepiej.

| Seed | Real-only OOF | Synthetic→real OOF | Zmiana |
|---|---:|---:|---:|
| 42 | 0.44441 | 0.51514 | +0.07073 |
| 7 | 0.46441 | 0.50436 | +0.03995 |
| 123 | 0.47505 | 0.50206 | +0.02701 |

| Fold | Seed | Kontrola | Synthetic→real | Zmiana | Epoka kontrola / synthetic | Próg kontrola / synthetic |
|---|---|---:|---:|---:|---|---|
| 1 | 42 | 0.46528 | 0.51817 | +0.05289 | 31 / 7 | 0.6 / 0.5 |
| 2 | 42 | 0.41896 | 0.49787 | +0.07891 | 19 / 5 | 0.4 / 0.6 |
| 3 | 42 | 0.44909 | 0.52965 | +0.08056 | 15 / 10 | 0.5 / 0.6 |
| 1 | 7 | 0.47504 | 0.51944 | +0.04439 | 32 / 4 | 0.7 / 0.6 |
| 2 | 7 | 0.46361 | 0.51275 | +0.04915 | 24 / 8 | 0.6 / 0.6 |
| 3 | 7 | 0.45441 | 0.48047 | +0.02606 | 28 / 12 | 0.4 / 0.5 |
| 1 | 123 | 0.47062 | 0.47572 | +0.00510 | 37 / 5 | 0.6 / 0.6 |
| 2 | 123 | 0.46501 | 0.50622 | +0.04121 | 24 / 7 | 0.5 / 0.6 |
| 3 | 123 | 0.48979 | 0.52465 | +0.03486 | 21 / 11 | 0.5 / 0.6 |

Poprawa średniej po seedach dotyczy 69.4% obrazów. SD OOF między trzema seedami: kontrola 0.01555, synthetic 0.00698. To 170 unikalnych obrazów, a nie 510 niezależnych obserwacji. CI obejmuje bootstrap obrazów po uśrednieniu seedów; nie obejmuje ponownego uczenia modeli ani niepewności wszystkich hiperparametrów. Jest to rozwojowy CV na wcześniej badanych danych.

## Małe maski, duże obszary i kalibracja

GT ≤5%: 45 obrazów. Foreground IoU 0.01786 → 0.01665; AP 0.04735 → 0.06147; AUC 0.52473 → 0.62719; delta względem pustej maski -0.06304 → -0.01082.

GT >50%: 85 obrazów. IoU klasy czystej: 0.07336 → 0.16980. To test odzyskiwania czystych otworów wewnątrz dużych oznakowanych obszarów.

Pełne przedziały wielkości: [size_stratified_metrics.csv](results/size_stratified_metrics.csv). Wyniki trybu znakowania: [embedding_mode_metrics.csv](results/embedding_mode_metrics.csv). Małe grupy <10 obrazów są oznaczone jako niewiarygodne do silnych wniosków. Progi zawsze dobrano wewnątrz treningu; nie przeszukiwano outer validation.

Przy wspólnym stałym progu 0.5: kontrola 0.46165, synthetic→real 0.50882. Efekt pretreningu pozostaje dodatni również bez doboru progu. Jest to dodatkowa diagnostyka, nie zmiana głównego protokołu po obejrzeniu wyników.

![Jakość zależnie od powierzchni](results/size_and_area.png)

## Czy pretrening nauczył się syntetycznej lokalizacji?

Końcowe mIoU na 200 osobnych syntetykach: **0.56390** (średnia trzech seedów; stały próg 0.5). Ten zbiór nie wybrał liczby epok.

| Seed | Synthetic-only na real170, t=0.5 | FG IoU | Macro AP |
|---|---:|---:|---:|
| 42 | 0.51112 | 0.48930 | 0.59002 |
| 7 | 0.51130 | 0.48104 | 0.59389 |
| 123 | 0.51217 | 0.50234 | 0.58982 |

Średnio synthetic-only osiąga 0.51153, a po fine-tuningu 0.50719. Dodatkowa zmiana po fine-tuningu: -0.00434, CI95 [-0.01269, +0.00411]. To porównanie pełnych procedur: synthetic-only używa stałego 0.5, a fine-tuning ma próg dobrany wewnątrz treningu. Nie izoluje czystego efektu aktualizacji wag od kalibracji.

Wynik syntetyczny i rzeczywisty nie są bezpośrednio porównywalne: różnią się domeną, geometrią i rozkładem powierzchni. Syntetyczna walidacja zawiera 10% pełnych/pustych masek, których nie ma w real170.

Synthetic-only nie korzysta z real-labels w treningu ani selekcji. Jego próg 0.5 nie jest kalibrowany na real. Wysoki wynik syntetyczny i niski rzeczywisty mogą wynikać z przesunięcia domeny; nie dowodzą automatycznie zapominania podczas fine-tuningu.

Dodatkowo zmierzono te same 200 syntetycznych obrazów po fine-tuningu, bez wybierania modelu na tej ocenie. Średnie AP/AUC wykluczają pełne i puste GT (20 z 200); mIoU obejmuje wszystkie 200. Wyniki po fine-tuningu uśredniają dziewięć modeli, a pretrain trzy modele.

| Model oceniany na syntetycznej walidacji, t=0.5 | mIoU | Macro AP | Macro AUC |
|---|---:|---:|---:|
| Po pretreningu | 0.56390 | 0.63073 | 0.79086 |
| Po fine-tuningu na real | 0.54569 | 0.61837 | 0.76264 |
| Kontrola real-only | 0.50095 | 0.57015 | 0.64014 |

[Retencja według trybu generacji](results/synthetic_holdout_by_mode.csv). To pomiar dodatkowy, nie kryterium wyboru hiperparametrów.

![Krzywe i parowane zmiany](results/learning_and_delta.png)

## Jakość i odtwarzalność danych

Walidacja pipeline sprawdza dokładnie niezmienione piksele poza maską, oficjalny downsampling 512→128 i zgodność nowych wyrównanych cech z aktywną bazą na trzech rzeczywistych obrazach / czterech poziomach. Zapisano wiadomości, identyfikatory i licencje COCO, hashe zdjęć i checkpointu, transformacje STN, bit accuracy, powierzchnię i PSNR.

Sprawdzono SHA i perceptual dHash względem wszystkich 510 obrazów treningowych konkursu oraz powtórzenia coverów. Heurystyka nie gwarantuje wykluczenia wszystkich wspólnych źródeł po silnej edycji; brak źródłowych identyfikatorów konkursu. Nie użyto testowych obrazów do takiego porównania.

[Jakość syntetyków](results/synthetic_quality.csv), [manifest](results/manifest.json), [preflight cech](results/feature_preflight.json), [konfiguracja](results/config.json), [konfiguracja generacji](results/generation_config.json). Bit accuracy dla całkiem czystych próbek odnosi się do losowej wiadomości i jest kontrolą negatywną. Dla maski pustej PSNR jest nieskończone i zapisane jako null, nie zero.

![Przykłady syntetyczne i residual](results/synthetic_example.png)

## Przykłady rzeczywiste

[Galeria: kontrola kontra synthetic→real](results/gallery/index.html). Obejmuje pogorszenia, poprawy i małe regiony. Selekcja przykładów jest diagnostyczna; pełne wyniki wszystkich obrazów znajdują się w [per_image_metrics.csv](results/per_image_metrics.csv).

## Koszt i granice wniosku

Suma zmierzonych czasów GPU: pretrening 3.4 min; fine-tuning 3.0 min; kontrola 6.3 min. Generacja i ekstrakcja cech CPU raportowane osobno w source_quality.json. Nie trenowano RGB/SRM ani nowego dekodera w tym eksperymencie.

Dodatkowy budżet pretreningu nie jest dopasowany liczbą kroków do real-only; badamy praktyczny pakiet „2k syntetyków + pretrening”, a nie izolowany efekt danych przy identycznej liczbie aktualizacji. Nie testowano innych liczebności, proporcji masek, niższego learning rate fine-tuningu ani mieszania real/synthetic podczas fine-tuningu. Samoblen­dowanie usuwa semantyczne szwy, ale nie odtwarza całej domeny generatywnych edycji konkursowych. COCO może być domeną znaną pretrained encoderowi; oceniamy transfer do real, nie nowość jego pierwotnych danych.

Historyczna baza CPU służy tylko dodatkowej kontroli. Główne wyniki pochodzą z dopasowanych nowych uruchomień na tym samym GPU, float32, TF32 off i deterministycznych operacjach. Zmiana urządzenia może zmienić wybór epoki; nie należy mieszać wyników CPU i GPU w jednym parowanym efekcie.

## Decyzja i kolejny eksperyment

Ta receptura uzasadnia dalszą pracę nad syntetykami dla StegaStamp. Trzy kolejne kroki, według priorytetu:

1. **Ostrożniejszy fine-tuning tej samej puli 2k**: porównać dotychczasowe LR=0.001 z LR=0.0001, dopuszczając epokę 0 jako kandydat w wewnętrznym wyborze. Obecna procedura wybierała wyłącznie epoki 1–50, mimo że checkpoint pretrain już działał dobrze. Stały checkpoint synthetic-only jest obowiązkową kontrolą. Koszt mały, bez nowej generacji; kryterium: parowana poprawa mIoU/foreground IoU bez utraty AP. Nie dobierać wariantu na outer validation.

2. **Powtórzyć 2k dla PixelSeal i TrustMark**, z ich dokładnymi encoderami i niezmienionymi bazami. Koszt głównie generacji/ekstrakcji. Obecny wynik nie pozwala zakładać takiej samej korzyści dla innych rodzin.

3. **Osobno zbadać małe regiony**: porównać obecną recepturę z kontrolowanym zwiększeniem udziału małych i cienkich oznakowanych fragmentów, zachowując liczbę 2k i kroki pretreningu. Hipoteza jest słabsza niż główny wynik: obecny generator już zawiera małe maski, a ich FG IoU nadal jest niskie. Wymaga wzrostu AP i FG IoU, nie tylko mIoU dzięki czystemu tłu; w razie braku efektu wrócić do jakości reprezentacji lokalnej.

Nie zwiększać równocześnie modelu, liczby danych i zakresu augmentacji. Nie ma jeszcze dowodu, że samo 10k zamiast 2k rozwiąże małe watermarki. Wyniki należy potwierdzić poza wielokrotnie używanym zbiorem rozwojowym przed uznaniem przewagi konkursowej.

Wynik nie odpowiada jeszcze za PixelSeal i TrustMark. Nie wygenerowano nowych predykcji testowych ani submission; nie zastąpiono aktywnej bazy automatycznie.

Pełne polecenia: [REPRODUCE.md](REPRODUCE.md). Wynik kontroli: [verification.json](results/verification.json).

## Zgłoszenie po fine-tuningu — aktualizacja

Na późniejsze, jawne polecenie użytkownika wykonano fine-tuning na wszystkich 170 realnych obrazach StegaStamp i predykcję testową. Zaakceptowane zgłoszenie **#1496**, publiczne mIoU **0.5235**, zmiana względem wcześniejszego najlepszego **+0.0112**.

[Pełny raport zgłoszenia](SUBMISSION_REPORT.md). Wcześniejsze stwierdzenia o nieużywaniu testu opisują etap eksperymentalny przed tą operacją.
