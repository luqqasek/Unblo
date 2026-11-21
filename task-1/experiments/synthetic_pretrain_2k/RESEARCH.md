# Pretraining na 2000 syntetycznych obrazach: podstawy i protokół

Stan sprawdzenia źródeł: **21 listopada 2026**. To notatka projektowa, nie raport z zakończonego treningu. Proponowane proporcje i hiperparametry są hipotezami do sprawdzenia na prawdziwej walidacji.

## Wniosek dla eksperymentu

Warto sprawdzić pretraining głowicy lokalizacyjnej na syntetycznych parach obraz–maska, a następnie fine-tuning na dostępnych danych rzeczywistych. Generator powinien korzystać z **dokładnych, zamrożonych encoderów zadania**, zamiast tworzyć nowy rodzaj znaku wodnego. Dla StegaStamp zachować inverse-STN i porównać z tymi samymi foldami oraz seedami aktywnej bazy. Taką bazę i protokół porównania określają [instrukcje zadania](../../AGENTS.md) oraz [opis eksperymentów](../README.md).

## Co rzeczywiście wynika z literatury

| Źródło pierwotne | Ustalenie istotne tutaj | Granica zastosowania |
|---|---|---|
| [Watermark Anything, ICLR 2025, wersja 2](https://arxiv.org/html/2411.07231v2) | Wspólny trening encodera i segmentującego extractora; maski pełne, prostokątne, nieregularne i obiektowe, także odwrócone; mieszanie obszarów znakowanych i czystych. Przy większej rozdzielczości skaluje się residual, a nie cały obraz wynikowy. | To uzasadnienie konstrukcji danych i celu segmentacji. Gotowy WAM wykrywa własny znak; nie wykazano, że zastąpi lokalizator StegaStamp, TrustMark czy PixelSeal. |
| [Mask Image Watermarking / MaskWM, NeurIPS 2025](https://arxiv.org/abs/2504.12739v3) | Maskowanie przy dekodowaniu uczy ekstrakcji z lokalnych obszarów; wariant ED uwzględnia maskę także w encoderze. Autorzy raportują przewagę nad WAM w swoim protokole. | Zmieniony encoder nie pasuje automatycznie do znaków już obecnych w danych konkursowych. |
| [Pixel Seal, grudzień 2025](https://arxiv.org/abs/2512.16874) | Trzy fazy treningu, adversarialny cel jakości, adaptacja wysokiej rozdzielczości i JND rozwiązują problemy odporności oraz widoczności znaków. | To trening systemu znakowania. Nie stanowi dowodu przewagi konkretnej głowicy lokalizacji w naszym zadaniu. |
| [VideoSeal — oficjalny kod i karty modeli](https://github.com/facebookresearch/videoseal) | PixelSeal i VideoSeal są osobnymi kartami/modelami w jednym repozytorium; PixelSeal ma 256 bitów. | Nie zastępować karty `pixelseal` domyślnym `videoseal` tylko dlatego, że API jest wspólne. |
| [TrustMark — konfiguracja autorów](https://github.com/adobe/trustmark/blob/main/python/CONFIG.md) | Residual powstaje w 256×256, potem jest skalowany. Przy proporcji większej niż 2:1 implementacja może znakować wyłącznie centralny kwadrat. Warianty modelu nie są wzajemnie zgodne. | Etykieta całego wydłużonego patcha byłaby błędna. W zadaniu używany jest Q bez ECC. |
| [StegaStamp — oryginalny encoder](https://github.com/tancik/StegaStamp/blob/master/encode_image.py) | Oryginalny skrypt przygotowuje obraz 400×400 i korzysta z BCH. | Wrapper zadania używa innego protokołu obrazu wynikowego i surowych 100 bitów; uruchomienie oryginalnego skryptu nie odtworzy danych zadania. |
| [What Breaks Local Watermarks?, AISec 2026](https://arxiv.org/abs/2609.16832) | Benchmark 55 transformacji ocenia m.in. WAM, MaskWM, TrustMark i PixelSeal. Autorzy rozdzielają odzyskanie wiadomości od lokalizacji; geometria i generatywne edycje lokalne bywają szczególnie trudne. MaskWM ma najlepszą odporność w tym zestawieniu, kosztem jakości obrazu czystego. | To wynik konkretnego benchmarku, nie uniwersalny ranking i nie pomiar na danych510. |

**Nie określamy WAM aktualnym, bezwarunkowym SOTA.** Jest bardzo trafnym punktem odniesienia metodologicznym; nowsze prace używają innych modeli, transformacji i kompromisów. WAM trenuje także encoder, więc jego dwuetapowy trening nie jest tym samym eksperymentem co nasz synthetic pretrain → real fine-tune.

## Dokładny protokół encodera zadania

Autorytatywne dla reprodukcji konkursu są lokalne [wrappery](../../wmloc/watermarks.py), [kontrakt danych](../../README.md) i [loader](../../wmloc/data.py). Wrappery deklarują zgodność z encoderami użytymi do przygotowania danych.

| Metoda | Parametry i kolejność operacji |
|---|---|
| PixelSeal | 256 bitów; przypięta karta `pixelseal_card.yaml`; `model.embed(..., is_video=False)` dostaje patch w jego rzeczywistych wymiarach. Zachować domyślną siłę i kwantyzację wrappera. Decoder pracuje w 256×256. |
| TrustMark | Q, 100 surowych bitów, `use_ECC=False`; przekazanie patcha przez PIL do `tm.encode`. Ograniczyć proporcje patcha do 2:1. Encoder i decoder mają różne rozdzielczości: odpowiednio 256 i 245. |
| StegaStamp | Patch → bicubic 400×400 → residual ONNX → odjęcie średniej każdego kanału → bilinear do wymiarów patcha → dodanie do oryginalnych pikseli → clip i uint8. 100 surowych bitów. Zachować dokładnie tę kolejność. |

Obraz konkursowy ma 512×512. Generować znak **przed resize'em do wejścia extractora**. Samo oznakowanie obrazu już pomniejszonego do 400/256 i ponowne skalowanie daje inną strukturę residualu. Prostokąty mogą osiągać proporcję 16:1 dla PixelSeal i StegaStamp, ale tylko 2:1 dla TrustMark. [Kontrakt lokalny](../../README.md).

## Proponowany generator 2k

Poniższy przepis jest propozycją implementacyjną, a nie zapożyczonym wynikiem naukowym.

1. Przygotować czyste obrazy bazowe 512×512. Wszystkie operacje przygotowania treści wykonać przed znakowaniem. Użyć prawdziwych czystych coverów lub niezależnego zewnętrznego zbioru; obecny obraz konkursowy nie jest czystym coverem tylko dlatego, że istnieje jego maska.
2. Losować niezależne, równomierne wiadomości długości właściwej dla metody. Zapisać seed, źródło, hash covera, encoder/checkpoint, wiadomość, prostokąt i typ edycji.
3. Rozróżnić **lokalny encoder**: wyciąć prostokąt `P`, wywołać `embed(P,bits)`, wkleić go w ten sam obraz. Początkowa maska to prostokąt.
4. Rozróżnić **globalny encoder z odtworzeniem czystego fragmentu**: oznakować cały cover `C`, następnie wybrane piksele przywrócić z `C`. Maska to pozostały znakowany obszar. `full_watermark=True` w metadanych oznacza ten sposób powstania obrazu, a nie koniecznie maskę z samych jedynek. [Semantyka loadera](../../wmloc/data.py).
5. Dla obu trybów odtwarzać czyste fragmenty maską prostokątną, eliptyczną, wielokątną lub pociągnięciami pędzla; maskę docelową aktualizować jako przecięcie poprzedniej maski z dopełnieniem odtworzenia. Daje to dokładne etykiety bez wnioskowania, czy generatywny inpainting faktycznie usunął residual.
6. Na początek proponuję 50% próbek lokalnych i 50% globalnych z czystym odtworzeniem, z geometrią różnorodną i bez skrajnych augmentacji niszczących znak. Rozkład można dopasować wyłącznie do części treningowej folda. Udział negatywów i pełnych masek sprawdzić jako osobną ablacją, nie cicho zmieniać go po obejrzeniu outer validation.
7. Zapisać maskę 512×512, potem użyć oficjalnego downsamplingu: komórka 128×128 jest dodatnia przy co najmniej 8 dodatnich pikselach z bloku 4×4. [Implementacja metryk](../../wmloc/utils.py). Próbki zapisywać bezstratnie; JPEG i inne zaburzenia dodawać dopiero jako jawny wariant eksperymentu.

Do pierwszego eksperymentu StegaStamp przeznaczyć pełne **2000 przykładów na tę metodę**. Jeśli zakres obejmuje wszystkie trzy metody, jawnie określić, czy 2000 jest liczbą łączną, czy liczbą na encoder; nie raportować 2000 wariantów jako 2000 niezależnych coverów. Maski proceduralne nie odtwarzają semantyki generatywnych edycji, dlatego fine-tuning na danych rzeczywistych pozostaje niezbędną hipotezą, a nie formalną gwarancją transferu.

## Ochrona przed przeciekiem i uczciwe porównanie

To zalecenia metodologiczne tego eksperymentu.

- Najpierw ustalić zewnętrzne foldy. Jeśli cover albo maska pochodzi z real510, tworzyć pochodne wyłącznie z części treningowej danego folda. Wszystkie warianty jednego źródła muszą pozostać razem. Globalny pretraining ze wszystkich real510 unieważni późniejszy wynik OOF.
- Syntetyczne trening/validation dzielić po źródle covera, nie po wygenerowanej nazwie. Przy ponownym użyciu jednego obrazu nie mieszać jego wariantów między podziałami. Dla źródeł zewnętrznych zapisać pochodzenie i sprawdzić duplikaty; dokładny SHA nie wykryje resized/cropped near-duplicates.
- Nie korzystać z testowych obrazów ani wiadomości do generacji lub selekcji hiperparametrów. Parametry masek, próg i epokę dobierać w wewnętrznym podziale real-train; wynik outer fold służy wyłącznie ocenie.
- Wagi synthetic-pretrain mogą być wspólne dla foldów tylko wtedy, gdy ich trening i dobór nie wykorzystały żadnego obrazu/etykiety/metryki outer validation. Osobno zapisać, czy fine-tuning jest real-only, czy miesza dane syntetyczne.
- Trzymać architekturę i inverse-STN aktywnej bazy. Porównywać real-only z synthetic→real na identycznych foldach i seedach 42/7/123. Raportować również koszt/liczbę kroków; dodatkowy trening jest potencjalnym czynnikiem zakłócającym.
- Liczyć oficjalne mIoU na 128×128, oddzielnie dla metody i dla przypadków lokalnego/globalnego encodera. Pokazać wynik przed fine-tuningiem, po fine-tuningu oraz różnicę względem dopasowanej bazy. Wysoka bit accuracy ani niski synthetic loss nie dowodzą poprawy lokalizacji.

## Kontrola jakości przed długim treningiem

Zweryfikować deterministyczność kilku próbek, zgodność wymiarów/bitów i masek, niezmienione piksele poza obszarem wklejenia oraz zgodność downsamplingu. Obejrzeć obraz, maskę i wzmocniony residual dla obu trybów generacji i skrajnych proporcji. Sprawdzić, czy po przygotowaniu wejścia extractora nie zgubiono skali znaku. Cechy StegaStamp wymagają zachowania korekty inverse-STN także dla syntetyków; samo poprawne wyrównanie maski na pliku nie wystarcza.

Ostateczna decyzja o przyjęciu pretrainingu powinna wynikać z dopasowanego porównania real OOF, nie z literatury ani syntetycznej walidacji. Ta notatka nie podaje niezmierzonych wyników, nie deklaruje reprodukcji SOTA i nie potwierdza jeszcze, że budżet 2000 próbek wystarczy.
