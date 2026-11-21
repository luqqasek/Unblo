# Częstość problemów obecnego najlepszego modelu

Zakres: zachowany najlepszy wariant #1202 — PixelSeal/TrustMark original seed42 oraz StegaStamp stable seed42. Nowszy eksperyment #1299 nie zastąpił go według raportu decoder_evidence. Oceniamy zapisane predykcje 3-fold OOF dla 510 unikalnych obrazów (170 na metodę). Każdy obraz oceniano modelem, który nie trenował na tym obrazie. To rozwojowy zbiór walidacyjny, nie wynik końcowego checkpointu na ukrytym teście.

Obejrzano pojedynczo 30 paneli RGB / GT / probability / prediction / błędy: 27 wybranych deterministycznie (najgorszy, środkowy i najlepszy mIoU w każdej metodzie i grupie powierzchni <=5%, >5–50%, >50%) oraz po jednej środkowej według mIoU pustej predykcji na metodę. Dobór celowy, nie losowy; częstości policzono na wszystkich 510 obrazach, nie na tych 30.

Kategorie są nakładającymi się objawami, a nie rozłącznymi przyczynami. Progi są jawnymi kryteriami tego audytu, nie oficjalnymi kategoriami konkursu. Kolejność oznacza częstość przy tych kryteriach, nie wagę straty. Nie należy sumować procentów.

| Problem | Obrazy / 510 | Odsetek | PixelSeal / 170 | TrustMark / 170 | StegaStamp / 170 |
|---|---:|---:|---:|---:|---:|
| Błędna lub brakująca granica maski | 363 | 71.2% | 113 | 121 | 129 |
| Niewykrywanie czystych fragmentów przy dużym watermarku | 227 | 44.5% | 70 | 74 | 83 |
| Pominięcie większości watermarku | 208 | 40.8% | 65 | 76 | 67 |
| Większość zaznaczenia trafia w czyste miejsca | 152 | 29.8% | 55 | 35 | 62 |
| Ani jeden piksel watermarku nie został trafiony | 129 | 25.3% | 36 | 58 | 35 |
| Duży błąd powierzchni maski | 128 | 25.1% | 46 | 31 | 51 |
| Bardzo słaba lokalizacja małych watermarków | 124 | 24.3% | 39 | 43 | 42 |
| Całkowicie pusta predykcja | 53 | 10.4% | 5 | 36 | 12 |

## Definicje i interpretacja

- **Błędna lub brakująca granica maski:** Internal boundary F1 < 0,10; tolerancja 2 komórki (8 px obrazu), bez ramki obrazu.
- **Niewykrywanie czystych fragmentów przy dużym watermarku:** GT > 50% obrazu i wykryte mniej niż 50% czystych pikseli.
- **Pominięcie większości watermarku:** Recall watermarku < 50%.
- **Większość zaznaczenia trafia w czyste miejsca:** Niepusta predykcja i precision watermarku < 50%.
- **Ani jeden piksel watermarku nie został trafiony:** TP = 0, przy niepustej masce GT.
- **Duży błąd powierzchni maski:** Bezwzględna różnica powierzchni predykcji i GT > 10 punktów procentowych obrazu.
- **Bardzo słaba lokalizacja małych watermarków:** GT <= 5% obrazu i IoU klasy watermark < 0,10.
- **Całkowicie pusta predykcja:** Zero pikseli dodatnich przy niepustej masce GT.

Wśród 255 obrazów z watermarkiem zajmującym ponad połowę obrazu 227 (89,0%) traci większość czystych pikseli. Dotyczy to zarówno otworów, jak i czystych obszarów przy krawędzi — nie wszystkie są zamkniętymi dziurami. Wśród 133 małych watermarków 124 (93,2%) ma IoU watermarku poniżej 0,10. Z 129 przypadków bez trafienia 53 to puste maski, a 76 to zaznaczenie wyłącznie błędnych miejsc.

Błąd granic obejmuje również brak maski i zaznaczenie całego obrazu; nie oznacza 363 przypadków samego rozmycia poprawnej lokalizacji. Podobna powierzchnia predykcji i GT nie gwarantuje poprawnego położenia (np. 6d05e72e601c i e300d14d9ad9). W oglądanych panelach zdarzają się poprawne lokalizacje z niedokładną krawędzią; nie wszystkie obrazy są porażką.

## Przegląd obrazów jeden po drugim

| Obraz | Obserwacja wizualna |
|---|---|
| [694715f71b85](results/examples/pixelseal_original_42_089.jpg) | Baseball: maska niemal pełna; rozległe czyste fragmenty z GT zostały zalane predykcją. |
| [d854e128465a](results/examples/pixelseal_original_42_002.jpg) | Łódź: model znajduje czystą plamę poniżej właściwej i pomija pozostałe czyste fragmenty. |
| [2360e3e4a53f](results/examples/pixelseal_original_42_132.jpg) | Umywalka: duży czysty obszar zlokalizowany dość dobrze, ale krawędzie są przesunięte i pojawiają się dodatkowe wycięcia u góry. |
| [8659c72e0980](results/examples/pixelseal_original_42_021.jpg) | Niebieski autobus: wąski pionowy watermark niemal pominięty; liczne fałszywe plamy zajmują 21,1% obrazu zamiast 1,2%. |
| [6d05e72e601c](results/examples/pixelseal_original_42_007.jpg) | Kobieta i dzieci: mały prostokąt po lewej pominięty całkowicie; predykcja w innych miejscach mimo podobnej powierzchni. |
| [c73e30f4cffc](results/examples/pixelseal_original_42_139.jpg) | Kominy: mały watermark w prawym górnym rogu znaleziony, ale zaznaczenie zbyt szerokie. |
| [2e5699234a38](results/examples/pixelseal_original_42_063.jpg) | Czerwony autobus: GT obejmuje autobus, predykcja niemal cały obraz; wyraźna nadsegmentacja. |
| [7aefd4b81812](results/examples/pixelseal_original_42_163.jpg) | Dziecko z zabawką: poziomy pas watermarku zastąpiony kilkoma drobnymi plamami; większość pasa ginie. |
| [7f233578c2dc](results/examples/pixelseal_original_42_140.jpg) | Słonie: dolny pas trafiony dobrze, ale lokalne granice uproszczone i miejscami przesunięte. |
| [08a436824488](results/examples/trustmark_original_42_038.jpg) | Żyrafy: duże błędy obu klas; rozległe tło z watermarkiem pominięte, część czystych koron drzew zaznaczona. |
| [0b08481293d1](results/examples/trustmark_original_42_107.jpg) | Dzieci z telefonem: poprawna dominująca klasa, lecz dodatkowe błędne czyste wycięcia przy krawędziach. |
| [8e744aa74502](results/examples/trustmark_original_42_157.jpg) | Talerz z jedzeniem: główny czysty otwór znaleziony; pozostał lokalny błąd jego kształtu i mały pominięty otwór. |
| [9ccd12ff89e9](results/examples/trustmark_original_42_055.jpg) | Autobus z psami: mały prostokąt u góry całkiem pominięty; znacznie większa plama znajduje się niżej i w lewo. |
| [e300d14d9ad9](results/examples/trustmark_original_42_028.jpg) | Łazienka: watermark przy suficie pominięty, predykcja na ścianie; podobna powierzchnia, zerowe pokrycie. |
| [be73d9dde1fe](results/examples/trustmark_original_42_152.jpg) | Koszykarze: mały prostokąt trafiony, lecz maska około 4,4 razy za duża; nawet najlepszy mały przypadek TrustMark ma duży nadmiar. |
| [6d015d1ee1a6](results/examples/trustmark_original_42_025.jpg) | Dziecko z misiami: duży prostokąt watermarku prawie cały pominięty; mała plama przy jego górnym prawym narożniku. |
| [da014a7b8652](results/examples/trustmark_original_42_024.jpg) | Ptak na łące: poziomy nieregularny pas pominięty w większości; model zaznacza pionową plamę poniżej. |
| [a558bea42351](results/examples/trustmark_original_42_020.jpg) | Telewizor: główny region zlokalizowany poprawnie, ale zaznaczenie wychodzi poza jego granice. |
| [d02e15fbddf3](results/examples/stegastamp_stable_42_068.jpg) | Sygnalizator nocą: prawie cały obraz ma watermark, model zaznacza tylko 28,8%; rozległe pominięcie. |
| [bb4b0481f38e](results/examples/stegastamp_stable_42_162.jpg) | Big Ben: pełna predykcja całkowicie zalewa wyraźny czysty prostokąt. |
| [42ee90cc410d](results/examples/stegastamp_stable_42_145.jpg) | Zegar i osoba: czysty fragment w lewym dolnym rogu częściowo znaleziony, lecz rozbity i niedoszacowany; dodatkowe błędne wycięcie u góry. |
| [53934df0429a](results/examples/stegastamp_stable_42_005.jpg) | Misie: watermark to 0,6% obrazu przy prawej krawędzi, model zaznacza 99,7% obrazu. |
| [34db63f9db87](results/examples/stegastamp_stable_42_006.jpg) | Samolot od przodu: pionowy prostokąt GT prawie pominięty; główna plama przesunięta w dół. |
| [de11c61674be](results/examples/stegastamp_stable_42_159.jpg) | Tort: część małego watermarku w lewym dolnym rogu trafiona, ale wiele dodatkowych plam poza nim. |
| [5b7127ca0a3a](results/examples/stegastamp_stable_42_013.jpg) | Samolot w chmurach: mały prostokąt 5,8% obrazu zamieniony w niemal pełną maskę. |
| [4571a5b76263](results/examples/stegastamp_stable_42_169.jpg) | Kieliszki: szeroki poziomy prostokąt zastąpiony rozproszonymi wyspami; dużo pominięć i fałszywych trafień. |
| [bd93c8feb303](results/examples/stegastamp_stable_42_131.jpg) | Żółty autobus: główny obszar znaleziony, ale maska jest za wąska po bokach; mała fałszywa wyspa po lewej. |
| [73478591d121](results/examples/pixelseal_original_42_015.jpg) | Znak STOP: pusta predykcja PixelSeal mimo pionowego watermarku przy lewej krawędzi. |
| [9adf90a51a71](results/examples/trustmark_original_42_064.jpg) | Góry: pusta predykcja TrustMark mimo regionu watermarku przy lewej dolnej krawędzi. |
| [6498956eca5d](results/examples/stegastamp_stable_42_168.jpg) | Ciężarówka: pusta predykcja StegaStamp mimo poziomego pasa watermarku. |

## Odtworzenie i weryfikacja

Uruchom z katalogu repo: `task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/frequency_audit.py`, następnie `task-1/.venv/Scripts/python.exe task-1/dashboard/build.py`.

Skrypt weryfikuje TP/FP/FN/TN, mIoU, powierzchnie i granice bezpośrednio względem zapisanych probabilities i GT. Nie zmienia modelu ani progów. [Pełne flagi per obraz](results/frequency_per_image.csv), [liczebności i zapis przeglądu](results/frequency_audit.json).

Przyczyny takie jak wpływ STN, globalny kontekst i kalibracja wymagają osobnych eksperymentów; nie przypisano im częstości na podstawie samego wyglądu masek.
