# Regime-aware MoE: czy specjalizacja segmentacji ma sens?

**Status: complete.** Eksperyment real-only, bez testu konkursowego i leaderboardu.

## Wniosek: uprościć MoE, zachować eksperta clean dla StegaStamp jako kandydata

**Modelowanie reżimu ma sens, lecz pełne trójekspertowe MoE nie jest obecnie najlepszą
odpowiedzią.** Dla StegaStamp potwierdzono przydatność specjalizacji w czystych
regionach dużych masek. Nie potwierdzono skuteczności eksperta małych watermarków
ani przewagi learned gating nad zwykłym averagingiem pod względem globalnego mIoU.
PixelSeal i TrustMark nie przeszły screeningu lokalizacji; nie rozszerzano ich ekspertów.

Fazy A/B obejmują wszystkie trzy rodziny i trzy outer foldy. Faza B dodatkowo wszystkie
seedy 42/7/123. Faza C: 3 screeningowe zestawy ekspertów, potem pozostałe 8 zestawów
StegaStamp. Każdy zestaw to S, L oraz kontrola U0; łącznie **33 nowe zewnętrzne dekodery**,
przy niezmienionych checkpointach M. 207 wierszy wyników wariant–fold–seed oznacza
również kombinacje tych samych map, nie 207 niezależnie trenowanych modeli.

### 1. Czy znana wiadomość wystarcza do rozpoznania reżimu?

**Do rozróżniania małych od dużych obszarów — w dużej mierze tak.** Wybrany wewnętrznie
gate osiąga macro F1 0.6457 / 0.6808 / 0.7464 dla PixelSeal / TrustMark / StegaStamp.
Recall small wynosi 69.8% / 75.6% / 80.0%, large 96.5% / 96.5% / 97.6%.
Medium pozostaje trudne: recall około 31% / 35% / 48%. Przy granicach reżimów
accuracy spada do około 41–55%; potrzebny jest fallback M.

Sam bit accuracy już niesie większość informacji. Dodatkowe statystyki dekodera
pomagają szczególnie StegaStamp: macro F1 0.6596→0.7464, log loss 0.4093→0.3533.
Nie ma spójnego zysku z globalnie pooled cech; dla StegaStamp są gorsze od decoder-only.
Gate nie korzystał z map segmentacji. Większy MLP nie był uzasadniony.

### 2. Czy wystarczy kalibracja zależna od reżimu?

**Nie w tej niskowymiarowej wersji.** A1−A0 mIoU to −0.0013 / +0.0002 / +0.0000.
AP/AUC/Top-K pozostają dokładnie identyczne, ponieważ zmieniamy tylko próg.
Dla StegaStamp small FG IoU spada 0.0202→0.0130. Sam gate dobrze rozpoznaje
globalną sytuację, ale nie dostarcza współrzędnych watermarku.

Pusta/pełna maska na całym OOF daje mIoU odpowiednio 0.2530/0.2470 dla PixelSeal,
0.2484/0.2516 dla TrustMark i 0.2540/0.2460 dla StegaStamp. Silniejsza kontrola
`GateConstant` wybiera pełną maskę, gdy P(large)≥0.5, inaczej pustą; dla StegaStamp
osiąga0.4549 bez jakiejkolwiek lokalizacji. Kolumna `mIoU minus constant` poniżej
używa innej kontroli: pełna maska, gdy średnia probability danego modelu przekracza
jego wybrany próg, inaczej pusta. Obie decyzje korzystają wyłącznie z predykcji.

### 3. Czy specjalizacja daje przewagę nad uniwersalną głowicą?

**Tak dla clean StegaStamp, nie dla small.** Pełne OOF, średnia po trzech seedach:

| StegaStamp | A0: baza | A3: M + clean | A4: pełne soft MoE | A5: uniform |
|---|---:|---:|---:|---:|
| mIoU | 0.4651 | 0.4707 | 0.4735 | 0.4767 |
| Small AP | 0.0493 | 0.0488 | 0.0472 | 0.0461 |
| Small Top-K | 0.0405 | 0.0403 | 0.0385 | 0.0371 |
| Small FG IoU | 0.0202 | 0.0191 | 0.0184 | 0.0160 |
| Large clean IoU | 0.0796 | **0.1293** | 0.1266 | 0.1108 |
| Large internal Boundary F1 | 0.0527 | **0.0865** | 0.0846 | 0.0716 |
| Recall clean w rzeczywistych dziurach | 0.0494 | **0.1314** | 0.1262 | 0.0736 |

A3 poprawia large clean IoU w **9/9** par fold–seed. Delta +0.0497,
warunkowy bootstrap CI95 [0.0369; 0.0629]. Internal Boundary F1 poprawia się
w 8/9 par. Zysk mIoU A3 wynosi +0.0056, CI95 [0.0013; 0.0104]; po wyłączeniu
screeningowego foldu1 nadal +0.0079. Dla A4 delta mIoU +0.0084,
CI95 [0.0037; 0.0134], lecz nie wszystkie seedy zyskują: seed123 lekko traci.

### 4. Czy clean expert naprawdę lokalizuje czyste regiony?

**Tak, choć nadal słabo i kosztem części foreground.** Bezprogowe AP czystych pikseli
w dużych maskach rośnie A0→A3 0.2349→0.2587, a clean oracle Top-K 0.2206→0.2441.
Kalibracja A1 nie zmienia tych rankingów; kontrola dodatkowego treningu U0 daje
clean AP 0.2322. To dowód nowego sygnału przestrzennego, nie tylko wyższego progu.

Odsetek prawdziwie czystych pikseli błędnie oznaczonych jako watermark spada
87.2%→72.8%. Rzeczywiste wewnętrzne dziury też są częściej odzyskiwane, ale
ich recall 0.1314 pozostaje niski. Nie wszystkie czyste regiony są dziurami:
ocena clean IoU obejmuje również pasy i obszary przy ramce. Globalny foreground
IoU maleje 0.4864→0.4674, więc nie jest to bezkosztowa poprawa obu klas.
Po wyłączeniu dziur mniejszych niż16 komórek maski pozostają44 unikalne obrazy;
ich hole recall nadal rośnie 0.0529→0.1327. Zysk nie jest artefaktem kilku
pojedynczych pikseli.

### 5. Czy mały ekspert znajduje wcześniej pomijane watermarki?

**Zdarza się, ale netto przegrywa.** W 135 ocenach small obraz–seed StegaStamp
S odnajduje 6 przypadków pominiętych przez M, ale traci 17 wcześniej trafionych.
To powtórzenia 45 unikalnych obrazów, nie 135 niezależnych próbek.
„Odnalezienie” oznacza tylko TP>0 i nie gwarantuje dobrej lokalizacji ani małej liczby FP.
S-only obniża small AP 0.0493→0.0467, Top-K 0.0405→0.0378, FG IoU
0.0202→0.0130. Całkowite pominięcia rosną 64.4%→72.6%; w pełnym A4 do71.1%.
Sama czterokrotna waga małych obrazów nie nauczyła skutecznej lokalizacji.
U0 bez tej specjalizacji ma lepsze small AP 0.0534 i Top-K 0.0456.

### 6. Czy learned gating dodaje wartość?

**Względem kalibracji — tak w odzyskiwaniu clean; względem uniform — zależy od celu.**
A3/A4 przewyższają A1 w large clean IoU o0.0569/0.0541. Względem U0 również
poprawiają clean IoU o0.0312/0.0285, z dodatnimi warunkowymi CI.
Uniform ma jednak najwyższe mIoU0.4767; A4−A5 = −0.0032,
CI95 [−0.0072; 0.0009]. Brak dowodu przewagi learned gate w globalnym wyniku.
Z kolei A3 ma lepsze large clean IoU niż uniform o0.0185, CI95 [0.0102; 0.0273].

Oracle GT routing A6 daje mIoU0.4732, podobne do A4. Nawet idealna znajomość
reżimu nie naprawia słabego eksperta S. Gate nie jest tu głównym ograniczeniem.
Oracle nie jest wdrażalny i nie oznacza maksimum po wszystkich ekspertach.

### 7. Co zachować, zmienić i czego jeszcze nie wiemy?

- **Zachować dotychczasowe najlepsze rozwiązanie i bazowe checkpointy.** Nie wykonano submission.
- **Rozwijać prostsze M+L dla StegaStamp**, z naciskiem na clean ranking, clean IoU,
  granice i ochronę małych watermarków. Obecny wynik uzasadnia kolejny kontrolowany
  test, nie automatyczną zmianę rozwiązania konkursowego.
- **Nie rozwijać obecnego S ani pełnego trójekspertowego MoE jako domyślnej architektury.**
  Większy gate nie zastąpi brakującej informacji lokalnej; dla PS/TM brak awansu
  oznacza tylko negatywny screening tej konkretnej wersji.
- **Syntetyki nie są udowodnioną koniecznością.** Clean expert generalizuje lepiej
  już na realnych danych. Small expert może być ograniczony liczbą przykładów,
  słabym sygnałem cech lub niewłaściwym sposobem ważenia/selekcji. Kontrolowane
  dodatkowe dane z małymi regionami są rozsądną przyszłą hipotezą, ale w tym
  eksperymencie ich nie użyto ani nie wygenerowano.

Wyniki są rozwojowe: foldy były wcześniej wielokrotnie analizowane. CI nie obejmują
całej niepewności doboru architektury. Poprawa clean jest mocniejszym wnioskiem
niż przewaga MoE jako kompletnego rozwiązania.

### Przykłady pokazujące ograniczenia

| Obraz StegaStamp | Seed | Co pokazuje | Delta mIoU A4−A0 |
|---|---:|---|---:|
| 53934df0429a | 123 | S trafia wcześniej pominięty mały region, lecz końcowe A4 nadal go pomija; liczne FP | +0.0154 |
| 5955ddeb6f1c | 42 | L odzyskuje dużą część wewnętrznych dziur, ale A4 usuwa też prawdziwy watermark | −0.0382 |
| d2540376f815 | 7 | Gate z pewnością ≥0.8 wybiera nieprawidłowy reżim | −0.0473 |
| d02e15fbddf3 | 7 | Największe pogorszenie: GT98.5%, nadmierne przewidywanie clean | −0.2328 |
| f6c3f5f3f205 | 42 | Duża rozbieżność ekspertów i wzrost mIoU; sam wzrost nie dowodzi znalezienia małego watermarku | +0.2625 |

![Odzyskane dziury i koszt w foreground](results/stegastamp_clean_hole_recovered_example.png)

![Porażka pełnego MoE](results/stegastamp_MoE_worse_example.png)


## Architektura i protokół

Zamrożony ekstraktor i jego cztery poziomy cech; StegaStamp z aktywną korektą inverse-STN. Wspólny trunk to cztery zamrożone projekcje 1×1 i fuzja wejść przy 32×32. M pozostaje dokładną zamrożoną głowicą bazową. S i L kopiują tylko jej decoder `fuse`; każdy ma 46 241 trenowane parametry. M nie jest ponownie uczony na outer train. Dodatkowy U0 to kontrola dalszego treningu identycznego dekodera bez wag reżimowych.

S: waga obrazu 4 dla GT≤5%, pozostałe 1; target watermark. L: waga 4 dla GT>50%, pozostałe 1; target jest prawdziwą maską clean. Ostatnia warstwa L startuje ze znakiem odwróconym względem M, więc początkowo 1−P(clean)=P(M). Clean Dice podkreśla rzadkie czyste piksele dużych masek; nie dodano osobnej wagi topologicznych dziur ani pseudoetykiet. S/L/U współdzielą zamrożone projekcje; nie tworzą trzech nowych backbone.

BCE+Dice 0.5/0.5, batch8, AdamW lr0.001, weight decay0.0001, maks.50 dodatkowych epok, patience8, wybór epoki przez zwykłe mean mIoU@0.5 na zapisanych inner validation. Wagi normalizowane sumą wag w batchu. PS/TM zachowują inner80/20; SS trzy inner foldy. Specjalistów wewnętrznych inicjalizowano z M nauczonego tylko na odpowiednim inner train. Outer specjaliści startują z właściwego niezmienionego outer M i uczą się wybraną liczbę epok. Seedy42/7/123, float32 CUDA bez TF32.

PS/TM seed42 wykorzystuje oryginalne checkpointy frozen_extractor; 7/123 zapisane modele A eksperymentu RGB/SRM o tej samej architekturze/procedurze. SS używa wszystkich trzech seedów aktywnego eksperymentu STN alignment. Nie wybierano najsłabszego seedu ani checkpointów synthetic→real. GPU odtworzyło istniejące mapy z tolerancją2e−5.

A4 = gS·S + gM·M + gL·(1−Lclean). A2 łączy S z M, A3 L z M, A5 średnia równomierna, Hard wybiera argmax gate. A6 używa GT reżimu wyłącznie jako diagnostyki; nie jest modelem deployable ani matematycznym maksimum po ekspertach. Wszystkie mieszanki mają próg wybrany na inner predictions z tej samej siatki0.3–0.7.

## Faza A — jakość gate

14 cech decoder-only: bit accuracy względem prawdziwej wiadomości, signed logits (średnia/SD/kwantyle), |logits|, log-likelihood wiadomości, oczekiwana poprawność bitu, entropia. Pooled wariant dodaje 8 składowych PCA z globalnych cech ekstraktora, dopasowanych wyłącznie na train danego dopasowania. StandardScaler również train-only. GBT nie potrzebuje skalowania. Gate nie wykorzystuje map segmentacji: nie ma wycieku z predykcji in-sample. Model wybiera najniższy inner log loss; nie zewnętrzne F1. Bez MLP i bez strojenia progów reżimów.

Gate jest dopasowany raz na outer fold i współdzielony między seedami głowic. Raportowana zmienność seedów dotyczy segmentacji; deterministycznych klasyfikatorów nie traktujemy jako trzech niezależnych powtórzeń.

| Metoda | Gate | Macro F1 | Bal. acc. | S precision | S recall | L precision | L recall | Log loss | ECE10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixelseal | majority | 0.2222 | 0.3333 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0706 | 0.0352 |
| pixelseal | bit_lr | 0.5582 | 0.5676 | 0.4286 | 0.4884 | 0.9444 | 1.0000 | 0.4583 | 0.1240 |
| pixelseal | decoder_lr | 0.5784 | 0.5798 | 0.4348 | 0.4651 | 0.9647 | 0.9647 | 0.4366 | 0.1269 |
| pixelseal | decoder_hgb | 0.6220 | 0.6265 | 0.5000 | 0.5814 | 0.9647 | 0.9647 | 0.5900 | 0.1549 |
| pixelseal | pooled_lr | 0.6061 | 0.6073 | 0.4773 | 0.4884 | 0.9540 | 0.9765 | 0.4831 | 0.1043 |
| pixelseal | selected | 0.6457 | 0.6573 | 0.5357 | 0.6977 | 0.9647 | 0.9647 | 0.4461 | 0.0924 |
| trustmark | majority | 0.2222 | 0.3333 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0751 | 0.0971 |
| trustmark | bit_lr | 0.6851 | 0.6940 | 0.5965 | 0.7556 | 0.9651 | 0.9765 | 0.4026 | 0.0784 |
| trustmark | decoder_lr | 0.6968 | 0.7058 | 0.5932 | 0.7778 | 0.9880 | 0.9647 | 0.3771 | 0.0321 |
| trustmark | decoder_hgb | 0.7274 | 0.7297 | 0.6400 | 0.7111 | 0.9878 | 0.9529 | 0.4474 | 0.1014 |
| trustmark | pooled_lr | 0.7004 | 0.7021 | 0.6000 | 0.6667 | 0.9880 | 0.9647 | 0.4030 | 0.0389 |
| trustmark | selected | 0.6808 | 0.6901 | 0.5862 | 0.7556 | 0.9762 | 0.9647 | 0.4029 | 0.0488 |
| stegastamp | majority | 0.2222 | 0.3333 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | 1.0650 | 0.0618 |
| stegastamp | bit_lr | 0.6596 | 0.6674 | 0.5536 | 0.6889 | 0.9767 | 0.9882 | 0.4093 | 0.0359 |
| stegastamp | decoder_lr | 0.7464 | 0.7505 | 0.6545 | 0.8000 | 0.9765 | 0.9765 | 0.3533 | 0.0439 |
| stegastamp | decoder_hgb | 0.6932 | 0.6931 | 0.6047 | 0.5778 | 0.9765 | 0.9765 | 0.4785 | 0.0920 |
| stegastamp | pooled_lr | 0.6929 | 0.6942 | 0.5800 | 0.6444 | 0.9882 | 0.9882 | 0.4024 | 0.0581 |
| stegastamp | selected | 0.7464 | 0.7505 | 0.6545 | 0.8000 | 0.9765 | 0.9765 | 0.3533 | 0.0439 |

ECE10: top-label confidence, 10 równych binów. Near-boundary: odległość≤0.02 od5% lub≤0.10 od50%; uncertain: max gate<0.6. Te definicje ustalono przed oceną.

| Metoda | N przy granicy | Accuracy przy granicy | Pewność przy granicy | N niepewnych | Accuracy niepewnych |
| --- | --- | --- | --- | --- | --- |
| pixelseal | 17 | 0.4118 | 0.6914 | 34 | 0.5588 |
| trustmark | 27 | 0.4444 | 0.5965 | 47 | 0.5532 |
| stegastamp | 22 | 0.5455 | 0.6164 | 38 | 0.6316 |

Regresja area: Ridge alpha10, decoder-only, train-only scaler, predykcje ograniczone do[0,1].

| Metoda | Area MAE | Spearman |
| --- | --- | --- |
| pixelseal | 0.0671 | 0.8258 |
| trustmark | 0.0625 | 0.8901 |
| stegastamp | 0.0663 | 0.8647 |

Kalibracja regresji według binów area jest w phase_a.json; wszystkie prawdopodobieństwa i etykiety gate w *_gate_per_image.csv.

![Macierze pomyłek](results/figures/gate_confusion.png)

![Area regression](results/figures/coverage_regression.png)

![Rozkłady gate](results/figures/gate_distributions.png)

## Faza B — kalibracja bez nowych map

Bglobal dobiera jeden próg z pięciu. A1: sigmoid(logit(global_t)+g·offset), offset∈{−0.7,0,+0.7}³: 27 z góry określonych możliwości; zero preferowane przy remisie. Użyto wyłącznie inner held-out map M i cross-fitted gate. Outer GT nie wybiera progu. A1 zachowuje mapę probability, zmienia tylko próg per obraz: AP/AUC/Top-K muszą być identyczne z A0.

| Metoda | Wariant | mIoU | Small AP | Small Top-K | Small FG IoU | Large clean IoU |
| --- | --- | --- | --- | --- | --- | --- |
| pixelseal | A0 | 0.5106 | 0.0768 | 0.0604 | 0.0392 | 0.1296 |
| pixelseal | Bglobal | 0.5091 | 0.0768 | 0.0604 | 0.0391 | 0.1257 |
| pixelseal | A1 | 0.5093 | 0.0768 | 0.0604 | 0.0376 | 0.1294 |
| pixelseal | GateConstant | 0.4481 | 0.0168 | 0.0168 | 0.0000 | 0.0147 |
| trustmark | A0 | 0.5073 | 0.0650 | 0.0500 | 0.0241 | 0.1440 |
| trustmark | Bglobal | 0.5069 | 0.0650 | 0.0500 | 0.0240 | 0.1430 |
| trustmark | A1 | 0.5075 | 0.0650 | 0.0500 | 0.0225 | 0.1497 |
| trustmark | GateConstant | 0.4492 | 0.0239 | 0.0239 | 0.0000 | 0.0124 |
| stegastamp | A0 | 0.4651 | 0.0493 | 0.0405 | 0.0202 | 0.0796 |
| stegastamp | Bglobal | 0.4671 | 0.0493 | 0.0405 | 0.0173 | 0.0918 |
| stegastamp | A1 | 0.4652 | 0.0493 | 0.0405 | 0.0130 | 0.0725 |
| stegastamp | GateConstant | 0.4549 | 0.0190 | 0.0190 | 0.0000 | 0.0060 |

## Screening ekspertów i awans

Fold1/seed42. Awans wymaga poprawy obu small AP i Top-K (A2) lub obu large clean IoU i internal Boundary F1 (A3); A4 może spełnić którykolwiek warunek. Dodatkowo mIoU nie może spaść o więcej niż0.01. Zysk samego mIoU nie wystarcza.

| Metoda | Wariant | Awans | Small | Large |
| --- | --- | --- | --- | --- |
| pixelseal | A2 | False | False | False |
| pixelseal | A3 | False | False | False |
| pixelseal | A4 | False | False | False |
| trustmark | A2 | False | False | False |
| trustmark | A3 | False | False | False |
| trustmark | A4 | False | False | False |
| stegastamp | A2 | False | False | True |
| stegastamp | A3 | True | False | True |
| stegastamp | A4 | True | False | True |
| Metoda | Wariant | mIoU | Small AP | Small Top-K | Small FG IoU | Large clean IoU |
| --- | --- | --- | --- | --- | --- | --- |
| pixelseal | A0 | 0.5191 | 0.0260 | 0.0127 | 0.0054 | 0.1367 |
| pixelseal | A1 | 0.5177 | 0.0260 | 0.0127 | 0.0048 | 0.1269 |
| pixelseal | A2 | 0.5197 | 0.0268 | 0.0127 | 0.0050 | 0.1367 |
| pixelseal | A3 | 0.5178 | 0.0259 | 0.0127 | 0.0060 | 0.1393 |
| pixelseal | A4 | 0.5187 | 0.0268 | 0.0127 | 0.0055 | 0.1393 |
| pixelseal | A5 | 0.5187 | 0.0266 | 0.0127 | 0.0051 | 0.1429 |
| pixelseal | A6 | 0.5178 | 0.0268 | 0.0127 | 0.0052 | 0.1392 |
| pixelseal | Hard | 0.5183 | 0.0270 | 0.0127 | 0.0060 | 0.1392 |
| pixelseal | U0 | 0.5188 | 0.0263 | 0.0127 | 0.0045 | 0.1545 |
| stegastamp | A0 | 0.4700 | 0.0682 | 0.0620 | 0.0353 | 0.0330 |
| stegastamp | A1 | 0.4717 | 0.0682 | 0.0620 | 0.0089 | 0.0337 |
| stegastamp | A2 | 0.4722 | 0.0741 | 0.0615 | 0.0035 | 0.0334 |
| stegastamp | A3 | 0.4772 | 0.0682 | 0.0620 | 0.0353 | 0.0794 |
| stegastamp | A4 | 0.4770 | 0.0741 | 0.0615 | 0.0329 | 0.0586 |
| stegastamp | A5 | 0.4750 | 0.0713 | 0.0639 | 0.0235 | 0.0553 |
| stegastamp | A6 | 0.4808 | 0.0675 | 0.0577 | 0.0265 | 0.0589 |
| stegastamp | Hard | 0.4743 | 0.0673 | 0.0577 | 0.0272 | 0.0581 |
| stegastamp | U0 | 0.4591 | 0.0602 | 0.0570 | 0.0024 | 0.0977 |
| trustmark | A0 | 0.5129 | 0.0514 | 0.0414 | 0.0104 | 0.1940 |
| trustmark | A1 | 0.5102 | 0.0514 | 0.0414 | 0.0070 | 0.2023 |
| trustmark | A2 | 0.5135 | 0.0510 | 0.0417 | 0.0114 | 0.1941 |
| trustmark | A3 | 0.5141 | 0.0514 | 0.0414 | 0.0130 | 0.2126 |
| trustmark | A4 | 0.5147 | 0.0510 | 0.0416 | 0.0134 | 0.2126 |
| trustmark | A5 | 0.5157 | 0.0515 | 0.0419 | 0.0153 | 0.1960 |
| trustmark | A6 | 0.5145 | 0.0506 | 0.0429 | 0.0135 | 0.2136 |
| trustmark | Hard | 0.5143 | 0.0524 | 0.0430 | 0.0178 | 0.2004 |
| trustmark | U0 | 0.4618 | 0.0543 | 0.0474 | 0.0240 | 0.0120 |

W rodzinach z awansem trenowano te same dwa eksperckie dekodery dla pozostałych foldów/seedów. Pozostałe mieszanki tych samych map zachowano jako kontrole/diagnostykę, nie jako dodatkowo wybrane zwycięskie konfiguracje. Rodzin bez awansu nie rozszerzano.

## Pełne wyniki i kontrole

| Metoda | Wariant | Protokół | mIoU | FG IoU | Clean IoU | AUC macro | AP macro | mIoU minus constant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixelseal | A0 | 3-fold x 3 seeds | 0.5106 | 0.5170 | 0.5042 | 0.7137 | 0.6305 | 0.0636 |
| pixelseal | Bglobal | 3-fold x 3 seeds | 0.5091 | 0.5173 | 0.5010 | 0.7137 | 0.6305 | 0.0623 |
| pixelseal | A1 | 3-fold x 3 seeds | 0.5093 | 0.5175 | 0.5010 | 0.7137 | 0.6305 | 0.0641 |
| pixelseal | A2 | screen fold1 seed42 | 0.5197 | 0.5022 | 0.5373 | 0.6860 | 0.6032 | 0.0702 |
| pixelseal | A3 | screen fold1 seed42 | 0.5178 | 0.4942 | 0.5414 | 0.6868 | 0.6015 | 0.0632 |
| pixelseal | A4 | screen fold1 seed42 | 0.5187 | 0.4958 | 0.5415 | 0.6809 | 0.6026 | 0.0640 |
| pixelseal | A5 | screen fold1 seed42 | 0.5187 | 0.4938 | 0.5436 | 0.6841 | 0.6024 | 0.0641 |
| pixelseal | A6 | screen fold1 seed42 | 0.5178 | 0.4943 | 0.5414 | 0.6740 | 0.6017 | 0.0632 |
| pixelseal | Hard | screen fold1 seed42 | 0.5183 | 0.4957 | 0.5409 | 0.6809 | 0.6019 | 0.0636 |
| pixelseal | S_only | screen fold1 seed42 | 0.5177 | 0.4716 | 0.5637 | 0.6746 | 0.6019 | 0.0726 |
| pixelseal | L_only | screen fold1 seed42 | 0.5161 | 0.4926 | 0.5395 | 0.6805 | 0.5997 | 0.0614 |
| pixelseal | U0 | screen fold1 seed42 | 0.5188 | 0.4885 | 0.5492 | 0.6776 | 0.6019 | 0.0662 |
| pixelseal | GateConstant | 3-fold x 3 seeds | 0.4481 | 0.4451 | 0.4512 | 0.5000 | 0.4939 | 0.0000 |
| trustmark | A0 | 3-fold x 3 seeds | 0.5073 | 0.4978 | 0.5168 | 0.6906 | 0.5979 | 0.0590 |
| trustmark | Bglobal | 3-fold x 3 seeds | 0.5069 | 0.4974 | 0.5163 | 0.6906 | 0.5979 | 0.0586 |
| trustmark | A1 | 3-fold x 3 seeds | 0.5075 | 0.4943 | 0.5206 | 0.6906 | 0.5979 | 0.0614 |
| trustmark | A2 | screen fold1 seed42 | 0.5135 | 0.4372 | 0.5897 | 0.6495 | 0.5288 | 0.0598 |
| trustmark | A3 | screen fold1 seed42 | 0.5141 | 0.4327 | 0.5955 | 0.6481 | 0.5288 | 0.0604 |
| trustmark | A4 | screen fold1 seed42 | 0.5147 | 0.4335 | 0.5959 | 0.6484 | 0.5292 | 0.0610 |
| trustmark | A5 | screen fold1 seed42 | 0.5157 | 0.4473 | 0.5840 | 0.6525 | 0.5308 | 0.0643 |
| trustmark | A6 | screen fold1 seed42 | 0.5145 | 0.4339 | 0.5952 | 0.6515 | 0.5293 | 0.0608 |
| trustmark | Hard | screen fold1 seed42 | 0.5143 | 0.4469 | 0.5818 | 0.6479 | 0.5297 | 0.0578 |
| trustmark | S_only | screen fold1 seed42 | 0.5063 | 0.4542 | 0.5585 | 0.6529 | 0.5308 | 0.0498 |
| trustmark | L_only | screen fold1 seed42 | 0.5153 | 0.4549 | 0.5757 | 0.6512 | 0.5322 | 0.0588 |
| trustmark | U0 | screen fold1 seed42 | 0.4618 | 0.4564 | 0.4672 | 0.6542 | 0.5325 | 0.0053 |
| trustmark | GateConstant | 3-fold x 3 seeds | 0.4492 | 0.4450 | 0.4535 | 0.5000 | 0.5033 | 0.0000 |
| stegastamp | A0 | 3-fold x 3 seeds | 0.4651 | 0.4864 | 0.4438 | 0.6699 | 0.5749 | 0.0292 |
| stegastamp | Bglobal | 3-fold x 3 seeds | 0.4671 | 0.4786 | 0.4557 | 0.6699 | 0.5749 | 0.0281 |
| stegastamp | A1 | 3-fold x 3 seeds | 0.4652 | 0.4731 | 0.4572 | 0.6699 | 0.5749 | 0.0199 |
| stegastamp | A2 | 3-fold x 3 seeds | 0.4690 | 0.4801 | 0.4579 | 0.6723 | 0.5730 | 0.0254 |
| stegastamp | A3 | 3-fold x 3 seeds | 0.4707 | 0.4674 | 0.4741 | 0.6811 | 0.5762 | 0.0368 |
| stegastamp | A4 | 3-fold x 3 seeds | 0.4735 | 0.4675 | 0.4795 | 0.6835 | 0.5743 | 0.0362 |
| stegastamp | A5 | 3-fold x 3 seeds | 0.4767 | 0.4728 | 0.4805 | 0.6741 | 0.5733 | 0.0292 |
| stegastamp | A6 | 3-fold x 3 seeds | 0.4732 | 0.4670 | 0.4794 | 0.6857 | 0.5754 | 0.0335 |
| stegastamp | Hard | 3-fold x 3 seeds | 0.4716 | 0.4651 | 0.4781 | 0.6832 | 0.5729 | 0.0337 |
| stegastamp | S_only | 3-fold x 3 seeds | 0.4656 | 0.4702 | 0.4610 | 0.6650 | 0.5668 | 0.0280 |
| stegastamp | L_only | 3-fold x 3 seeds | 0.4752 | 0.4673 | 0.4831 | 0.6653 | 0.5662 | 0.0331 |
| stegastamp | U0 | 3-fold x 3 seeds | 0.4699 | 0.4697 | 0.4701 | 0.6788 | 0.5746 | 0.0289 |
| stegastamp | GateConstant | 3-fold x 3 seeds | 0.4549 | 0.4509 | 0.4589 | 0.5000 | 0.4921 | 0.0000 |

Screeningu57 obrazów nie należy porównywać z pełnym OOF170 obrazów. Pełny wynik jest średnią metryk po obrazach OOF i trzech seedach, a nie ensemble probabilities. Seedy powtarzają te same obrazy; to nadal170 unikalnych obserwacji/metodę.

## Lokalizacja małych watermarków

| Metoda | Wariant | Protokół | AP | Top-K | FG IoU | Puste maski | Całkowicie pominięte | FP / obraz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pixelseal | A0 | 3-fold x 3 seeds | 0.0768 | 0.0604 | 0.0392 | 0.1395 | 0.7984 | 0.0356 |
| pixelseal | Bglobal | 3-fold x 3 seeds | 0.0768 | 0.0604 | 0.0391 | 0.1318 | 0.7984 | 0.0384 |
| pixelseal | A1 | 3-fold x 3 seeds | 0.0768 | 0.0604 | 0.0376 | 0.1085 | 0.7597 | 0.0439 |
| pixelseal | A2 | screen fold1 seed42 | 0.0268 | 0.0127 | 0.0050 | 0.0000 | 0.9375 | 0.0280 |
| pixelseal | A3 | screen fold1 seed42 | 0.0259 | 0.0127 | 0.0060 | 0.1875 | 0.9375 | 0.0205 |
| pixelseal | A4 | screen fold1 seed42 | 0.0268 | 0.0127 | 0.0055 | 0.0000 | 0.9375 | 0.0220 |
| pixelseal | A5 | screen fold1 seed42 | 0.0266 | 0.0127 | 0.0051 | 0.0000 | 0.9375 | 0.0227 |
| pixelseal | A6 | screen fold1 seed42 | 0.0268 | 0.0127 | 0.0052 | 0.0000 | 0.9375 | 0.0235 |
| pixelseal | Hard | screen fold1 seed42 | 0.0270 | 0.0127 | 0.0060 | 0.0625 | 0.9375 | 0.0230 |
| pixelseal | S_only | screen fold1 seed42 | 0.0268 | 0.0127 | 0.0078 | 0.1875 | 0.9375 | 0.0127 |
| pixelseal | L_only | screen fold1 seed42 | 0.0265 | 0.0127 | 0.0039 | 0.1250 | 0.9375 | 0.0257 |
| pixelseal | U0 | screen fold1 seed42 | 0.0263 | 0.0127 | 0.0045 | 0.0000 | 0.8750 | 0.0312 |
| pixelseal | GateConstant | 3-fold x 3 seeds | 0.0168 | 0.0168 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| trustmark | A0 | 3-fold x 3 seeds | 0.0650 | 0.0500 | 0.0241 | 0.2741 | 0.7926 | 0.0288 |
| trustmark | Bglobal | 3-fold x 3 seeds | 0.0650 | 0.0500 | 0.0240 | 0.2815 | 0.7926 | 0.0287 |
| trustmark | A1 | 3-fold x 3 seeds | 0.0650 | 0.0500 | 0.0225 | 0.2889 | 0.8000 | 0.0260 |
| trustmark | A2 | screen fold1 seed42 | 0.0510 | 0.0417 | 0.0114 | 0.4286 | 0.8095 | 0.0155 |
| trustmark | A3 | screen fold1 seed42 | 0.0514 | 0.0414 | 0.0130 | 0.3810 | 0.8095 | 0.0199 |
| trustmark | A4 | screen fold1 seed42 | 0.0510 | 0.0416 | 0.0134 | 0.3333 | 0.8095 | 0.0195 |
| trustmark | A5 | screen fold1 seed42 | 0.0515 | 0.0419 | 0.0153 | 0.2857 | 0.8095 | 0.0261 |
| trustmark | A6 | screen fold1 seed42 | 0.0506 | 0.0429 | 0.0135 | 0.2857 | 0.8095 | 0.0208 |
| trustmark | Hard | screen fold1 seed42 | 0.0524 | 0.0430 | 0.0178 | 0.2381 | 0.8095 | 0.0307 |
| trustmark | S_only | screen fold1 seed42 | 0.0506 | 0.0429 | 0.0175 | 0.2381 | 0.8095 | 0.0345 |
| trustmark | L_only | screen fold1 seed42 | 0.0517 | 0.0429 | 0.0199 | 0.2381 | 0.7619 | 0.0507 |
| trustmark | U0 | screen fold1 seed42 | 0.0543 | 0.0474 | 0.0240 | 0.0476 | 0.6667 | 0.0796 |
| trustmark | GateConstant | 3-fold x 3 seeds | 0.0239 | 0.0239 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| stegastamp | A0 | 3-fold x 3 seeds | 0.0493 | 0.0405 | 0.0202 | 0.2370 | 0.6444 | 0.1276 |
| stegastamp | Bglobal | 3-fold x 3 seeds | 0.0493 | 0.0405 | 0.0173 | 0.2667 | 0.6963 | 0.1131 |
| stegastamp | A1 | 3-fold x 3 seeds | 0.0493 | 0.0405 | 0.0130 | 0.3481 | 0.7630 | 0.0891 |
| stegastamp | A2 | 3-fold x 3 seeds | 0.0472 | 0.0384 | 0.0159 | 0.2889 | 0.7259 | 0.0958 |
| stegastamp | A3 | 3-fold x 3 seeds | 0.0488 | 0.0403 | 0.0191 | 0.2667 | 0.6889 | 0.1155 |
| stegastamp | A4 | 3-fold x 3 seeds | 0.0472 | 0.0385 | 0.0184 | 0.2667 | 0.7111 | 0.0976 |
| stegastamp | A5 | 3-fold x 3 seeds | 0.0461 | 0.0371 | 0.0160 | 0.2963 | 0.7407 | 0.0793 |
| stegastamp | A6 | 3-fold x 3 seeds | 0.0467 | 0.0378 | 0.0139 | 0.2667 | 0.7481 | 0.0861 |
| stegastamp | Hard | 3-fold x 3 seeds | 0.0505 | 0.0405 | 0.0172 | 0.2519 | 0.7259 | 0.0951 |
| stegastamp | S_only | 3-fold x 3 seeds | 0.0467 | 0.0378 | 0.0130 | 0.2815 | 0.7259 | 0.0943 |
| stegastamp | L_only | 3-fold x 3 seeds | 0.0437 | 0.0356 | 0.0173 | 0.3259 | 0.7333 | 0.0878 |
| stegastamp | U0 | 3-fold x 3 seeds | 0.0534 | 0.0456 | 0.0207 | 0.2667 | 0.7185 | 0.0832 |
| stegastamp | GateConstant | 3-fold x 3 seeds | 0.0190 | 0.0190 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |

Top-K zna prawdziwą liczbę pikseli GT: wyłącznie diagnostyka. Remisy są rozstrzygane przez oczekiwaną precyzję losowego wyboru wśród remisów. Całkowicie pominięty watermark oznacza TP=0; FP area oznacza liczbę fałszywie dodatnich pikseli podzieloną przez całą powierzchnię obrazu.

## Czyste regiony w dużych maskach

| Metoda | Wariant | Protokół | Clean IoU | FP / true clean | Internal BF1@2 | Recall clean w dziurach |
| --- | --- | --- | --- | --- | --- | --- |
| pixelseal | A0 | 3-fold x 3 seeds | 0.1296 | 0.7904 | 0.0780 | 0.1089 |
| pixelseal | Bglobal | 3-fold x 3 seeds | 0.1257 | 0.7984 | 0.0734 | 0.1079 |
| pixelseal | A1 | 3-fold x 3 seeds | 0.1294 | 0.7925 | 0.0760 | 0.1086 |
| pixelseal | A2 | screen fold1 seed42 | 0.1367 | 0.7803 | 0.0795 | 0.1149 |
| pixelseal | A3 | screen fold1 seed42 | 0.1393 | 0.7679 | 0.0791 | 0.1204 |
| pixelseal | A4 | screen fold1 seed42 | 0.1393 | 0.7679 | 0.0792 | 0.1204 |
| pixelseal | A5 | screen fold1 seed42 | 0.1429 | 0.7539 | 0.0900 | 0.1338 |
| pixelseal | A6 | screen fold1 seed42 | 0.1392 | 0.7678 | 0.0799 | 0.1198 |
| pixelseal | Hard | screen fold1 seed42 | 0.1392 | 0.7680 | 0.0797 | 0.1198 |
| pixelseal | S_only | screen fold1 seed42 | 0.1667 | 0.6357 | 0.1184 | 0.2214 |
| pixelseal | L_only | screen fold1 seed42 | 0.1392 | 0.7678 | 0.0799 | 0.1198 |
| pixelseal | U0 | screen fold1 seed42 | 0.1545 | 0.6956 | 0.1074 | 0.1819 |
| pixelseal | GateConstant | 3-fold x 3 seeds | 0.0147 | 0.9647 | 0.0000 | 0.0000 |
| trustmark | A0 | 3-fold x 3 seeds | 0.1440 | 0.7756 | 0.0850 | 0.1427 |
| trustmark | Bglobal | 3-fold x 3 seeds | 0.1430 | 0.7762 | 0.0857 | 0.1436 |
| trustmark | A1 | 3-fold x 3 seeds | 0.1497 | 0.7589 | 0.0879 | 0.1550 |
| trustmark | A2 | screen fold1 seed42 | 0.1941 | 0.7109 | 0.1330 | 0.1564 |
| trustmark | A3 | screen fold1 seed42 | 0.2126 | 0.6338 | 0.1302 | 0.2626 |
| trustmark | A4 | screen fold1 seed42 | 0.2126 | 0.6337 | 0.1302 | 0.2627 |
| trustmark | A5 | screen fold1 seed42 | 0.1960 | 0.7241 | 0.1315 | 0.1774 |
| trustmark | A6 | screen fold1 seed42 | 0.2136 | 0.6320 | 0.1301 | 0.2647 |
| trustmark | Hard | screen fold1 seed42 | 0.2004 | 0.7164 | 0.1287 | 0.1956 |
| trustmark | S_only | screen fold1 seed42 | 0.1552 | 0.8108 | 0.1005 | 0.1273 |
| trustmark | L_only | screen fold1 seed42 | 0.2012 | 0.7171 | 0.1285 | 0.1946 |
| trustmark | U0 | screen fold1 seed42 | 0.0120 | 0.9856 | 0.0062 | 0.0100 |
| trustmark | GateConstant | 3-fold x 3 seeds | 0.0124 | 0.9647 | 0.0000 | 0.0508 |
| stegastamp | A0 | 3-fold x 3 seeds | 0.0796 | 0.8725 | 0.0527 | 0.0494 |
| stegastamp | Bglobal | 3-fold x 3 seeds | 0.0918 | 0.8405 | 0.0599 | 0.0650 |
| stegastamp | A1 | 3-fold x 3 seeds | 0.0725 | 0.8672 | 0.0482 | 0.0516 |
| stegastamp | A2 | 3-fold x 3 seeds | 0.0835 | 0.8589 | 0.0540 | 0.0546 |
| stegastamp | A3 | 3-fold x 3 seeds | 0.1293 | 0.7283 | 0.0865 | 0.1314 |
| stegastamp | A4 | 3-fold x 3 seeds | 0.1266 | 0.7390 | 0.0846 | 0.1262 |
| stegastamp | A5 | 3-fold x 3 seeds | 0.1108 | 0.8123 | 0.0716 | 0.0736 |
| stegastamp | A6 | 3-fold x 3 seeds | 0.1281 | 0.7308 | 0.0864 | 0.1287 |
| stegastamp | Hard | 3-fold x 3 seeds | 0.1270 | 0.7342 | 0.0844 | 0.1287 |
| stegastamp | S_only | 3-fold x 3 seeds | 0.0845 | 0.8436 | 0.0604 | 0.0652 |
| stegastamp | L_only | 3-fold x 3 seeds | 0.1213 | 0.7681 | 0.0818 | 0.1118 |
| stegastamp | U0 | 3-fold x 3 seeds | 0.0981 | 0.7942 | 0.0639 | 0.0845 |
| stegastamp | GateConstant | 3-fold x 3 seeds | 0.0060 | 0.9765 | 0.0000 | 0.0000 |

Internal BF1 usuwa sztuczną granicę ramki obrazu; tolerancja2 komórki128×128. Dziura do diagnostyki to komponent clean całkowicie otoczony GT watermark, wykryty binary_fill_holes. To węższy zbiór niż wszystkie czyste regiony: pasy przy ramce nie są dziurami. Ta definicja nie tworzy żadnych etykiet do treningu. Hole recall nie wystarcza samodzielnie, ponieważ maska całkowicie clean uzyskałaby1. FP/true clean jest jednocześnie odsetkiem czystych pikseli błędnie oznaczonych jako watermark.

Dodatkowa diagnostyka StegaStamp: ranking pikseli clean przy GT watermark>50%, bez progu. Pozwala odróżnić lepsze maski binarne od lepszego uporządkowania czystych pikseli.

| Wariant | Large clean AP | Large clean oracle Top-K |
| --- | --- | --- |
| A0 | 0.2349 | 0.2206 |
| A1 | 0.2349 | 0.2206 |
| A2 | 0.2349 | 0.2206 |
| A3 | 0.2587 | 0.2441 |
| A4 | 0.2587 | 0.2440 |
| A5 | 0.2481 | 0.2327 |
| A6 | 0.2589 | 0.2440 |
| Bglobal | 0.2349 | 0.2206 |
| GateConstant | 0.0877 | 0.0877 |
| Hard | 0.2593 | 0.2444 |
| L_only | 0.2589 | 0.2440 |
| S_only | 0.2242 | 0.2118 |
| U0 | 0.2322 | 0.2176 |

Kontrola wrażliwości: po wyłączeniu komponentów dziur mniejszych niż16 komórek maski, pozostają 44 unikalne obrazy StegaStamp. Ich średni hole clean recall A0→A3 wynosi 0.0529→0.1327. To diagnostyka po treningu, nie filtr etykiet uczących ani kryterium awansu. W galerii przykłady odzyskanych dziur również wymagają komponentu≥16 komórek, aby nie prezentować przypadkowych pojedynczych pikseli.

## Zmienność i parowane różnice

| Metoda | Wariant | mIoU SD seedów | mIoU SD foldów | Small AP SD seedów | Large clean SD seedów |
| --- | --- | --- | --- | --- | --- |
| pixelseal | A0 | 0.0028 | 0.0135 | 0.0024 | 0.0192 |
| pixelseal | A1 | 0.0025 | 0.0102 | 0.0024 | 0.0140 |
| trustmark | A0 | 0.0103 | 0.0126 | 0.0065 | 0.0127 |
| trustmark | A1 | 0.0095 | 0.0116 | 0.0065 | 0.0068 |
| stegastamp | A0 | 0.0140 | 0.0104 | 0.0157 | 0.0399 |
| stegastamp | A1 | 0.0054 | 0.0085 | 0.0157 | 0.0433 |
| stegastamp | A3 | 0.0069 | 0.0086 | 0.0156 | 0.0185 |
| stegastamp | A4 | 0.0043 | 0.0064 | 0.0112 | 0.0185 |
| stegastamp | A5 | 0.0051 | 0.0087 | 0.0117 | 0.0158 |
| stegastamp | U0 | 0.0042 | 0.0154 | 0.0153 | 0.0174 |
| Wariant | Kontrola | Zakres | Metryka | Delta | CI95 low | CI95 high |
| --- | --- | --- | --- | --- | --- | --- |
| A3 | A0 | all | miou | 0.0056 | 0.0012 | 0.0104 |
| A3 | A0 | all | ap | 0.0012 | 0.0003 | 0.0022 |
| A3 | A0 | all | clean_iou | 0.0303 | 0.0236 | 0.0379 |
| A3 | A0 | folds2_3 | miou | 0.0079 | 0.0026 | 0.0132 |
| A3 | A0 | folds2_3 | ap | 0.0010 | -0.0002 | 0.0021 |
| A3 | A0 | folds2_3 | clean_iou | 0.0347 | 0.0257 | 0.0445 |
| A3 | A0 | small | ap | -0.0005 | -0.0013 | -0.0000 |
| A3 | A0 | small | topk | -0.0001 | -0.0003 | -0.0000 |
| A3 | A0 | large | clean_iou | 0.0497 | 0.0369 | 0.0629 |
| A3 | A0 | large | boundary_f1 | 0.0338 | 0.0237 | 0.0440 |
| A3 | A0 | large | hole_clean_recall | 0.0819 | 0.0533 | 0.1129 |
| A3 | A1 | all | miou | 0.0056 | 0.0004 | 0.0112 |
| A3 | A5 | all | miou | -0.0059 | -0.0104 | -0.0013 |
| A3 | U0 | all | miou | 0.0008 | -0.0058 | 0.0075 |
| A4 | A0 | all | miou | 0.0084 | 0.0037 | 0.0134 |
| A4 | A0 | all | ap | -0.0006 | -0.0034 | 0.0019 |
| A4 | A0 | all | clean_iou | 0.0357 | 0.0285 | 0.0433 |
| A4 | A0 | folds2_3 | miou | 0.0118 | 0.0056 | 0.0178 |
| A4 | A0 | folds2_3 | ap | -0.0012 | -0.0043 | 0.0017 |
| A4 | A0 | folds2_3 | clean_iou | 0.0440 | 0.0349 | 0.0539 |
| A4 | A0 | small | ap | -0.0021 | -0.0076 | 0.0027 |
| A4 | A0 | small | topk | -0.0020 | -0.0074 | 0.0029 |
| A4 | A0 | large | clean_iou | 0.0470 | 0.0347 | 0.0599 |
| A4 | A0 | large | boundary_f1 | 0.0318 | 0.0219 | 0.0418 |
| A4 | A0 | large | hole_clean_recall | 0.0768 | 0.0491 | 0.1062 |
| A4 | A1 | all | miou | 0.0083 | 0.0033 | 0.0137 |
| A4 | A5 | all | miou | -0.0032 | -0.0072 | 0.0009 |
| A4 | U0 | all | miou | 0.0036 | -0.0033 | 0.0104 |

Bootstrap2000 parowanych losowań obrazów, po uśrednieniu różnic seedów dla każdego obrazu. Przedziały są warunkowe, nie uwzględniają ponownego treningu ani wielokrotnego rozwoju na tych foldach. Foldy2/3 pokazano osobno od screeningowego foldu1. Wyniki warunkowane predicted regime i niepewnością gate, oraz medium precision/recall/area error: stratified_metrics.csv i comparison.csv. Wszystkie seedy i foldy: osobne CSV.

![mIoU według reżimu](results/figures/miou_by_regime.png)

![Lokalizacja według reżimu](results/figures/localization_by_regime.png)

## Przykłady, artefakty i reprodukcja

[Galeria wszystkich ocenionych obrazów i seedów](results/gallery/index.html), [dobór sukcesów i porażek](results/visual_selections.json), [porównania](results/comparison.csv), [per-image](results/per_image_metrics.csv), [foldy](results/fold_metrics.csv), [seedy](results/seed_metrics.csv), [parowane delty](results/paired_deltas.csv), [gate](results/gate_comparison.csv), [weryfikacja](results/verification.json).

Przykłady wybierane jawnie: small znaleziony przez S po miss M; największy wzrost hole recall w komponentach≥16 komórek; błędny gate z pewnością≥0.8; największa strata mIoU; największa średnia SD map ekspertów. Brak przykładu danej kategorii jest zapisany jako brak, nie zastąpiony wygodniejszym przypadkiem. Wszystkie mapy ekspertów, final probabilities, maski, progi, checkpointy i historie w results/.

Instrukcja uruchomienia: [README](README.md). Fazy A/B zostały zapisane przed treningiem ekspertów; SCREENING.md przed potwierdzeniem. Zachowano niezmienione wcześniejsze checkpointy i źródłowe splity. Nie wygenerowano danych syntetycznych, nie odczytywano testu konkursowego i nie wysłano zgłoszenia.

## Ograniczenia

Mały, wielokrotnie analizowany zbiór; brak nowego nietkniętego holdoutu. Gate ma mało danych medium, a oddzielenie medium/small jest trudne. S/L używają jednego ustalonego współczynnika4, bez szerokiego searchu. Zamrożone projekcje ograniczają zdolność specjalizacji. L zmienia orientację Dice na clean oraz wagę obrazów, więc nie izolujemy obu efektów. U0 kontroluje dalszy trening, ale nie wszystkie możliwe różnice capacity/ensemblingu. GT oracle A6 nie jest rozwiązaniem wdrożeniowym.
