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
