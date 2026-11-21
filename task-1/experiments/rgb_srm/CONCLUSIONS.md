## Odpowiedzi i rekomendacja

**Zachować dotychczasowe najlepsze rozwiązanie. D warto dalej badać tylko dla StegaStamp;
nie ma podstaw do zastąpienia nim wszystkich trzech modeli.** Ukończono 60 przebiegów:
54 dla A/D (3 metody × 3 foldy × 3 seedy) i 6 screeningowych B/C.

1. **Czy RGB-only przewyższa cechy ekstraktora? Nie w tym screeningu.** Na tym samym
   foldzie macro AP A→B wynosi PixelSeal 0.6066→0.5037, TrustMark 0.5364→0.4854,
   StegaStamp 0.6352→0.5873. RGB poprawia small AP StegaStamp 0.0318→0.0443,
   ale obniża jego small Top-K 0.0392→0.0307. Żaden B/C nie spełnił kryterium
   awansu. To dowód przewagi tych ekstraktorów nad tą konkretną lekką CNN;
   nie rozdziela wpływu pretrainingu od większej pojemności zamrożonych modeli.

2. **Czy SRM pomaga? Brak spójnego potwierdzenia.** W screeningu B→C small AP
   rośnie dla PixelSeal 0.0252→0.0354 i TrustMark 0.0330→0.0365, lecz spada dla
   StegaStamp 0.0443→0.0222. Macro AP C jest niższe od B we wszystkich metodach.
   C nie potwierdzano na innych foldach/seedach, więc nie przesądzamy o SRM ogólnie.

3. **Czy fuzja pomaga? Lokalnie dla StegaStamp, bez poprawy globalnego mIoU.**
   Po 3 foldach × 3 seedach D podnosi jego small AP 0.0425→0.0564 (+0.0139),
   Top-K 0.0312→0.0452 (+0.0140) i FG IoU 0.0153→0.0205. Small AP i Top-K
   poprawiają się w średniej OOF każdego z trzech seedów, ale tylko w dwóch
   z trzech foldów: fold3 AP 0.0314→0.0295, Top-K 0.0220→0.0123.
   Warunkowy CI95 delty small AP wynosi [0.0022; 0.0284]; dla Top-K
   [−0.0004; 0.0311], dla FG IoU [−0.0012; 0.0137]. To obiecujący, ograniczony
   sygnał. W PixelSeal small AP spada 0.0775→0.0716, w TrustMark 0.0716→0.0672;
   również small Top-K obu metod spada. TrustMark zyskuje globalne AUC
   0.6987→0.7153, ale traci small FG IoU i clean IoU dużych masek 0.1298→0.1071.

4. **Czy to rzeczywista lokalizacja? Zysk StegaStamp dotyczy rankingu, nie tylko
   dominującej klasy.** AP/Top-K nie zależą od wybranego progu; oba rosną,
   podczas gdy OOF mIoU spada 0.4682→0.4664. D przewyższa losowy ranking
   małych masek: AP 0.0564 vs 0.0196, Top-K 0.0452 vs 0.0190. Nadal są to
   słabe wyniki bezwzględne: small FG IoU zaledwie 0.0205, small AUC spada
   0.5203→0.5110, a small mIoU 0.4394 jest niższe niż 0.4905 pustej maski.
   Poprawa dotyczy najlepszych punktów rankingu; problem lokalizacji nie został rozwiązany.

5. **Następny krok: kontrolowana ablacja D dla StegaStamp.** Zachować obecną
   architekturę A dla PixelSeal/TrustMark oraz najlepsze wcześniejsze rozwiązanie.
   Jako następny eksperyment porównać D z identyczną fuzją, ale wyzerowanymi
   dziewięcioma kanałami SRM (ta sama liczba parametrów, inicjalizacja i budżet),
   na wszystkich tych samych foldach/seedach; główne kryteria to small AP i Top-K.
   Pozwoli to ustalić, czy sygnał D pochodzi z residuals, czy z gałęzi RGB.
   Nie zmieniać jednocześnie loss, selekcji epoki i dekodera. Szczególnie
   sprawdzić fold3, gdzie zysk się nie przenosi. Wdrażanie D teraz byłoby przedwczesne.

Wszystkie wnioski dotyczą rozwojowych foldów wielokrotnie wykorzystywanych w projekcie.
Pełne wyniki poniżej rozdzielają screening od OOF; nie wykorzystano leaderboardu.
