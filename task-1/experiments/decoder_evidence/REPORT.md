# Wytrenowane dekodery i zgodność ze znaną wiadomością

Celem jest wykorzystanie informacji z wiadomości do lokalizacji, bez uczenia dekodowania od zera. Wszystkie cechy powstają bez masek GT.
PixelSeal: ostatnia mapa ConvNeXt → oryginalny pixel_decoder.output_upscaling → oryginalna warstwa linear zastosowana w każdej pozycji. TrustMark: ostatnia mapa ResNet → oryginalna fc w każdej pozycji. Średnia przestrzenna logitów jest sprawdzana względem prawdziwego globalnego odczytu dla wszystkich 170 obrazów. To projekcje globalnie trenowanego dekodera, nie gwarancja lokalności.
Mapy PS/TM: średnia sigmoid logitu zgodnego ze znanym bitem, trafność znaku, średnia zgodnego logitu podzielona przez średni moduł logitów oraz globalna zgodność. Brak maskowania GT podczas ekstrakcji.
StegaStamp: zasłonięcie kolejno 16 obszarów 128×128 w oryginalnym obrazie 512×512 średnim kolorem tego obrazu. Dekoder pozostaje w całości zamrożony. Mapy opisują spadek zgodności sigmoid i log-prawdopodobieństwa, globalną zgodność oraz trafność bitów po zasłonięciu. To wrażliwość na interwencję, nie prawdopodobieństwo lokalnej obecności; zasłanianie może wprowadzać artefakty.
map_only: mała głowica uczona tylko na czterech mapach. residual: istniejąca głowica + uczona korekta z map, początkowo zerowa. residual_wrong: identyczna architektura, ale wiadomości cyklicznie zamienione między obrazami; kontrola dodatkowej pojemności i informacji niezwiązanej ze znajomością poprawnej wiadomości.
Kontrfaktyczna ocena oznacza podmianę poprawnej wiadomości na błędną tylko na wejściu wytrenowanego modelu; dla residual_wrong odwrotnie. Nie zmieniamy przy tym progu.
blend: średnia ważona prawdopodobieństw bazowej głowicy i map_only. Waga z [0,.25,.5,.75,1] i próg z [.3,.4,.5,.6,.7] dobierane wyłącznie na odtworzonych modelach wewnętrznych. Następnie te same ustawienia łączą modele wytrenowane na całym outer-train. Wariant dodany po obejrzeniu screeningu map_only; to dodatkowy element selekcji rozwojowej.
Drugi etap po słabym wyniku zbiorczych map: bit_map, bit_residual i bit_residual_wrong zachowują każdą z 256/100 map bitów osobno. PS/TM: sigmoid logitu pomnożonego przez znak oczekiwanego bitu. StegaStamp: tanh spadku log-prawdopodobieństwa danego bitu /5 po zasłonięciu. Architektury i kontrole analogiczne do pierwszego etapu; wybór po screeningu również obciąża ocenę selekcją.
Screening: pierwszy zapisany fold, 57 obrazów na metodę. Seed 42, batch 8, AdamW .001, 50 epok / patience 8, loss .5 BCE + .5 Dice. Epoka i próg wybrane wyłącznie na wewnętrznym podziale; następnie trening od nowa na całym outer-train.
Próg CV ustalony przed wynikami: dla danej metody poprawa >=.005 nad baseline oraz >=.005 nad kontrfaktyczną błędną wiadomością. Korekta residual musi dodatkowo przewyższyć model trenowany z błędną wiadomością. CV tylko najlepszego wariantu spełniającego warunki dla danej metody. Baseline to identyczny wcześniejszy protokół frozen_extractor/cv3, nie późniejszy wariant stabilnej selekcji StegaStamp.
To porównanie rozwojowe; CV zawiera fold screeningowy i jest obciążone wyborem wariantu. Do selekcji wariantów i pomiaru CV nie użyto testu konkursowego ani leaderboardu.

Stan: complete

| Metoda | Wariant | Baseline | mIoU | Zmiana | Kontrfaktyczna wiadomość | Epoka | Próg |
|---|---|---:|---:|---:|---:|---:|---:|
| pixelseal | map_only | 0.519131 | 0.483269 | -0.035862 | 0.264290 | 38 | 0.7 |
| pixelseal | residual | 0.519131 | 0.521217 | +0.002086 | 0.520549 | 13 | 0.7 |
| pixelseal | residual_wrong | 0.519131 | 0.521776 | +0.002645 | 0.521778 | 13 | 0.6 |
| pixelseal | blend | 0.519131 | 0.519615 | +0.000484 | 0.521309 | 38 | 0.4 |
| pixelseal | bit_map | 0.519131 | 0.267336 | -0.251795 | 0.260799 | 1 | 0.6 |
| pixelseal | bit_residual | 0.519131 | 0.509690 | -0.009441 | 0.509876 | 11 | 0.3 |
| pixelseal | bit_residual_wrong | 0.519131 | 0.516239 | -0.002892 | 0.516239 | 9 | 0.5 |
| trustmark | map_only | 0.512931 | 0.463196 | -0.049736 | 0.275533 | 19 | 0.5 |
| trustmark | residual | 0.512931 | 0.505343 | -0.007588 | 0.507487 | 17 | 0.5 |
| trustmark | residual_wrong | 0.512931 | 0.509060 | -0.003871 | 0.509082 | 4 | 0.6 |
| trustmark | blend | 0.512931 | 0.510794 | -0.002137 | 0.507884 | 19 | 0.7 |
| trustmark | bit_map | 0.512931 | 0.441576 | -0.071355 | 0.284960 | 1 | 0.7 |
| trustmark | bit_residual | 0.512931 | 0.505083 | -0.007849 | 0.505184 | 12 | 0.6 |
| trustmark | bit_residual_wrong | 0.512931 | 0.494516 | -0.018416 | 0.494530 | 18 | 0.5 |
| stegastamp | map_only | 0.465554 | 0.222051 | -0.243503 | 0.222051 | 1 | 0.6 |
| stegastamp | residual | 0.465554 | 0.474857 | +0.009303 | 0.470703 | 10 | 0.6 |
| stegastamp | residual_wrong | 0.465554 | 0.394398 | -0.071156 | 0.392673 | 10 | 0.6 |
| stegastamp | blend | 0.465554 | 0.463569 | -0.001985 | 0.463460 | 1 | 0.4 |
| stegastamp | bit_map | 0.465554 | 0.410537 | -0.055017 | 0.187799 | 20 | 0.6 |
| stegastamp | bit_residual | 0.465554 | 0.480421 | +0.014867 | 0.261460 | 21 | 0.6 |
| stegastamp | bit_residual_wrong | 0.465554 | 0.451906 | -0.013649 | 0.444162 | 24 | 0.6 |

## pixelseal

Globalna trafność bitów zamrożonego dekodera: 0.7644. Rodzaj map: pretrained spatial projection. Maksymalny błąd odtworzenia projekcji: 1.621246337890625e-05.
Wariant do CV: brak.

## trustmark

Globalna trafność bitów zamrożonego dekodera: 0.7539. Rodzaj map: pretrained spatial projection. Maksymalny błąd odtworzenia projekcji: 1.33514404296875e-05.
Wariant do CV: brak.

## stegastamp

Globalna trafność bitów zamrożonego dekodera: 0.7464. Rodzaj map: 4x4 occlusion sensitivity. Maksymalny błąd odtworzenia projekcji: None.
Wariant do CV: bit_residual.
Foldy: 0.480421, 0.416890, 0.494317. Średnia 0.463876; SD 0.041280. OOF 0.463697, zmiana względem baseline +0.053182.

## Porównanie z nowszą stabilną selekcją StegaStamp

Ta sama procedura trzech stratyfikowanych foldów wewnętrznych co stegastamp_stable_selection, seed 42. Zmieniono wyłącznie model przez dodanie korekty z pretrained informacji bitowej. Screening na foldzie 1: delta >=.005 i przewaga nad błędną wiadomością >=.005, następnie pozostałe foldy. Wariant wybrany po poprzednich analizach; to nadal porównanie rozwojowe.
| Fold | Aktualny baseline | Wiadomość + stabilna selekcja | Zmiana | Błędna wiadomość |
|---|---:|---:|---:|---:|
| 1 | 0.468685 | 0.474146 | +0.005461 | 0.282221 |
| 2 | 0.444979 | 0.470020 | +0.025041 | 0.303974 |
| 3 | 0.454954 | 0.486496 | +0.031543 | 0.317213 |
Stan: complete; próg screeningu: True.
Średnia foldów: 0.456206 → 0.476887, zmiana +0.020681; SD nowego wariantu 0.008574.

OOF (obrazy ważone równo): 0.476831. Poprawa w 3/3 foldach.
Wniosek: użytecznym wariantem dla StegaStamp jest korekta z per-bit wrażliwością wytrenowanego dekodera na zasłanianie, połączona ze stabilną selekcją. PixelSeal i TrustMark nie uzyskały potwierdzonej poprawy w screeningu, więc zachowują bazową głowicę.
To wyniki jednego seedu 42 na wcześniej używanych foldach. Podmiana wiadomości pokazuje jej znaczenie dla predykcji, ale nie izoluje wkładu każdego kanału ani nie dowodzi dokładnej lokalności dekodera. Wyniki CV należy odróżniać od osobnej oceny leaderboardu poniżej.

## Trening końcowy i predykcje testowe

Model bit_residual StegaStamp wytrenowany na wszystkich 170 obrazach treningowych, seed 42, 21 epok, próg 0.6. Epoki i próg to mediany z trzech foldów stabilnego CV; ustalone przed predykcjami testowymi.
Checkpoint: `results/full_train/stegastamp_head.pt`; SHA256 `ffda68187cac4842b72418e3d3c95ec395172c58bc618c67a6e2c40e724a8242`.
Wygenerowano submission dla 1020 obrazów: nowy model dla 340 StegaStamp, 680 predykcji PixelSeal/TrustMark identycznych z submission #1202. Sprawdzono nazwy, rozmiary, binarność, zgodność po ponownym załadowaniu checkpointu i niezmienność pozostałych metod.
Plik `results/full_train/submission.pt`, 16734635 bajtów, SHA256 `01d51cbbbe39a79edb4ad221df75550facb18e861f8b343f41e35b1786c12713`. Udział pikseli dodatnich StegaStamp: 0.5127. Globalna trafność bitów na teście: 0.7530; nie jest to metryka lokalizacji.

## Publiczny leaderboard

Zaakceptowane submission #1299: publiczne mIoU **0.510900**, poprzednie #1202 **0.512300**, zmiana **-0.001400**.
To wynik całego zgłoszenia na ukrytej publicznej części testowej, nie mIoU samego StegaStamp ani kolejny wynik CV.
Wniosek po wysyłce: poprawa rozwojowego CV nie przełożyła się na publiczny test. Ten wariant nie zastępuje najlepszego zgłoszenia #1202. Sam wynik nie rozstrzyga, czy przyczyną jest dobór wariantu na znanych foldach, zmienność treningu czy różnica rozkładu danych.

Odtworzenie: `prepare.py` generuje zamrożone cechy; `run.py` wykonuje screening i warunkowe CV; `stable.py` sprawdza nowy wariant selekcji; `train_full_predict.py` trenuje końcową głowicę i zapisuje kompletne zgłoszenie testowe. Raport i dashboard aktualizują się po kolejnych wynikach. Checkpointy odtwarzają zapisane predykcje z tolerancją 1e-7.
