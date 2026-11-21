# Znana wiadomość: pomiary nowych propozycji

**Nie potwierdzono zapowiadanych dużych zysków.** Poniższe wyniki rozdzielają sygnał wiadomości od poprawy gotowego segmentatora.

[Plan i reprodukcja](README.md) · [Audyt źródeł i matematyki](RESEARCH.md)

## Zakres i protokół

Wykonane liczby obrazów: `{'pixelseal': 170, 'trustmark': 170, 'reembed_pixelseal': 170, 'reembed_trustmark': 170, 'reembed_stegastamp': 170}`. PS/TM: natywne ekstraktory i dwie skale, IG od rozmytego obrazu, Keep/Drop oraz pilot uczonej fuzji. Wszystkie trzy metody: re-embedding NCC. Żaden nowy model nie został zgłoszony.

Podstawą Keep/Drop są zapisane OOF ensemble trzech seedów: PixelSeal 4k/2500 i TrustMark 4k/5000. Każdy próg dobierany z wiadomości i logitów, bez użycia GT. Oryginalny próg jest jednym z kandydatów. Oracle używa GT i jest wyłącznie diagnostyką.

## Mapy: macro ROC-AUC / Average Precision

Macro oznacza średnią metryki liczonej osobno na każdym obrazie. AP zależy od powierzchni maski; przy braku sygnału porównujemy z null i raportujemy małe regiony osobno.

| Metoda | Mapa | AUC | AP | AUC minus null [CI95] |
|---|---|---:|---:|---|
| pixelseal | dense256_cos | 0.6335 | 0.5809 | +0.1549 [+0.1052; +0.2039] |
| pixelseal | dense512_cos | 0.6322 | 0.5830 | +0.1278 [+0.0803; +0.1739] |
| pixelseal | blur_ig | 0.5022 | 0.4992 | +0.0031 [+0.0010; +0.0053] |
| pixelseal | blur_ig_calibrated | 0.5024 | 0.4997 | — |
| pixelseal | multiscale_cos | 0.6577 | 0.5993 | +0.1618 [+0.1123; +0.2093] |
| pixelseal | baseline_probability | 0.7869 | 0.6641 | — |
| trustmark | dense245_cos | 0.6013 | 0.5625 | +0.1024 [+0.0562; +0.1480] |
| trustmark | dense512_cos | 0.5133 | 0.5204 | -0.0023 [-0.0423; +0.0376] |
| trustmark | blur_ig | 0.5007 | 0.5093 | +0.0013 [-0.0024; +0.0048] |
| trustmark | blur_ig_calibrated | 0.5017 | 0.5098 | — |
| trustmark | multiscale_cos | 0.5974 | 0.5604 | +0.0889 [+0.0418; +0.1377] |
| trustmark | baseline_probability | 0.7656 | 0.6504 | — |
| pixelseal | ncc16 | 0.4988 | 0.4973 | -0.0024 [-0.0192; +0.0132] |
| pixelseal | ncc32 | 0.4980 | 0.4971 | -0.0000 [-0.0224; +0.0209] |
| pixelseal | ncc32_calibrated | 0.5016 | 0.4972 | — |
| trustmark | ncc16 | 0.5189 | 0.5137 | +0.0254 [+0.0130; +0.0389] |
| trustmark | ncc32 | 0.5346 | 0.5155 | +0.0508 [+0.0302; +0.0725] |
| trustmark | ncc32_calibrated | 0.5378 | 0.5151 | — |
| stegastamp | ncc16 | 0.5539 | 0.5061 | +0.0547 [+0.0384; +0.0718] |
| stegastamp | ncc32 | 0.5750 | 0.5107 | +0.0798 [+0.0564; +0.1033] |
| stegastamp | ncc32_calibrated | 0.5654 | 0.5082 | — |

Wszystkie mapy, podziały small/large i sposób osadzania: [map_metrics.csv](results/map_metrics.csv). Dodatnia różnica względem losowej wiadomości dowodzi sygnału zależnego od wiadomości, nie zysku po fuzji.

## Keep/Drop: rzeczywista zmiana mIoU

| Metoda | Kryterium | mIoU | Delta do bazy | CI95 delty |
|---|---|---:|---:|---|
| pixelseal | baseline | 0.55899 | +0.00000 | [+0.00000; +0.00000] |
| pixelseal | contrast_true | 0.54395 | -0.01505 | [-0.02466; -0.00578] |
| pixelseal | retain95_true | 0.56157 | +0.00257 | [-0.00298; +0.00858] |
| pixelseal | contrast_null1 | 0.54149 | -0.01751 | [-0.02706; -0.00856] |
| pixelseal | retain95_null1 | 0.55607 | -0.00293 | [-0.00624; +0.00004] |
| pixelseal | contrast_null2 | 0.54755 | -0.01144 | [-0.02126; -0.00174] |
| pixelseal | retain95_null2 | 0.55867 | -0.00032 | [-0.00483; +0.00424] |
| pixelseal | oracle | 0.59690 | +0.03791 | [+0.03047; +0.04606] |
| trustmark | baseline | 0.54745 | +0.00000 | [+0.00000; +0.00000] |
| trustmark | contrast_true | 0.53983 | -0.00762 | [-0.01880; +0.00350] |
| trustmark | retain95_true | 0.54518 | -0.00227 | [-0.00989; +0.00492] |
| trustmark | contrast_null1 | 0.52951 | -0.01795 | [-0.02989; -0.00695] |
| trustmark | retain95_null1 | 0.54416 | -0.00330 | [-0.01006; +0.00281] |
| trustmark | contrast_null2 | 0.53700 | -0.01045 | [-0.02257; +0.00191] |
| trustmark | retain95_null2 | 0.54786 | +0.00041 | [-0.00782; +0.00838] |
| trustmark | oracle | 0.59365 | +0.04620 | [+0.03727; +0.05617] |

`contrast`: maksimum signed score keep minus drop. `retain95`: najmniejsza maska zachowująca >=95% przyrostu score względem rozmycia i zostawiająca <=5% poza maską; brak spełnienia warunku zachowuje próg bazowy. Null1/2 to osobne fałszywe wiadomości. Nie wybrano najlepszego progu na GT.

### Małe znaki (GT <=5%)

| Metoda | Kryterium | mIoU | Delta |
|---|---|---:|---:|
| pixelseal | baseline | 0.50536 | +0.00000 |
| pixelseal | contrast_true | 0.48387 | -0.02150 |
| pixelseal | retain95_true | 0.50367 | -0.00169 |
| trustmark | baseline | 0.49873 | +0.00000 |
| trustmark | contrast_true | 0.48407 | -0.01466 |
| trustmark | retain95_true | 0.50121 | +0.00248 |

## Czy doklejenie map pomaga segmentatorowi?

Po sprawdzeniu sygnału uruchomiono małą korektę zamrożonego segmentatora seed42. Dwa ramiona mają identyczną architekturę, inicjalizację, kolejność danych i procedurę wyboru; różnią się poprawnością wiadomości. Każde ramię wybiera epokę i próg na inner-validation, potem jest trenowane od nowa na outer-train. Wyjściowa korekta zerowa odtwarza bazę. Dane outer-validation nie wybierają epoki ani progu.

| Metoda | Fold | Baza | Fuzja true | Fuzja wrong | True model z wrong w inference | Bramka CV |
|---|---:|---:|---:|---:|---:|---|
| pixelseal | 1 | 0.56768 | 0.56787 | 0.56793 | 0.56756 | NIE |
| trustmark | 1 | 0.52381 | 0.52575 | 0.52381 | 0.52646 | NIE |

Bramka: co najmniej +0,005 nad bazą i nad modelem uczonym z wrong na foldzie 1. Wyniki tego pilota nie stanowią pełnego CV ani pomiaru na niezależnym teście. Niespełnienie bramki zatrzymuje dalsze foldy. Odtworzenie wewnętrznego modelu bazowego sprawdzono względem zapisanej historii oryginalnego treningu.

## Kontrola dodatnia re-embedding

Po negatywnym teście realnych danych sprawdzono osiem deterministycznie wybranych coverów COCO na metodę. Oryginalne osadzenie na pełnym obrazie, centralna czysta dziura 256x256, ta sama wiadomość, rozdzielczość i faza. Osobno czysty cover i fałszywa wiadomość. To mała kontrola diagnostyczna po wynikach, nie nowy wynik konkursowy.

| Metoda | N | AUC NCC32 true | AUC wrong | Trafność bitów na pełnym watermarku |
|---|---:|---:|---:|---:|
| pixelseal | 8 | 0.5631 | 0.4880 | 1.0000 |
| trustmark | 8 | 0.6545 | 0.5035 | 1.0000 |
| stegastamp | 8 | 0.6880 | 0.5233 | 1.0000 |

## Kontrole implementacji i ograniczenia

- Gęste logity odtwarzają średni odczyt oryginalnego ekstraktora; sprawdzone na każdym obrazie. Rozdzielczość 512 celowo omija resize wrappera, więc jest zmianą skali ekstraktora.
- `blur_ig` to dokładnie wzór z dostarczonej propozycji: **straight-line integrated gradients od rozmytej bazy**, nie oryginalny algorytm BlurIG ze ścieżką zmiennego sigma. RGB sumowane po kanałach; nie wykonano osobnej ścieżki wyłącznie luminancji.
- Cztery prawe końce przedziałów nie zapewniają dokładnej kompletności. Test 16 kroków obejmuje osiem z góry ustalonych obrazów na rodzinę; pozostający błąd jest jawnie raportowany.
- Odejmowanie średniej dwóch losowych wiadomości nazwano w plikach `calibrated` dla zgodności z propozycją. To szumna kontrola kontrfaktyczna, nie udowodnione usunięcie biasu tekstury. Dla liniowego signed score oczekiwanie po losowej wiadomości wynosi już zero.
- Re-embedding wykorzystuje pełny obraz i oryginalne wrappery uint8. Brak zgodności fazy, cropa albo odpowiedzi zależnej od treści może pogarszać wynik. Negatywny test nie wyklucza metody z poprawnie odtworzoną geometrią.
- Powtarzane dane rozwojowe, wiele map bez korekty wielokrotnych porównań. CI obejmują bootstrap sparowanych obrazów, nie ponowny trening ani nowe zdjęcia.

```json
{
  "pixelseal": {
    "projection_max_error": 7.62939453125e-06,
    "mean_bit_accuracy": 0.7644071691176471,
    "ig4_completeness_mae": 0.6277858078164761,
    "ig16_subset_n": 8,
    "ig4_subset_completeness_mae": 0.7022056417772546,
    "ig16_subset_completeness_mae": 0.47170220233965665,
    "ig4_ig16_mean_correlation": 0.9688562374315645
  },
  "trustmark": {
    "projection_max_error": 7.62939453125e-06,
    "mean_bit_accuracy": 0.7539411653490627,
    "ig4_completeness_mae": 0.07670108264257365,
    "ig16_subset_n": 8,
    "ig4_subset_completeness_mae": 0.10079586997744627,
    "ig16_subset_completeness_mae": 0.05850758301676251,
    "ig4_ig16_mean_correlation": 0.9817928121760646
  }
}
```

## Dalsze decyzje

Mapy przechodzące ustaloną bramkę sygnału (AUC >=.60, dolny CI przewagi nad null >0, small AP nie gorsze od null):

- pixelseal: `dense256_cos`
- pixelseal: `dense256_acc`
- trustmark: `dense245_cos`

**Żaden wariant nie uzyskał w tych testach podstaw do wdrożenia.** Piloty fuzji nie przeszły bramki, a +0,00257 mIoU Keep/Drop dla PixelSeal ma CI95 obejmujący zero i lekko pogarsza małe znaki. Obietnica natychmiastowego +0,03–0,08 nie znajduje potwierdzenia.

**Najbardziej konkretny dalszy kierunek: re-embedding z dopasowaniem geometrii dla StegaStamp.** NCC32 ma AUC 0,60714 na 85 realnych obrazach z globalnym osadzeniem i 0,54295 na 85 z lokalnym osadzeniem; przy pełnej zgodności geometrii na ośmiu syntetycznych kontrolach AUC wynosi 0,68798. To poszlaka, nie dowód, że sama korekta geometrii rozwiąże różnicę domen. Kolejny test powinien izolować skalę, położenie i zależność residualu od treści na sparowanych coverach, a dopiero potem sprawdzać fuzję.

Bramka sygnału uzasadniła opisany wyżej test małej fuzji. Sama zgodność wiadomości nie wystarcza do zastąpienia obecnego modelu. Nie należy wdrażać surowego maksimum keep-minus-drop, ani traktować wartości oracle jako wyniku osiągalnego bez etykiet.

Nie testowano tu pełnego odmrażania ekstraktora z masked bit loss, różniczkowalnej optymalizacji maski, segmentacji spójnych składowych ani konwersji ONNX StegaStamp do autograd. Wyniki nie obalają tych odrębnych hipotez.
