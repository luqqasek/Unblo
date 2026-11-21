# Lokalny odczyt wiadomości jako zadanie pomocnicze

Wariant inspirowany WAM: wspólna głowica 32×32, osobne wyjścia segmentacji i bitów. Frozen ekstraktory i podstawowy loss 0.5 BCE + 0.5 Dice pozostają bez zmian. Wiadomość jest wyłącznie celem treningowym; nie jest wejściem modelu.
Pomocniczy BCE bitów ważony udziałem watermarku w komórce 32×32 (average pooling maski 128×128). Normalizacja osobno dla obrazu przez liczbę bitów i masę maski; następnie średnia obrazów. Czyste komórki nie dostają gradientu wiadomości; graniczne dostają wagę częściowego pokrycia. To adaptacja równania WAM, nie identyczna implementacja batchowej normalizacji jego kodu.
Seed 42, AdamW 0.001, batch 8, maksymalnie 50 epok, patience 8. Wybór epoki i progu wyłącznie według segmentacji na wewnętrznym podziale. Następnie nowy trening na całym outer-train.
Pierwszy fold: 57 obrazów na metodę. Do CV kwalifikuje poprawa łączna >=0.005 i poprawa w co najmniej dwóch metodach. Tylko najlepsza zakwalifikowana waga przechodzi do pełnego CV. To screening rozwojowy na znanym podziale, nie niezależny test.
Waga 0 odtwarza istniejący baseline: jednakowe wagi początkowe segmentacji, kolejność batchy i wyniki; dodanie głowicy nie zmienia punktu odniesienia.
[Research i uzasadnienie wyboru](RESEARCH.md).

Stan: complete

| Wariant | PixelSeal | TrustMark | StegaStamp | Łącznie | Zmiana | Próg CV |
|---|---:|---:|---:|---:|---:|---|
| control | 0.519131 | 0.512931 | 0.465554 | 0.499205 | +0.000000 | kontrola |
| message01 | 0.523549 | 0.505013 | 0.389433 | 0.472665 | -0.026540 | niespełniony |
| message1 | 0.521556 | 0.514039 | 0.401777 | 0.479124 | -0.020082 | niespełniony |

## Czy model odczytuje wiadomość?

Trafność bitów: średnia ważona pokryciem watermarku, następnie średnia po obrazach. Kontrola shuffled porównuje te same predykcje z wiadomością kolejnego obrazu walidacyjnego; majority to stały bit większości wyznaczony wyłącznie z outer-train. Wiadomości kontrolne są użyte tylko diagnostycznie. Duże pole recepcyjne może rozgłaszać globalny odczyt na wiele komórek, więc sama trafność nie dowodzi lokalizacji.

| Wariant | Metoda | Bity na watermarku | Shuffled | Majority train | Bity na czystym | BCE bitów | Epoka | Próg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| control | pixelseal | 0.4915 | 0.4956 | 0.4989 | 0.4942 | 0.8956 | 15 | 0.3 |
| control | trustmark | 0.4999 | 0.4947 | 0.5165 | 0.4984 | 0.9108 | 12 | 0.7 |
| control | stegastamp | 0.4942 | 0.4910 | 0.4919 | 0.4942 | 0.7692 | 22 | 0.4 |
| message01 | pixelseal | 0.4991 | 0.4922 | 0.4989 | 0.4941 | 0.7074 | 15 | 0.5 |
| message01 | trustmark | 0.5022 | 0.5005 | 0.5165 | 0.5068 | 0.7238 | 4 | 0.5 |
| message01 | stegastamp | 0.4915 | 0.4945 | 0.4919 | 0.4974 | 0.7005 | 19 | 0.6 |
| message1 | pixelseal | 0.5022 | 0.4925 | 0.4989 | 0.4983 | 0.7103 | 15 | 0.5 |
| message1 | trustmark | 0.4982 | 0.5056 | 0.5165 | 0.4991 | 0.7252 | 10 | 0.6 |
| message1 | stegastamp | 0.4932 | 0.4933 | 0.4919 | 0.4940 | 0.7029 | 10 | 0.5 |

## Wniosek

Żadna waga zadania pomocniczego nie przeszła screeningu. Pozostawiamy dotychczasową głowicę i pomijamy pełne CV. Sprawdzić trafność bitów w tabeli względem shuffled i majority: wynik bliski 50% nie potwierdza użytecznego odczytu wiadomości.
To wynik tej adaptacji: losowo inicjalizowana głowica bitów, zamrożone cechy, mały zbiór i selekcja epoki według segmentacji. Nie obala wspólnego treningu embeddera i ekstraktora w WAM ani wykorzystania istniejącego, wytrenowanego dekodera bitów.

Checkpointy odtwarzają zapisane predykcje z tolerancją 1e-7. Nie użyto danych testowych i nie wysyłano submission.
Odtworzenie: `task-1/.venv/Scripts/python.exe task-1/experiments/local_message/run.py`. Artefakty w results/: checkpointy, predykcje, diagnostyka i podsumowanie.
