# RTX 3070: pełne kanały bitowe i zachowanie dekodera

Status: zakończony test diagnostyczny, brak potwierdzonej poprawy i brak nowego submissionu. Rozbieżność z historycznym baseline wymaga wyjaśnienia przed pełnym treningiem.

## Protokół

RTX 3070, FP32, seed 42. Zamrożony aktualny MessageLocalizer po 400 krokach syntetycznych (nie końcowe wagi najlepszego adaptowanego modelu). Oryginalne 3 foldy, 170 obrazów na metodę. Trening głowy wyłącznie na części treningowej danego foldu: 200 kroków, batch 16, AdamW, LR 0.001. Stały próg 0.5 i ostatni krok, bez wyboru na outer validation. Bez dodatkowego JPEG, blur lub augmentacji skali.

Porównanie: brak korekty; głowa na 6 skalarnych mapach; głowa na wszystkich podpisanych kanałach bitowych obu skal; kontrola z losową wiadomością w kanałach korekty. Bazowy lokalizator nadal dostaje poprawną wiadomość także w tej kontroli. Głowy mają po 16 kanałów ukrytych, więc nie mają identycznej liczby parametrów. To tani screening na zamrożonych cechach, a nie pełna ablacja uczenia całego modelu. Wszystkie mapy cech są sprowadzone do 32×32; ma to ograniczoną moc dla bardzo małych obiektów.

## Wyniki lokalizacji

mIoU = średnia per-image. Pozostałe kolumny to foreground IoU w zadanych koszykach. Clean holes = metadata full_watermark; ta grupa nakłada się na koszyki powierzchni.

| Metoda | Wariant | mIoU | ≤1% | 1–5% | 5–20% | >20% | Clean holes |
|---|---|---:|---:|---:|---:|---:|---:|
| pixelseal | baseline | 0.51836 | 0.00000 | 0.02175 | 0.16940 | 0.85459 | 0.90899 |
| pixelseal | scalar | 0.51962 | 0.00000 | 0.02182 | 0.16620 | 0.85427 | 0.90830 |
| pixelseal | bits | 0.51064 | 0.00000 | 0.01576 | 0.17221 | 0.83760 | 0.88908 |
| pixelseal | bits_wrong | 0.50623 | 0.00000 | 0.01340 | 0.18756 | 0.82364 | 0.88011 |

Liczebności pixelseal: le1: 17, 1to5: 26, 5to20: 30, gt20: 97, clean_holes: 85.
Pełne bity minus skalarne mapy: -0.00897; warunkowy paired bootstrap 95%: [-0.018069090988777448, 9.448393147729465e-05]. Ten przedział nie uwzględnia zmienności ponownego treningu.

| Metoda | Wariant | mIoU | ≤1% | 1–5% | 5–20% | >20% | Clean holes |
|---|---|---:|---:|---:|---:|---:|---:|
| trustmark | baseline | 0.51503 | 0.00000 | 0.04020 | 0.23145 | 0.86259 | 0.90057 |
| trustmark | scalar | 0.51600 | 0.00000 | 0.03536 | 0.20999 | 0.86055 | 0.89805 |
| trustmark | bits | 0.51651 | 0.00000 | 0.02966 | 0.22779 | 0.86273 | 0.89988 |
| trustmark | bits_wrong | 0.51447 | 0.00000 | 0.03524 | 0.20229 | 0.85630 | 0.89293 |

Liczebności trustmark: le1: 9, 1to5: 36, 5to20: 25, gt20: 100, clean_holes: 85.
Pełne bity minus skalarne mapy: +0.00051; warunkowy paired bootstrap 95%: [-0.0011589099193152627, 0.0021074410983525295]. Ten przedział nie uwzględnia zmienności ponownego treningu.

## Odczyt wiadomości

16 istniejących coverów COCO, te same znane losowe wiadomości i oficjalny uint8 embedder. Każdy rozmiar jest osobnym osadzeniem w całym cropie, a nie wycięciem z już oznakowanego obrazu. Odczyt na natywnej skali. To dane syntetyczne; nie diagnoza małych regionów konkursowych.

| Metoda | Bok regionu | Oryginalny dekoder | Po synthetic400 |
|---|---:|---:|---:|
| pixelseal | 64 | 0.6160 | 0.6609 |
| pixelseal | 128 | 0.9189 | 0.8989 |
| pixelseal | 512 | 0.9993 | 0.9995 |
| trustmark | 64 | 0.9425 | 0.8650 |
| trustmark | 128 | 0.9956 | 0.9113 |
| trustmark | 512 | 1.0000 | 0.9194 |

## Weryfikacja i ograniczenia

- Powtórna inferencja od świeżo wczytanego checkpointu bez testu drift i bez hooków odtwarza bieżący baseline: PS 0.51836, TM 0.51503. Maksymalna różnica logitów wobec cache <0.00015.
- Historyczne wyniki tych checkpointów: PS 0.51958, TM 0.55929. Przyczyna rozbieżności pozostaje nierozstrzygnięta. Nie porównywać obecnych wartości bezpośrednio do najlepszego adaptowanego OOF.
- SHA-256 źródłowego pełnego checkpointu TrustMark zgadza się z kopią; wszystkie tensory obu odchudzonych checkpointów są dokładnie równe oryginałom. Potwierdzono zgodność SHA-256 metadata.json, masks_128.npz i jednego obrazu TrustMark na Windows i JURECA. Nie wykonano pełnego audytu hashy wszystkich obrazów.
- Kod modelu i ekstraktora jest zgodny. Natywna skala TrustMark 245. Oryginalny wrapper i oryginalny gęsty ekstraktor mają 100% zgodności bitów na 4 obrazach każdej rodziny; maksymalna różnica logitów <0.000006.
- Wagi i źródła pobrano z JURECA; wszystkie obliczenia tej serii uruchomiono na Windows RTX 3070. Nie uruchomiono treningu na klastrze.

## Decyzja

Pełne kanały bitowe w tej konfiguracji nie rozwiązują małych masek. Nie jest to dowód, że pełne bity nigdy nie pomogą: nie uczono całego modelu, a cechy mają rozdzielczość 32×32. TrustMark wykazuje spadek bit accuracy w porównaniu wewnątrz tego samego środowiska; dodatkowy loss zachowania wiadomości jest sensowną następną hipotezą, nie potwierdzoną naprawą. Przed drogimi eksperymentami należy odtworzyć historyczny baseline i osobno sprawdzić adaptowane checkpointy. Żaden wariant nie zastępuje aktualnego best.

## Artefakty

run.py, verify.py, forward_check.py oraz results/{family}/*.json. Na Windows zachowano też features.pt, głowy per fold i pełne mapy float32 w NPZ. Checkpointy i duże wyniki pozostają poza Git.
