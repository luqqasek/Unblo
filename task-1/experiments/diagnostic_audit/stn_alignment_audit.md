# StegaStamp: aktywny STN i niezgodność współrzędnych

## Co wykonuje rzeczywisty model

W grafie `stega_stamp_logits.onnx` działają `gen_stn/stn_conv_*`, dense, `gen_stn/MatMul`, `gen_stn/add`, `gen_stn/Reshape:0` (macierz B×2×3), sampler GatherND i `gen_stn/AddN:0`. Ten ostatni tensor trafia do `gen_decoder/sub` i wszystkich czterech pobieranych cech. Zatem **cechy są po STN, a GT pozostaje w układzie oryginalnego obrazu**. Obecny cache i głowica nie odwracają transformacji.

To ustalono przez graf zależności i wykonanie sesji z dodatkowymi wyjściami, bez zmiany oryginalnego pliku ONNX. Oryginalny kod autorów również zawiera STN, ale nie zastępuje to weryfikacji lokalnego eksportu: [StegaStamp models.py](https://github.com/tancik/StegaStamp/blob/master/models.py). Kolejność centrowania obrazu i sampler w audycie wynikają z rzeczywistego ONNX, nie z założenia o tożsamości wersji upstream.

## Konwencja współrzędnych

Macierz A mapuje punkt wyjściowej siatki STN na wejście, nie odwrotnie. Siatka docelowa ma `linspace(-1,1,400)`. Wykonany graf przelicza znormalizowaną współrzędną na piksel jako `(A*u + 1)*199`: stała wynosi 398/2, a nie standardowe 399/2 ani 400/2. Indeksy są przycinane do 0…399 przed wagami interpolacji; poza zakresem daje to m.in. czarne brzegi. Samo przyjęcie `affine_grid(align_corners=False)` byłoby błędne.

Napisano niezależny bilinear sampler zgodny z węzłami ONNX. Maksymalny błąd wobec rzeczywistego obrazu po STN na 6 obrazach: 2,81e−5. Punkty affine po transformacji i odwróceniu wracają z błędem 5,6e−17. Dodatkowy test używa macierzy kompensującej dokładnie skalę 398/399 i sprawdza tożsamość wszystkich siatek.

Korekta probability: oryginalny piksel 128 jest mapowany do środka odpowiedniego piksela w 400 (`(j+0,5)*400/128−0,5`), następnie do współrzędnych źródła STN, przez A^-1 i do siatki odczytu. `grid_sample` używa align_corners=False, bilinear, padding_mode=border. To korekta mapy wyjściowej głowicy, która wcześniej uczyła się bez korekty; nie gwarantuje, że głowica reprezentuje dokładnie układ cech.

Korekta cech: dodatkowo uwzględnia rzeczywiste stride 2/4/16/32, środki pierwszych komórek 1/3/15/15 i środki binów adaptive-average-pooling. Dla pooled map ich rozstaw nie jest idealnie równomierny; odwrócenie korzysta z odcinkowo liniowej mapy środków. Informacja utracona przez pooling i padding nie jest odtwarzana.

## Rozkład transformacji: 170 obrazów

| Wielkość | Średnia | SD | Min–max |
|---|---:|---:|---|
| Obrót z atan2(a10,a00), stopnie | −0,209 | 1,007 | −2,480…3,393 |
| Większa skala (singular value) | 1,0735 | 0,0279 | 1,0131…1,1810 |
| Mniejsza skala | 1,0406 | 0,0243 | 0,9552…1,1018 |
| Przesunięcie X, przybliżone piksele 512 | −2,331 | 5,573 | −21,239…15,652 |
| Przesunięcie Y, przybliżone piksele 512 | −1,021 | 7,558 | −32,837…25,499 |
| Maksymalny ruch narożnika, przybliżone piksele 512 | 31,863 | 11,860 | 7,418…69,398 |

Skala opisuje mapowanie destination-to-source, nie powiększenie obiektu na ekranie. Shear zapisano jako cosinus kąta między kolumnami części liniowej, bez arbitralnej dekompozycji affine. Wszystkie macierze mają dodatni wyznacznik. Narożniki i translacje w 512 są opisowym przeliczeniem normalized coordinates ×256; dokładny sampler wykorzystuje stałą 199.

## Wyniki trzech odrębnych testów

| Test | Przed | Po | Różnica | Protokół |
|---|---:|---:|---:|---|
| Odwrócenie STN na probabilities original | 0,410515 | 0,411363 | +0,000848 | 170 OOF, ta sama głowica |
| Odwrócenie STN na probabilities stable seed42 | 0,456213 | 0,455479 | −0,000734 | 170 OOF, ta sama głowica |
| Odwrócenie STN na cechach stable seed42 | 0,456213 | 0,425896 | −0,030317 | 170 OOF, ta sama głowica, bez retreningu |
| Ponownie uczona sonda liniowa, raw vs aligned | 0,403753 | 0,423537 | +0,019784 | 3 istniejące foldy, seed42, 30 epok, BCE |

W sondzie poprawa występuje na każdym foldzie: +0,02016 / +0,02329 / +0,01583. Parowany bootstrap 95% dla średniej różnicy mIoU: [0,01612; 0,02333]. Macro ROC-AUC 0,65181 → 0,67094, AP 0,56172 → 0,57008, FG IoU 0,38761 → 0,40868. Potwierdzenie obejmuje fold screeningowy; nie jest niezależnym testem wyboru wariantu.

W obu wariantach sondy identyczne: splity, seed, capacity, BCE, optimizer, liczba epok, inner wybór progu. Po korekcie cech sonda jest trenowana od nowa; nie utożsamiamy tego z poprawą gotowej głowicy ani publicznego wyniku. Interpolacja podczas wyrównania również wygładza cechy — brak osobnej kontroli o identycznym wygładzaniu ogranicza czysto przyczynową interpretację.

## Przesunięcia i equivariance

Test probabilities: {−4,−2,+2,+4} komórki osobno na osi X/Y, identity oraz horizontal/vertical/both flips. Najlepszy wspólny shift StegaStamp daje tylko +0,00019; flipy pogarszają. Per-image wybór transformacji jest wyłącznie oracle zapisanym w CSV. Nie ma dowodu na globalny błąd odwrócenia osi czy stały offset.

Na 6 obrazach/metodę przetłumaczono RGB o +16 px w prawo, bez resize, edge padding, oceniono środek z odciętym marginesem. Korelacja przesuniętej mapy predykcji: PixelSeal 0,910, TrustMark 0,860, StegaStamp 0,438. To nie izoluje STN od dużego receptive field, aliasingu lub globalnej normalizacji, ale pokazuje szczególną niestabilność StegaStamp. Native mapy i ich statystyki są w [equivariance.csv](results/equivariance.csv).

## Wniosek

**Tak, występuje mierzalna niezgodność układów współrzędnych, a ponowny trening sondy po jej korekcie pomaga. Nie jest to całe wyjaśnienie słabej lokalizacji.** Proste odwrócenie gotowych predykcji nie pomaga, sonda aligned nadal jest słaba, szczególnie wobec problemu małych regionów. Najbardziej uzasadniony kolejny eksperyment to ta sama istniejąca głowica trenowana od początku na wyrównanych cechach, z identycznym stable nested wyborem i kontrolą raw. Nie należy podmieniać cech w aktualnym checkpointcie produkcyjnym.

[Macierze](results/stn_transforms.csv), [korekta probabilities](results/stn_corrections.csv), [korekta cech](results/stn_feature_correction.csv), [potwierdzenie sondy](results/alignment_probe_confirmation.json), [przykład transformacji](results/figures/stn_0.png).
