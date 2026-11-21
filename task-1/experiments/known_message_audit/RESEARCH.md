# Audyt lokalizacji ze znaną wiadomością

Stan: 22 listopada 2026. Research i audyt kodu; bez nowego treningu, pomiaru jakości, zmiany dashboardu i submission. Wyniki liczbowe poniżej pochodzą z istniejących raportów. Wnioski matematyczne oznaczają wyprowadzenia dla podanych założeń, a propozycje eksperymentów pozostają nieprzetestowane.

**Znana wiadomość daje uzasadniony sygnał zgodności, ale podstawowa propozycja została już sprawdzona w tym repozytorium.** Projekcja liniowa przed poolingiem, soft accuracy, zgodność znaków, względny signed dot i kanały per bit istnieją w `decoder_evidence`. PixelSeal i TrustMark nie uzyskały tam potwierdzonej poprawy. Najbardziej sensowna nowa próba to kontrolowana diagnostyka geometrii i skalowania istniejącego odczytu, następnie mały pilot BlurIG. Nie ma podstaw do prognozy zysku 0,03–0,08 ani wyniku 0,75 mIoU.

## 1. Co naprawdę implementują modele

**PixelSeal:** wrapper zmniejsza RGB do 256×256, normalizuje przez `2*x-1`, uruchamia ConvNeXtV2, `pixel_decoder.output_upscaling`, średnią przestrzenną i `linear`. Karta ma `pixelwise=False`, `upscale_stages=[1]`, ostatni wymiar 768 i globalne `nbits=256`. Wartość `nbits=16` wewnątrz szablonu jest nadpisywana przez `build_extractor`; nie oznacza 16-bitowego checkpointu. Sam etap `[1]` nie zmienia rozdzielczości, ale zawiera konwolucję 3×3, normalizację i aktywację: nie wolno projektować surowego wyjścia ConvNeXt z pominięciem tego etapu. Źródła: [wrapper](../../wmloc/watermarks.py), [karta](../../checkpoints/pixelseal_card.yaml), [builder](../../checkpoints/videoseal/videoseal/models/extractor.py), [Upsample](../../checkpoints/videoseal/videoseal/modules/common.py), [oficjalny PixelDecoder w lokalnym commicie](https://github.com/facebookresearch/videoseal/blob/b1bf4c081e40bc8971f072f33e3b16912a35f3b4/videoseal/modules/pixel_decoder.py#L71).

**TrustMark Q:** używana ścieżka to ResNet-50, ostatnia mapa `layer4`, global average pooling i `fc` do 100 bitów. Wrapper wykonuje resize do 245×245 i wywołuje bezpośrednio wewnętrzny ResNet. Nie należy zastępować tej ścieżki wyższym wrapperem `SecretDecoder`, który ma własne zasady resize. Źródła: [wrapper TrustMark](../../wmloc/watermarks.py), [lokalny SecretDecoder](../../checkpoints/trustmark-localization/python/trustmark/unet.py), [oficjalna konstrukcja ResNet](https://github.com/adobe/trustmark/blob/4c60b9fdb4125fca3001f0803e6890a6b55371ac/python/trustmark/unet.py#L409), [pooling w torchvision](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).

W obu przypadkach można zdefiniować `z_k(u)=W_k h(u)+b_k`, wtedy dokładnie algebraicznie `mean_u z_k(u)=W_k mean_u h(u)+b_k`. To równoważność **logitów**, nie `mean(sigmoid(z)) = sigmoid(mean(z))`. PixelSeal ma dodatkowy kanał 0, pomijany przy wiadomości. Oryginalne mapy mają około 8×8; interpolacja do 128×128 nie odzyskuje utraconej informacji przestrzennej. `decoder_evidence` sprawdził rekonstrukcję globalnego odczytu na wszystkich 170 obrazach na rodzinę: maksymalne różnice 1,62e-5 i 1,34e-5. To istniejący pomiar, nie nowa weryfikacja wykonana w tym audycie. [Kod i kontrola](../decoder_evidence/prepare.py), [raport](../decoder_evidence/REPORT.md).

**Równoważność algebraiczna nie dowodzi lokalności.** Duże pola recepcyjne mogą przenosić sygnał z watermarku do komórek tła. ConvNeXtV2 dodatkowo zawiera GRN z normą liczoną po wymiarach przestrzennych, więc nawet opis „wyłącznie lokalny CNN przed GAP” jest zbyt mocny. [Kod ConvNeXt](../../checkpoints/videoseal/videoseal/modules/convnext.py), [GRN](../../checkpoints/videoseal/videoseal/modules/common.py).

## 2. Signed dot, cosine i soft accuracy: podobne, lecz różne obiekty

Dla znanych bitów `m_k`, znaków `s_k=2*m_k-1` i lokalnych logitów `z_k(u)`:

| Score | Definicja | Interpretacja / ograniczenie |
|---|---|---|
| Signed dot | `D=sum(s*z)/K` | Zachowuje skalę logitów; kilka dużych kanałów może dominować |
| Cosine | `C=sum(s*z)/(sqrt(K)*||z||_2)` | Normalizuje energię; potrzebna obsługa zerowej normy |
| Relative agreement | `R=sum(s*z)/(sum(abs(z))+eps)` | Obecny kod; normalizacja L1, nie cosine |
| Soft accuracy | `P=mean(sigmoid(s*z))` | Miękka zgodność z wiadomością, nie skalibrowane prawdopodobieństwo lokalnej obecności |
| Hard accuracy | `H=mean(1[s*z>0])` | Odrzuca magnitudę; trzeba ustalić obsługę logitów równych zero |

Nie są równoważnymi rankingami pikseli. Sigmoid nasyca się, a normalizacje zmieniają względną wagę słabych i mocnych odpowiedzi. Tabela to bezpośrednia analiza wzorów; implementowane `P`, `H`, `R` oraz globalny `P` są w [spatial_evidence](../decoder_evidence/prepare.py). Samo dodanie `D` lub `C` jest małą ablacją już badanego mechanizmu, a nie nową metodą dekodowania.

## 3. Random-message calibration: wartość oczekiwana nie jest wynikiem pojedynczego losowania

Poniższe to wyprowadzenie warunkowe: ustalamy obraz i jego logity `z`, a następnie losujemy niezależne, uczciwe znaki wiadomości `r_k`.

`E[D_r|z]=0`, `Var(D_r|z)=sum(z_k^2)/K^2`.

Przy niezerowej normie `E[C_r|z]=0`, `Var(C_r|z)=1/K`. Dla soft accuracy `E[P_r|z]=0.5`, ponieważ `sigmoid(z)+sigmoid(-z)=1`. Hard accuracy ma średnią 0,5 dla niezerowych logitów; ścisłe `>0` daje zero dla remisów, więc nie należy przemilczać tego wyjątku.

Jedna błędna wiadomość nie daje zerowej mapy; średnia z ograniczonej liczby wiadomości również zwykle nie. Antytetyczne pary `r,-r` dają dokładnie zero dla signed dot/cosine i 0,5 dla soft accuracy, lecz nie są niezależnymi próbami do estymacji niepewności. Odjęcie znanej średniej zerowej niczego nie poprawia; odejmowanie jej estymacji Monte Carlo dodaje szum. Odjęcie 0,5 od `P` nie zmienia rankingu przestrzennego.

Warto odróżnić **skalowanie przez odchylenie null** od odejmowania jego średniej. `D/std_null = sum(s*z)/||z||_2 = sqrt(K)*C`: dla ustalonego K jest to dokładnie ranking cosine, a nie osobny nowy wariant.

Dla score log-prawdopodobieństwa też istnieje proste uproszczenie:

`log sigmoid(s*z) = s*z/2 - log(2*cosh(z/2))`.

Zatem `mean_k log sigmoid(s_k*z_k) - E_r mean_k log sigmoid(r_k*z_k) = D/2`. Jeśli ścieżka atrybucji jest wspólna i gradienty istnieją, kontrast atrybucji tych score również redukuje się do atrybucji `D/2`. To wyprowadzenie nie wymaga nowego źródła empirycznego.

**Nie wynika stąd, że zgodność prawdziwej wiadomości na tle ma średnią zero.** Rzeczywista wiadomość nie jest niezależna od obrazu z watermarkiem i jego cech; globalny kontekst może ujawnić ją również poza GT. Shuffling wiadomości z datasetu nie musi odpowiadać niezależnym uczciwym bitom. Rozkłady warunkowe względem skali, tekstury, pozycji i czystych coverów trzeba zmierzyć. Sens mają kontrole wieloma błędnymi wiadomościami, pary clean/marked i kalibracja na outer-train; nie należy interpretować ich jako gwarantowanej segmentacji.

## 4. Skąd bierze się sqrt(K), a czego nie obiecuje

W modelu `s_k*z_k = mu + epsilon_k`, z jednakową średnią i niezależnymi błędami o wariancji `sigma^2`, średnia ma SNR `mu*sqrt(K)/sigma`. To rachunek dla hipotetycznego modelu, nie wynik dla watermarków.

Przy korelacji błędów `rho` i jednakowej wariancji:

`Var(mean epsilon)=sigma^2*(1+(K-1)*rho)/K`,

`K_eff=K/(1+(K-1)*rho)`.

Ogólnie wariancja wynosi `1^T Sigma 1 / K^2`. Wspólne cechy dekodera, zależne błędy, nierówna trudność bitów, bias i zanik sygnału w małych regionach naruszają prosty model. Nawet rzeczywisty wzrost SNR globalnego score nie wyznacza wzrostu dwuklasowego mIoU. Liczby `sqrt(256)=16` i `sqrt(100)=10` nie są mnożnikami jakości lokalizacji ani uzasadnieniem +0,03–0,08.

## 5. BlurIG: kompletność dotyczy całki i endpointów

BlurIG całkuje gradient score wzdłuż ścieżki rozmycia, z pochodną obrazu po parametrze skali. Publikacja jawnie przybliża całkę sumą Riemanna. Oficjalny kod ma skończone `max_sigma`, liczbę kroków oraz numeryczną pochodną rozmycia. [Publikacja, równania 2–4](https://arxiv.org/html/2004.03383v1), [implementacja autorów](https://github.com/PAIR-code/saliency/blob/master/saliency/core/blur_ig.py).

Dla idealnej całki po ścieżce od `b=blur_sigma_max(x)` do `x`, suma **podpisanych** atrybucji wynosi `F(x)-F(b)` przy standardowych warunkach różniczkowalności. W implementacji skończonej to przybliżenie. Ani 16, ani 32 kroki nie gwarantują dokładnej kompletności. Rozmyty endpoint nie musi mieć score 0 ani być czystym coverem. Wzięcie modułu, ReLU, normalizacja min–max i późniejsze smoothing zmieniają tę sumę. [Aksjomaty i completeness](https://proceedings.mlr.press/v70/sundararajan17a.html).

Pilot powinien mierzyć błąd `abs(sum(A)-(F(x)-F(b)))`, endpoint score, zbieżność 16→32→64 kroków oraz stabilność przy innym `sigma_max`. Zdefiniować jawnie sigma versus wariancję, padding i kanały RGB. Własne wnioskowanie: gaussowska ścieżka nie gwarantuje braku artefaktów w skończonym obrazie ani zgodności z rozkładem dekodera; atrybucja poprawnie opisująca jego decyzję nadal może nie odtwarzać GT.

Wrappery PixelSeal/TrustMark mają `@torch.no_grad()`, więc ich publiczne `logits()` nie nadają się wprost do gradientów. Potrzebna jest ścieżka przez natywny model z zachowaniem preprocessing, `eval()` i zamrożonymi parametrami, ale gradientem po wejściu. StegaStamp przechodzi przez NumPy/ONNX Runtime; gradient wymaga osobnego wiernego modelu różniczkowalnego lub metody perturbacyjnej. [Kod wykonawczy](../../wmloc/watermarks.py).

## 6. Re-embedding: ta sama wiadomość nie oznacza tego samego residualu

`E(x,m)-x` jest residualem ponownego osadzenia na obecnym obrazie. Nie jest z definicji estymatorem wcześniej dodanego watermarku. Encodery dostają także treść obrazu, operują w przeskalowanych współrzędnych i wykonują blending/clipping/kwantyzację. TrustMark dodatkowo odejmuje średnią residualu; StegaStamp skaluje residual 400×400 z powrotem do rozmiaru wejścia. [Wrappery](../../wmloc/watermarks.py), [TrustMark encode](../../checkpoints/trustmark-localization/python/trustmark/trustmark.py).

Jeśli watermark został osadzony w cropie, pełnoobrazowe ponowne osadzenie zmienia skalę i fazę wzorca, kontekst oraz pozycję krawędzi. [Generator syntetyków](../synthetic_scaling/generate.py) rzeczywiście rozróżnia osadzanie na cropie od pełnego obrazu z przywracaniem czystych regionów. Nie ma powodu zakładać, że ponowne osadzenie tej samej wiadomości jest idempotentne. Adobe opisuje ponowne znakowanie jako nadpisanie poprzedniego watermarku, a nie pomiar jego lokalnej obecności. [Oficjalne FAQ](https://github.com/adobe/trustmark/blob/main/FAQ.md).

Test rozstrzygający musi mieć pary clean/marked z tym samym coverem, wiadomością i geometrią; następnie osobno zmieniać wiadomość, crop, skalę i fazę. Porównać correlation oraz RMS, bez zakładania znaku efektu. Istniejący [ReMark](../remark_localization/REPORT.md) badał remover, a nie tę samą procedurę re-embedding; jego słaby wynik nie obala wszystkich takich hipotez, lecz nie jest też dowodem ich sukcesu.

## 7. Mask-pooled loss, keep/drop i TTA

Dla przewidzianej maski `p_u` i pooling `z_bar=sum(p_u*z_u)/(sum(p_u)+eps)`, loss wiadomości może wybrać tylko najłatwiejszy fragment. Gdy `eps` jest pomijalne, globalne przeskalowanie `p` nie zmienia score: powierzchnia pozostaje nieidentyfikowalna. Przy `eps>0` mała masa wpływa na bias i może tworzyć dodatkowe optimum. Bez normalizacji model może powiększać maskę lub jej amplitudę. Jeśli `z_u` rozgłasza globalnie wiadomość, wiele dowolnych masek ma ten sam loss. To degeneracje wynikające bezpośrednio z definicji, nie stwierdzenie, że każda implementacja musi zawieść.

Jeżeli loss bitów jest po prostu ważony **przewidzianą** maską bez odpowiedniego ograniczenia, model może zmniejszać koszt przez jej wygaszenie. Nadzór GT nad maską i bity ważone **GT** eliminują część skrótów, lecz to już wariant podobny do [local_message](../local_message/run.py), który nie poprawił wyników. Nie podawać wiadomości jako wejścia głowicy, której zadaniem jest rekonstruować tę samą wiadomość bez kontroli kopiowania.

Keep/drop należy zdefiniować na obrazach `x_keep=p*x+(1-p)*b` i `x_drop=(1-p)*x+p*b`. Zachowanie odczytu w keep oraz jego utrata w drop badają wystarczalność i konieczność **dla tego dekodera oraz wybranego b**. Nie identyfikują jednoznacznie całego watermarku. Progi, fill, rozmycie krawędzi i powierzchnia maski zmieniają wynik. Optymalizacja maski per obraz może wykorzystać artefakty granic albo adversarialną słabość dekodera; wysoki score po TTA nie jest dowodem lepszego GT IoU. Praca o meaningful perturbation uzasadnia jawne modelowanie perturbacji, a nie traktowanie saliency jako segmentacji. [Fong i Vedaldi](https://arxiv.org/abs/1704.03296).

Do małego pilota: z góry ustalony budżet optymalizacji, niska rozdzielczość maski, ograniczenie oddalenia od początkowej segmentacji, kontrole random-message i clean, kilka neutralnych fill oraz niezależna ocena mIoU. Kalibracja progów wyłącznie wewnątrz outer-train. To proponowane zabezpieczenia eksperymentalne, nie gwarancja. Estetyka heatmapy nie wystarcza; sprawdzić wrażliwość na wiadomość i parametry modelu. [Sanity Checks for Saliency Maps](https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf).

## 8. BCH i surowe bity

W tej ścieżce PixelSeal dostaje 256 bitów jako `msgs`, TrustMark ma jawne `use_ECC=False`, a StegaStamp otrzymuje 100-bitowy tensor `secret`. Porównujemy dokładnie te bity i kolejność z metadanych. Nie wolno narzucać ograniczeń BCH z demonstracji innej konfiguracji: poprawne słowo kodowe nie zostało tu zadeklarowane. Sukces ECC przy globalnym odczycie i pewność lokalizacji byłyby zresztą odrębnymi miarami. [Wrappery datasetu](../../wmloc/watermarks.py), [gałąź encode TrustMark](../../checkpoints/trustmark-localization/python/trustmark/trustmark.py), [wcześniejszy audyt wiadomości](../local_message/RESEARCH.md).

## 9. Stan dowodów w repozytorium

| Kierunek | Co już zrobiono | Co wolno wnioskować |
|---|---|---|
| `decoder_evidence`: PixelSeal | Baseline 0,519131; map_only 0,483269; residual 0,521217, lecz wrong-trained residual 0,521776; per-bit residual 0,509690 | Znajomość wiadomości nie dała potwierdzonej poprawy w tym screeningu |
| `decoder_evidence`: TrustMark | Baseline 0,512931; map_only 0,463196; residual 0,505343; per-bit residual 0,505083 | Nie powtarzać tej konfiguracji jako niezbadanego pomysłu |
| `decoder_evidence`: StegaStamp | Per-bit occlusion + stabilna selekcja: średnia foldów 0,456206→0,476887; publiczne całe submission 0,5123→0,5109 | Rozwojowy sygnał wiadomości istniał, lecz nie potwierdzono transferu na publiczny test |
| `local_message` | Pomocnicze lokalne BCE: łączne 0,499205→0,472665 lub 0,479124; odczyt blisko 50% | Nie zadziałała ta losowo inicjalizowana głowica z zamrożonymi cechami; nie obalono wszystkich lokalnych dekoderów |
| `message_localizer` | W chwili audytu tylko `incumbent.py` i zapisane mapy/metadane OOF PS/TM | To odtworzenie bazy, nie wynik nowego localizera |

Źródła liczb: [decoder_evidence/REPORT.md](../decoder_evidence/REPORT.md), [local_message/REPORT.md](../local_message/REPORT.md), [message_localizer/incumbent.py](../message_localizer/incumbent.py). Historyczne wartości nie są wspólnym rankingiem z nowszymi syntetycznie trenowanymi bazami. Aktywny protokół StegaStamp wymaga zachowania inverse-STN, matched foldów i seedów 42/7/123. [Zasady rozwoju](../README.md).

## 10. Priorytety dalszego sprawdzenia

1. **Najpierw ustalić, gdzie jest odzyskiwalny lokalny sygnał.** Na istniejących syntetykach z parami cover/marked porównać znaną i błędne wiadomości, pełny obraz i poprawną geometrię cropa; rozdzielić patch oraz restore, małe regiony, czyste otwory i tło. Oracle crop służy diagnozie ograniczenia, nie ocenie wdrożeniowej. Jeśli nawet prawidłowa geometria nie daje lokalnego rankingu, dalsza fuzja tej mapy ma słabą przesłankę.
2. **Mała ablacją scalar score i kalibracji.** Na tych samych zamrożonych logitach porównać `D`, `C` i istniejące `R/P`; nie liczyć `sqrt(K)*C` jako niezależnej metody. Sprawdzić clean-calibration oraz kontrast matched geometry. Najpierw AP/FG IoU i błędy w otworach; następnie lekka fuzja z aktualną bazą, z kontrolą błędnej wiadomości o identycznej pojemności. Różnica względem starego `decoder_evidence` musi być jawna.
3. **BlurIG dla PS/TM jako nowy kanał dowodu.** Zaczynać od signed score i małej, wcześniej wybranej próby. Warunki kontynuacji: skończone i zbieżne gradienty, poprawne endpointy, przewaga lokalnego rankingu nad generic edges i random-message, a potem poprawa matched CV. To inna interwencja niż istniejące StegaStamp 4×4 occlusion, ale nie ma zmierzonej korzyści.
4. **Jeżeli preflight odczytu jest pozytywny: dostrajanie istniejącego dekodera na lokalnych syntetykach.** Zachować wytrenowaną projekcję bitową; trenować lokalny cel na wielu wiadomościach i niewidzianych coverach, z dopasowaną kontrolą segmentacyjną. To odróżnia próbę od nieudanego treningu bitów od zera. Wymaga osobnego eksperymentu, ponieważ równocześnie zmienia reprezentację i koszt obliczeń.
5. **Re-embedding i mask optimization/TTA dopiero po kontroli geometrii.** Koszt i ryzyko dopasowania do score są większe; priorytet niższy niż mierzalna diagnostyka istniejących dekoderów.

W każdym awansie: wybór hiperparametrów w inner-train/validation, identyczna baza i seedy, parowane wyniki per obraz, osobno obie klasy oraz przedziały niepewności. Wielokrotnie oglądane foldy pozostają rozwojowe; trzy seedy nie zwiększają liczby niezależnych obrazów. Próg +0,005 może być praktyczną bramką kosztową, nie dowodem statystycznym. Zakładanych +0,03–0,08 ani 0,75 mIoU nie wpisywać jako oczekiwanego wyniku. [Dotychczasowy audyt progów i ograniczeń walidacji](../roadmap_070/REPORT.md) pokazuje, dlaczego sam globalny odczyt lub lepszy próg nie wystarcza.
