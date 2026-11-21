# Audyt rzeczywistej architektury

Źródłem są wykonane modele, hooki PyTorch, rozszerzony graf ONNX i zapisane checkpointy, nie same nazwy warstw. Szczegółowe drzewa modułów, graf i SHA-256 znajdują się w `results/`.

## Wejście i ekstraktory

Wszystkie 510 obrazów ma 512×512 RGB uint8. `to_tensor` tworzy `[1,3,512,512]` float32 /255. Bez crop, letterbox, BGR i augmentacji. Każda rodzina ma oddzielną głowicę.

| Metoda | Rzeczywista klasa / konfiguracja | Wejście dekodera | Wagi |
|---|---|---|---|
| PixelSeal | `Videoseal.detector = ConvnextExtractor`, `ConvNeXtV2`, depths 3/3/9/3, channels 96/192/384/768 | bilinear antialias 256×256, align_corners=False; następnie x*2−1 | `checkpoints/pixelseal/checkpoint.pth`; card `pixelseal_card.yaml`; przypięty commit w provenance |
| TrustMark | Q, `tm.decoder.decoder.decoder`, torchvision ResNet-50, raw 100 bits, ECC wyłączone | bilinear antialias 245×245, align_corners=False; x*2−1 | zainstalowany pakiet `trustmark/models/decoder_Q.ckpt`, `trustmark_Q.yaml` |
| StegaStamp | wykonywany eksport WAVES ONNX, 100 bitów; STN + 7 convolutions + dense decoder | bicubic antialias 400×400, clamp [0,1], NHWC; STN; potem −0,5 w grafie | `checkpoints/stegastamp/stega_stamp_logits.onnx` |

Pełne wersje bibliotek, konfiguracje i hashe: [provenance.json](results/provenance.json). Hash potwierdza tożsamość lokalnego artefaktu, nie dowodzi tożsamości wag z oryginalnym treningiem autorów. Nie utożsamiamy eksportu ONNX ze wszystkimi wersjami upstream.

W StegaStamp sieć przewidująca macierz STN również widzi obraz wycentrowany: `Transpose__445` pobiera `gen_encoder/sub:0` (image−0,5). Sam sampler pobiera `image` w [0,1], a `gen_decoder/sub` odejmuje 0,5 po resamplingu. Są to dwa miejsca centrowania różnych ścieżek grafu, nie podwójna normalizacja tego samego tensora.

## Rzeczywiste miejsca ekstrakcji

| Model | Hook / tensor | Native B,C,H,W | Stride konwolucji na wejściu dekodera | Cache H,W | Układ |
|---|---|---|---:|---|---|
| PixelSeal | convnext.stages[0] po wszystkich 3 blokach | 1,96,64,64 | 4 | 32,32 | pełny obraz po resize |
| PixelSeal | convnext.stages[1] | 1,192,32,32 | 8 | 32,32 | pełny obraz po resize |
| PixelSeal | convnext.stages[2] | 1,384,16,16 | 16 | 16,16 | pełny obraz po resize |
| PixelSeal | convnext.stages[3] | 1,768,8,8 | 32 | 8,8 | pełny obraz po resize |
| TrustMark | net.layer1 | 1,256,62,62 | 4 | 32,32 | pełny obraz po resize |
| TrustMark | net.layer2 | 1,512,31,31 | 8 | 31,31 | pełny obraz po resize |
| TrustMark | net.layer3 | 1,1024,16,16 | 16 | 16,16 | pełny obraz po resize |
| TrustMark | net.layer4 | 1,2048,8,8 | 32 | 8,8 | pełny obraz po resize |
| StegaStamp | dec_conv_2/Relu:0 | 1,32,200,200 | 2 | 32,32 | po aktywnym STN |
| StegaStamp | dec_conv_4/Relu:0 | 1,64,100,100 | 4 | 32,32 | po aktywnym STN |
| StegaStamp | dec_conv_6/Relu:0 | 1,128,25,25 | 16 | 25,25 | po aktywnym STN |
| StegaStamp | dec_conv_7/Relu:0 | 1,128,13,13 | 32 | 13,13 | po aktywnym STN |

Stride w oryginalnych pikselach: pomnożyć przez 512/256, 512/245 lub 512/400. W StegaStamp dodatkowo działa macierz STN. `512/H` w CSV to średni rozstaw siatki, nie dokładny stride ani pole recepcyjne; np. 13 komórek StegaStamp nie oznacza stride 400/13. Padding i zaokrąglenia mają znaczenie.

Hooki sprawdzono na pełnych forwardach. Dla PS/TM każdy z czterech hooków wywołał się 176 razy: 170 obrazów + 6 przesunięć. Cache odtworzono liczbowo na 6 równomiernie wybranych obrazach każdej metody, bity na wszystkich 170. Zgodność nazw i masek sprawdzono dla wszystkich. ONNX: prześledzono zależności wszystkich czterech wyjść do `gen_stn/AddN:0`, a nie tylko dopasowano nazwy.

## Głowica

`frozen_extractor/run.py:Head`: każdy poziom → Conv1×1(C,32) → GroupNorm(8,32) → ReLU → bilinear do 32×32, align_corners=False. Concatenate 128 kanałów → Conv3×3(128,32,padding=1) → GroupNorm(8,32) → ReLU → Conv3×3(32,32,padding=1) → ReLU → Conv1×1(32,1). Logity są interpolowane do 128×128, następnie sigmoid. Predykcja to `probability >= threshold`.

**Ważna różnica względem skróconego opisu: adaptive average pooling do maksymalnie 32×32 następuje już przed zapisem cache i przed projekcjami głowicy.** Nie ma progressive FPN, top-down sumowania ani skipów U-Net. GroupNorm agreguje statystyki także po przestrzeni, więc lokalny convolutional decoder ma również zależność globalną.

## Tryb wykonania i walidacja

PS/TM po załadowaniu mają zero modułów w trybie train; oryginalne `extract()` dodatkowo ustawia requires_grad=False. Sprawdzono zamrożenie i niezmienność wszystkich buforów, w tym BatchNorm. Dropout/DropPath nie jest aktywny w eval. ONNX jest statycznym grafem inference. Głowice mają GroupNorm, bez BatchNorm i dropout; CV predict ustawia eval i inference_mode.

Odtworzono wszystkie 18 użytych checkpointów głowic: 9 original (3 metody × 3 foldy) i 9 stable StegaStamp (3 seedy × 3 foldy). Probabilities są zgodne z zapisami OOF do 1e−7, progi dokładnie zgodne. Train/validation rozłączne według niezmienionego manifestu. Skrypty finalnej predykcji używają tej samej ekstrakcji, `cv.predict`, sigmoid i `>=`; audyt nie uruchamia test inference ani submission.

Cache historycznie nie zawiera hashy checkpointów/preprocessingu — to ryzyko odtwarzalności, nie stwierdzona pomyłka. Audyt zapisuje te hashe w osobnym manifeście i weryfikuje je na końcu. Numeryczne statystyki native map: 6 obrazów/metodę; statystyki cache: wszystkie 170. Źródła: [feature_shapes.csv](results/feature_shapes.csv), [feature_statistics.csv](results/feature_statistics.csv), [cached_feature_statistics.csv](results/cached_feature_statistics.csv), [execution_checks.json](results/execution_checks.json).

## Pola recepcyjne

StegaStamp przy ustalonym STN: teoretyczne pola convolutions 2/4/6/7 wynoszą 7/19/43/75 pikseli obrazu 400×400; stride 2/4/16/32, środek pierwszej komórki 1/3/15/15 w indeksach pikseli. STN sam zależy od całego obrazu, więc całkowite pole modelu jest globalne. Pooling i interpolacja poszerzają zasięg.

TrustMark ResNet-50: maksymalna ścieżka convolutional daje 35/91/267/427 pikseli na siatce 245; to nie rozmiar aktywnego obiektu. PixelSeal ma GRN z normą po wymiarach przestrzennych w każdym bloku, więc prosta suma rozmiarów kerneli nie opisuje pełnego pola: jest globalne.

Wykonano gradientowy pomiar efektywnego zasięgu dla PS/TM: 3 obrazy/metodę, norma aktywacji centralnej pozycji, bez wyboru według GT. Promień obejmujący 90% bezwzględnej masy gradientu na obrazie 512 wynosi średnio około 258/269/259/264 px dla PixelSeal i 24/69/208/243 px dla TrustMark. To pomiar warunkowy, zależny od obrazu i celu gradientu; nie uniwersalna charakterystyka modelu. ONNX StegaStamp nie ma tutaj autograd: zamiast zgadywać gradient, zmierzono translation equivariance. [effective_receptive_fields.csv](results/effective_receptive_fields.csv).
