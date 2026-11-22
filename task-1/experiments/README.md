# Eksperymenty zadania 1

[Pełny indeks dokumentacji i wykonanych prac](DOCUMENTATION_INDEX.md).

## Aktualny wynik — zgłoszenie #3188

**Najlepszy potwierdzony publiczny mIoU: 0,6736**, +0,0059 wobec #3054.
Zmieniono PixelSeal na gałąź global/carrier EMA1561; TrustMark i StegaStamp
pozostają bez zmian. Ten sam rozwojowy fold 57 obrazów PixelSeal:
0,570883 → 0,581724. FG IoU ≤5% nadal wynosi 0.
[Raport i potwierdzenie serwera](pcmv/REPORT.md).

## Poprzedni wynik — zgłoszenie #3054

**Najlepszy potwierdzony publiczny mIoU: 0,6677**, wcześniej #3005: 0,6597 oraz #2675: 0,6440.
[Raport zgłoszenia](next_ideas/SUBMISSION_450_REPORT.md) · [Walidacja](next_ideas/REPORT.md).
Zmieniono wyłącznie StegaStamp:450 kroków adaptacji z75% replayem syntetycznym,
próg0,5, ustalony fallback dla pustych/pełnych masek. 3-fold OOF z tym samym
fallbackiem:0,724414→0,744788, delta+0,020373, CI95[+0,014373,+0,026550].
Czyste IoU przy WM>50%:0,491693→0,538708; FG<=1%:0,220095→0,239436.
Wszystkie340 testowych SS pochodzą z refitu na170 obrazach; PS/TM bitowo z#2675.
Serwer przyjął plik1020×128×128 uint8 (16,73MB)22.11 o04:15UTC.
Endpoint450 i kryteria wyboru ustalono przed wynikami450; publiczny wynik nie
wchodził do selekcji. To wielokrotnie używane foldy rozwojowe, nie niezależny test.

## Poprzedni wynik — zgłoszenie #2675

**Najlepszy potwierdzony publiczny mIoU: 0,6440**, wcześniej #2409: 0,6369.
[Raport zgłoszenia](local_presence/SUBMISSION_REPORT.md) ·
[Eksperymenty i negatywne wyniki](local_presence/REPORT.md).
PixelSeal: dodatkowe 150 kroków pełnego fine-tuningu z replayem na wszystkich
170 realnych obrazach. TrustMark: zamrożony skan cropów wyłącznie dla pustych
masek #2409, 8 korekt testowych. StegaStamp bez zmian.
Walidacja PS: **0,55748 → 0,57088 na jednym foldzie (57 obrazów)**;
TM: **0,65265 → 0,65705 na istniejących OOF (170 obrazów)**.
PS ≤1% nadal ma FG IoU 0; TM ≤1% rośnie 0,08059→0,24539 (3/9 trafień).
Nie znamy wyniku ukrytego zbioru końcowego ani publicznych wyników per metoda.

## Next ideas — nowe priorytety

[Audyt szans i wyników](next_ideas/REPORT.md) · [Protokół](next_ideas/PROTOCOL.md).
Nowe zadanie użytkownika po zatrzymaniu TriForensic: analiza strat, ostrożny
fallback dla pustych masek, 3-fold adaptacja natywnego StegaStamp i oracle decoder.
Około 48% mierzonego błędu przypada na czystą klasę w255 obrazach po przywróceniu
czystych obszarów. W tej analizie PS używa starszego OOF #2409 — nie nazywać
całości aktualnym OOF #2675. StegaStamp top1% fallback:0,69496→0,69761,
mały eksploracyjny zysk. Nowa adaptacja150 kroków: **StegaStamp OOF0,69496→0,72202**,
delta+0,02706, bootstrap95%[+0,01878,+0,03519], poprawa we wszystkich3 foldach.
Z ustalonym fallbackiem0,72441. FG<=1%0,18982→0,21166; clean IoU na WM>50%
0,44265→0,49107. Najmniejszych pominiętych dziur nadal4/12 (wcześniej3).
Kontynuacja450 ukończona i spełniła ustalone kryteria: OOF0,744788, publiczny
rekord#3054:0,6677. Szczegóły oraz negatywne wyniki są w raporcie.

## TriForensic-WM — aktualny screen

[Raport z pomiarami](triforensic_wm/REPORT.md) · [Protokół](triforensic_wm/PROTOCOL.md).
Jeden optimizer i OneCycle, pełne obrazy oraz pary patchy w każdej aktualizacji,
4 A100 na jednym nodzie. PixelSeal zatrzymany na 1561 krokach: **0,58164 mIoU**
na stałym foldzie 57 obrazów, wobec **0,57088** nowszego poprzedniego screenu.
**FG IoU ≤5% nadal 0**; wzrost lokalnego AUC nie przełożył się na małe maski.
Finalny refiner wnosi niemal zerową korektę. **Trening TrustMark przerwany na
żądanie użytkownika**: ostatni oceniony checkpoint 200 ma mIoU 0,64415 wobec
bazy 0,64662. Końcowy refit PixelSeal anulowany przed startem. Nie powstały
nowe predykcje testowe ani submission. Zachowano checkpointy i wyniki.

## Poprzednia baza — zgłoszenie #2409

**Najlepszy potwierdzony publiczny mIoU: 0,6369**, wcześniej #2163:0,6145.
PixelSeal: message_localizer po dopasowanej kontynuacji; TrustMark: model z wiadomością
plus propozycje NRGMark i weryfikacja cropów; StegaStamp bez zmian.
Walidacja rozwojowa: PS **0,58566** (3-fold OOF), TM **0,65265** (OOF + zamrożona
korekta bez GT), SS **0,69496** (synthetic-only). Protokoły nie są identyczne.
TM FG IoU1–5% **0,07164→0,50398**; <=1% **0→0,08059**, tylko1/9 trafień.
Krótki test nowych małych prostokątów/residualu PS nie rozwiązał problemu.

**Zacznij od [aktualnego handoffu](RESEARCH_HANDOFF_2026-11-22.md)**.
[Nowy raport](small_zoom/REPORT.md) · [Dane obiektowe](region_data/REPORT.md) ·
[Diagnoza RTX3070](small_region_diagnosis/REPORT.md) ·
[Poprzednie modele #2163](message_localizer/REPORT.md).
User dopuszcza pojedynczy ustalony fold jako szybką estymację kolejnej próby;
nie nazywaj jej pełnym OOF. Duże poprawy walidacyjne mogą prowadzić do nowych
zgłoszeń, bez strojenia hiperparametrów na leaderboardzie.

## Nowe próby małych watermarków — local_presence

[Raport i pełna tabela](local_presence/REPORT.md). Trzy kolejne alokacje jednego
node’a z czterema A100, łącznie **14 min 17 s** pracy node’a.
- TrustMark: skan cropów tylko dla pustych predykcji, bez GT, poprawił istniejące
  OOF **0,65265 → 0,65705**. FG IoU ≤1% **0,08059 → 0,24539**, trafienia **1/9 → 3/9**.
  Pozostałe 168 masek nie zmieniły się. To dwa trafienia, nie dowód dużego zysku testowego.
- PixelSeal: najlepszy nowy screen na jednym foldzie **0,55748 → 0,56013**.
  Trafiony tylko 1/7 najmniejszych znaków; FG IoU 1–5% nadal 0. Przedział bootstrap
  zmiany mIoU obejmuje zero. Nie nazywaj tego pełnym OOF ani potwierdzoną generalizacją.
- Pierwsze głowy 1×1/clean-hole i dalsza adaptacja na realnych patchach nie pomogły;
  regresje zapisano w raporcie. Czyste dziury pozostają nierozwiązane.
- **Po dalszym pełnym fine-tuningu wysłano #2675/public 0,6440; patrz aktualny raport powyżej.**

### Pełny fine-tuning po pytaniu o wykorzystanie danych

Na tym samym foldzie PS dalszy trening całego modelu dał **0,55748 → 0,57088**
(+0,01340; bootstrap zmiany 0,00774–0,01930). Replay online obejmujący 1212 źródeł
osiągnął 0,56524, więc nie przebił stałych 128 kompozycji. TM: stała pula 0,64096,
online 0,64620, baza 0,64662. To test wykorzystania istniejących danych, nie nowych
niezależnych źródeł. PS ≤1% nadal ma FG IoU 0; zysk dotyczy głównie większych masek.
Zgłoszony kandydat #2675: PS pełna kontynuacja + TM skan pustych masek,
SS bez zmian. Ostateczny status i odpowiedź serwera: [raport](local_presence/REPORT.md).

## Organizacja eksperymentów

[`pretrained_forensics`](pretrained_forensics/REPORT.md): TruFor, Mesorch i DetectiveSAM,
1530 zapisanych map na510 realnych obrazach. Diagnostyki direct/inverse/oracle,
kontrole geometrii oraz nested E1/E2 z63 wewnętrznymi głowicami cross-fit.
Baza seed42: **0,50624 mIoU**, E1 forensic: **0,46816**, E2: **0,47946**.
E2 poprawia własną kontrolę o0,00352, ale nie oryginalną bazę; StegaStamp
0,51514→0,51629 z CI obejmującym zero. **NO-GO dla rozbudowy tej fuzji**;
E3 pominięto. Bez nowych syntetyków, dostrajania modeli forensic i submission.

Każdy eksperyment ma własny katalog, np. `baseline/`, `sliding_window/` albo
`wykorzystanie_embeddingu/`, z kodem, opisem hipotezy i poleceniem uruchomienia.
Duże artefakty zapisujemy w ignorowanym `results/`. Historyczne eksperymenty
holdout używają podziału baseline'u (seed 42, 34 obrazy walidacyjne na metodę);
dalsza optymalizacja StegaStamp używa protokołu aktywnej bazy opisanego wyżej.
Zachowujemy oficjalną siatkę 128 × 128 i mIoU.
Wspólne loadery, metryki i walidacja submission pozostają w `wmloc/` i `src/`.

## Test 2000 syntetyków

[`synthetic_pretrain_2k`](synthetic_pretrain_2k/REPORT.md): StegaStamp,
2000 niezależnych coverów COCO, prawdziwy encoder, niezmieniona głowica inverse-STN.
Dopasowana kontrola GPU **0,46129** → synthetic→real **0,50719** OOF mIoU
(3 foldy × 3 seedy; 9/9 dodatnich par, CI95 zmiany +0,03495…+0,05725).
Synthetic-only: **0,51153** przy stałym progu 0,5; dodatkowej przewagi fine-tuningu
nie potwierdzono. Zysk dotyczy głównie czystej klasy i redukcji fałszywych alarmów;
foreground IoU nie wzrosło. Te liczby opisują rozwojową walidację StegaStamp.
Na późniejsze polecenie użytkownika wariant po fine-tuningu zgłoszono jako #1496:
publiczne mIoU całego zgłoszenia **0,5235**, poprawa **+0,0112** względem #1202.
[Reprodukcja](synthetic_pretrain_2k/REPRODUCE.md) i [raport zgłoszenia](synthetic_pretrain_2k/SUBMISSION_REPORT.md).

## Follow-up: continuation and shared clean-hole screen rejected

PS/TM450 continuation completed on the same57-image fold: PS fixed0.570883→0.568250,
PS online0.565243→0.564761, TM fixed0.640957→0.631045, TM online0.646204→0.645036.
No arm passed the predeclared selection gate; none was submitted. TM screen
uses capped original rectangles, without the later empty-mask scan. See
[next_ideas/MESSAGE_EXTENSION_REPORT.md](next_ideas/MESSAGE_EXTENSION_REPORT.md).

The new [IML-ViT clean-hole experiment](iml_holes/PROTOCOL.md) adapts the full
pretrained model across families on fold-training restoration images. Native512
input, explicit original1024 positional-parameter adaptation, no image resize,
FP32 first projection and deeper BF16. Target is clean regions; GT-free full
message agreement plus baseline predicted area gates a fixed equal blend.
All171 validation images receive newly computed original-decoder scores; prior
GT-filtered oracle rows are NOT used for routing. Official weights strict-load
and SHA256 checks passed. The150-update early-stop gate rejected this model: mean fixed-fold mIoU
0.659924→0.641382 (delta−0.018542); all three families regressed. No test
predictions or submission were made. See [iml_holes/REPORT.md](iml_holes/REPORT.md).

Completed: [PCMV](pcmv/REPORT.md), one PixelSeal scorer fit/evaluation.
The stronger EMA1561 global/carrier branch was deployed as #3188; TrustMark and
StegaStamp remain unchanged from #3054. Both PCMV and the proposal-only control
preserve the strongest base's 0.581724 mIoU on the fixed 57-image fold: calibration
selected no action, so neither made test predictions or a submission. The job
used GPUs 0–1, leaving GPUs 2–3 for the user's experiment, and has completed.
SS1000 was canceled while pending on user steering, with zero new updates.

## New best accepted: #3188 (2026-11-22)

Public mIoU **0.6736**, +0.0059 over #3054. Only PixelSeal changed to EMA1561
global/carrier branch; TrustMark and StegaStamp masks remain byte-identical.
Same57-image PS validation0.570883→0.581724. No all170 PS refit; trained113 real
images plus synthetic data. Small FG<=5% remains unresolved; see
[pcmv/REPORT.md](pcmv/REPORT.md) for exact buckets, artifact hash and receipt.
PCMV completed without gain (0.581724 → 0.581724); no duplicate submission or extended training. Tiny foreground and clean-hole failures remain unresolved.

## Native PixelSeal region screen: rejected

[Report](pixelseal_regions/REPORT.md): exact native thin-crop pairs from public
photos and GT-clean regions of real training images; new95,610-parameter region
classifier and500,494-box search, one fixed57-image development fold. The
predeclared geometry covers all16 small cases at IoU>=0.5 (oracle coverage only),
but recognition remains chance after400 updates: held-out AUC0.5027. FullFP32
audit AUC0.5016 confirms readout quantization alone does not rescue this checkpoint.
Inner calibration selects no changes; validation0.581724→0.581724, all16 small
foreground cases still missed. No test inference or submission. Two A100 workers,
143s allocated node time; node released. Best accepted remains#3188/public0.6736.
