# Eksperymenty zadania 1

## Aktywna baza do dalszej optymalizacji

Decyzja użytkownika: **StegaStamp po korekcie STN**, eksperyment
[`stegastamp_stn_alignment`](stegastamp_stn_alignment/REPORT.md), jest od teraz
modelem bazowym do dalszej optymalizacji. Powód: usunięcie potwierdzonej
niezgodności współrzędnych cech i masek, mimo mieszanych wyników obecnego CV.

Kolejne warianty StegaStamp powinny zachować wyrównanie czterech poziomów cech
przed pełną głowicą. Punktem odniesienia są checkpointy
`stegastamp_stn_alignment/results/stegastamp_fold{1,2,3}_seed{42,7,123}.pt`,
cache `results/aligned_features.pt` i podziały `results/splits.json` tego eksperymentu.
Porównania wykonujemy na tych samych trzech zewnętrznych foldach i wszystkich
seedach 42/7/123, ze stabilnym doborem epoki i progu wewnątrz treningu.
Seed 42 pozostaje domyślnym seedem pojedynczego przebiegu; wynik seedu 123
nie stanowi decyzji o zmianie domyślnego seedu.

Wynik bazowy StegaStamp: średnie OOF mIoU **0,46512** po trzech seedach;
seed 42: **0,45169**. Przyjęcie bazy rozwojowej nie oznacza potwierdzonej przewagi
na leaderboardzie. Najlepsze zaakceptowane zgłoszenie to obecnie **#1496**
z wynikiem **0,5235**: syntetyczny pretraining StegaStamp i fine-tuning na wszystkich
170 rzeczywistych obrazach. PixelSeal i TrustMark zachowują predykcje ze zgłoszenia #1202.
[Raport zgłoszenia](synthetic_pretrain_2k/SUBMISSION_REPORT.md).

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
