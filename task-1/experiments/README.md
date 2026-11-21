# Eksperymenty zadania 1

## Aktualna baza — 2026-11-22

**Najlepsze zaakceptowane zgłoszenie: #2163, publiczny mIoU 0,6145.**
PixelSeal i TrustMark: nowe modele ze znaną wiadomością i odmrożonym oryginalnym
 ekstraktorem, po reficie; StegaStamp: natywny model RGB z pilota.
Walidacja rozwojowa: PS **0,57192**, TM **0,57872** (3-fold OOF, seed42),
StegaStamp **0,69496** (synthetic-only). Protokoły nie są identyczne.

**Zacznij od [aktualnego handoffu badawczego](RESEARCH_HANDOFF_2026-11-22.md)**:
wykonane eksperymenty, ograniczenia, wnioski, ścieżki checkpointów i dalsze cele.
[Raport nowych modeli](message_localizer/REPORT.md) ·
[wcześniejsze zgłoszenie #2008: 0,5993](native_hybrid_submission/REPORT.md).

Historyczna baza inverse-STN pozostaje właściwą kontrolą dla eksperymentów na
starym ekstraktorze StegaStamp. Nowy segmentator RGB nie ma STN i jego wyjść
nie wolno inverse-warpować. Nie zastępuj nowej bazy starszym wynikiem #1496.

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
