# Brief do researchu: co sprawdziliśmy i co dalej

Stan na 21.11.2026. Zakres: Task 1 — lokalizacja watermarków PixelSeal, TrustMark i StegaStamp. To synteza lokalnych raportów i zapisanych wyników, bez nowego przeglądu literatury ani nowych treningów. „Brak wyniku” oznacza brak zakończonego pomiaru w przejrzanych artefaktach, a nie dowód nieskuteczności pomysłu.

## Punkt startowy dla kolejnego agenta

**Najlepsze zapisane zaakceptowane zgłoszenie: #1496, publiczne mIoU 0,5235**, wobec 0,5123 dla #1202. Nowy StegaStamp: pretraining na 2000 syntetyków, następnie fine-tuning na 170 rzeczywistych obrazach; PixelSeal i TrustMark zachowały predykcje #1202. Publiczny wynik dotyczy całego zgłoszenia, nie samego StegaStamp. [Raport i potwierdzenie serwera](../synthetic_pretrain_2k/SUBMISSION_REPORT.md).

**Aktywna baza rozwojowa:** zamrożony ekstraktor StegaStamp, cztery poziomy cech wyrównane inverse-STN i obecna głowica segmentacyjna. To decyzja użytkownika; zachować geometrię w kolejnych wariantach. Baza real-only osiąga średnie OOF 0,46512 po trzech seedach, lecz sam zysk korekty STN nie był jednoznaczny. W eksperymencie syntetycznym świeża dopasowana kontrola GPU ma 0,46129 — to właściwy punkt odniesienia dla jego efektu, zamiast historycznej kontroli CPU. [Baza STN](../stegastamp_stn_alignment/REPORT.md).

Dane: 510 obrazów treningowych, po 170 na rodzinę; obrazy 512×512, maski oceniane na siatce 128×128. Test obejmuje 1020 obrazów. Metryka konkursowa to średnia mIoU dwóch klas per obraz. Dla StegaStamp używamy tych samych trzech outer-foldów i seedów 42/7/123, z wyborem epoki i progu wyłącznie wewnątrz treningu. Seed 42 pozostaje domyślny.

## Co próbowaliśmy — wyniki i granice wniosków

| Ścieżka | Co rzeczywiście wykonano | Wynik i lekcja | Dowód |
|---|---|---|---|
| Proste punkty odniesienia | Pusta/pełna maska, przestrzenna częstość masek per metoda | Prior: holdout 0,2628, publiczne #1065 0,2934. Rozkład masek daje wynik bez analizy obrazu. | [Baseline](../baseline/README.md) |
| Gotowe lokalizatory | TrustMark bbox top1/suma oraz segmenter StegaStamp z mirrora; 5-fold ocena | TrustMark top1 OOF 0,3552 vs prior 0,2849. StegaStamp 0,2786 vs 0,2754: brak przekonującej przewagi; to konkretna kopia wag. | [Lokalizatory](../official_localization/README.md), [CV](../official_localization_cv/README.md) |
| Bbox, fallback i patche | Usunięcie pełnoobrazowego fallbacku, limit 4/20 bbox, patche 2×2 | TrustMark z pustą maską przy braku detekcji: OOF 0,4759. Patche: 0,2577; 93,2% wycinków bez detekcji. Więcej bbox nie pomogło. Publiczne #1096: 0,3416 dla całego zgłoszenia. | [Patche](../trustmark_patches/README.md) |
| Zamrożone ekstraktory + mała głowica | Cztery poziomy cech, projekcje 1×1, fuzja 32×32, BCE+Dice; holdout i 3-fold CV | OOF seed42: PixelSeal 0,50776, TrustMark 0,49581, StegaStamp 0,41052. Mocny skok względem priory, ale duża niestabilność StegaStamp. | [Głowice](../frozen_extractor/REPORT.md), [CV3](../frozen_extractor/CV3_REPORT.md) |
| Loss i rozdzielczość | BCE-only, proporcja BCE/Dice, balanced BCE, focal gamma 1/2, symmetric Dice, decoder128 | Żaden wariant nie przeszedł bramki screeningu pierwszego foldu. Łączna delta symmetric Dice −0,0090; decoder128 −0,0416. Brak pełnego CV tych wariantów; nie jest to odrzucenie wszystkich FPN/U-Net. | [Loss](../loss_screen/REPORT.md), [Ablacje](../head_ablation/REPORT.md) |
| Uczenie lokalnej wiadomości | Dodatkowa losowo inicjalizowana głowica bitów, wagi 0,1 i 1 | Bity blisko 50%, segmentacja gorsza; brak awansu do CV. Ta adaptacja na 170 obrazach nie nauczyła użytecznego dekodowania. | [Local message](../local_message/REPORT.md) |
| Stabilność selekcji | Diagnoza seedów, następnie 3 stratyfikowane inner-foldy zamiast jednego holdoutu dla StegaStamp | Średnia dziewięciu wyników fold–seed 0,4455 → 0,4610; poprawa 6/9. Ograniczenie załamania seedu42; publiczne #1202: 0,5123. | [Diagnoza](../stegastamp_diagnosis/REPORT.md), [Selekcja](../stegastamp_stable_selection/REPORT.md) |
| Informacja ze znanej wiadomości | Projekcje pretrained dekoderów PS/TM; StegaStamp: occlusion 4×4, mapy per-bit, residual/blend i kontrole błędnej wiadomości | StegaStamp + stabilna selekcja: OOF seed42 0,47683 vs 0,45621, ale publiczne #1299 0,5109 vs #1202 0,5123. Poprawa CV nie przeniosła się na publiczny test. PS/TM nie przeszły screeningu. | [Decoder evidence](../decoder_evidence/REPORT.md) |
| Audyt reprezentacji i geometrii | Preprocessing, GT, checkpointy, 510 OOF, shuffle, sondy warstw, tiny-overfit, granice, obszar, progi, STN | Potwierdzono niezgodność współrzędnych StegaStamp. Cechy mają sygnał przestrzenny, ale małe watermarki są słabo rangowane. Sama rozdzielczość 32×32 nie wyjaśnia skali błędu. | [Audyt](../diagnostic_audit/REPORT.md) |
| Pełna głowica po inverse-STN | 3 foldy × 3 seedy, trening od zera, matched kontrola stable | OOF 0,46094 → 0,46512, 5/9 dodatnich par; CI obejmuje zero. Seed42 się pogorszył. Poprawna geometria jest bazą rozwojową, nie samodzielnie udowodnionym wzrostem leaderboardu. | [STN](../stegastamp_stn_alignment/REPORT.md) |
| RGB / RGB+SRM / fuzja | 60 przebiegów: A/D pełne 3×3 dla trzech metod, B/C screening | RGB-only i RGB+SRM nie awansowały. Fuzja StegaStamp zwiększa small AP 0,0425 → 0,0564, ale mIoU 0,4682 → 0,4664. Wkład samego SRM nierozdzielony; fold3 nie potwierdza zysku rankingu. | [RGB/SRM](../rgb_srm/REPORT.md) |
| ReMark jako lokalizator TrustMark | Zamrożony remover, residual pełnego obrazu i okien 128/256/384, cechy, 51 map/wariantów, kontrole clean/matched/generic | Wybrany residual lokalny mIoU 0,3677 vs baza 0,4958; full residual 0,3364. Bramka użytecznego sygnału niespełniona, integracji nową głowicą nie trenowano. | [ReMark](../remark_localization/REPORT.md) |
| Gate, kalibracja i MoE według powierzchni | Klasyfikacja small/medium/large, regresja area, nested progi; eksperci small i clean-large, kontrola dalszego treningu, mieszanki | Gate rozpoznaje reżim, lecz nie naprawia lokalnego rankingu. StegaStamp A4: mIoU 0,4651 → 0,4735, large clean IoU 0,0796 → 0,1266; small AP spada. Prosta średnia A5 ma 0,4767, a przewaga A4 nad kontrolą U0 nie jest rozstrzygnięta. PS/TM bez awansu ekspertów. Bez submission. | [Regime MoE](../regime_moe/REPORT.md) |
| 2000 syntetyków StegaStamp | Prawdziwy encoder, 2000 różnych COCO + 200 osobnych walidacyjnych; 3 seedy pretrain, 9 fine-tuningów i 9 kontroli real-only | OOF 0,46129 → 0,50719, delta +0,04590, 9/9 dodatnich par, CI95 [+0,03495; +0,05725]. Publiczne #1496 0,5235. To najsilniejszy dotychczasowy dowód użytecznego dalszego kierunku. | [Syntetyki](../synthetic_pretrain_2k/REPORT.md) |

Liczby w różnych wierszach mają różne protokoły. Porównania przyczynowe opieramy na kontrolach wewnątrz eksperymentu, nie na rankingu wszystkich powyższych wartości.

## Czego się nauczyliśmy

1. **Wysokie mIoU nie musi oznaczać znalezienia watermarku.** Dla małego regionu pusta maska daje około 0,49 mIoU. Potrzebne są FG IoU, macro AP, clean IoU i wyniki według powierzchni. Top-K korzysta z prawdziwej powierzchni GT: jest diagnostyką, nie metodą inferencji.
2. **Największe nierozwiązane problemy to małe znaki i czyste fragmenty dużych masek.** Historyczny audyt #1202: 124/133 małych watermarków miało FG IoU <0,10, a 227/255 dużych masek traciło większość czystych pikseli. To statystyki starego modelu, nie pomiar #1496. [Częstości błędów](../diagnostic_audit/FREQUENCY_REPORT.md).
3. **Syntetyki poprawiły głównie klasę czystą.** Clean IoU 0,43426 → 0,53375; FG IoU 0,48833 → 0,48063, bez potwierdzonej poprawy. Dla GT≤5% FG IoU 0,01786 → 0,01665, mimo wzrostu AUC i AP. Model mniej fałszywie zaznacza, ale nadal słabo znajduje małe regiony.
4. **Fine-tuning po syntetykach nie ma jeszcze potwierdzonej wartości dodanej w mIoU.** Synthetic-only: 0,51153 przy progu 0,5; synthetic→real: 0,50719 z nested progiem. CI różnicy obejmuje zero. Różnią się również procedury kalibracji, więc nie jest to izolowany dowód szkodliwości aktualizacji wag. Synthetic-only nie został wysłany na leaderboard.
5. **Odczyt wiadomości i przewidywanie wielkości są łatwiejsze niż lokalizacja.** Globalne bit accuracy silnie koreluje z powierzchnią; gate MoE potrafi rozpoznać duże maski, ale nie wskazuje sam ich granic. Poprawa z eksperta clean-large nie dowodzi rozwiązania small-region detection.
6. **Większa głowica, nowy loss i wyższa rozdzielczość nie były automatycznym rozwiązaniem.** Tiny-overfit osiągał >0,96 train mIoU, a diagnostyczna rekonstrukcja GT przez siatkę32 około 0,95. To argument za badaniem generalizacji i lokalnej informacji; nie gwarancja wystarczającej reprezentacji w każdym przypadku.
7. **Walidacja rozwojowa ma ograniczenia.** Te same 170 obrazów na rodzinę były wielokrotnie oglądane. Trzy seedy to nadal 170 unikalnych obrazów. Bootstrap jest warunkowy względem zapisanych predykcji. Przykład #1299 pokazuje, że lepsze CV może nie poprawić publicznego testu; wynik prywatny pozostaje nieznany.

## Czego jeszcze nie rozstrzygnęliśmy

| Pytanie / wariant | Stan na moment briefu | Co trzeba ustalić |
|---|---|---|
| Syntetyki PS/TM; 2k vs 4k; liczba danych vs kroki treningu | Jest przygotowywany plan i kod w synthetic_scaling; brak raportu zakończonych pomiarów w momencie przeglądu | Najpierw sprawdzić aktualne artefakty. Plan obejmuje 2k/4k × 2500/5000 kroków dla trzech metod; nie przedstawiać tego jako potwierdzonego zysku. |
| Łagodniejszy fine-tuning i epoka0 | Brak zakończonego porównania | LR 0,0001 vs 0,001; dopuścić checkpoint przed fine-tuningiem w wewnętrznej selekcji, z kontrolą synthetic-only. |
| Mieszanie real/synthetic podczas fine-tuningu | Brak zakończonego porównania | Czy zachowuje lokalny sygnał i poprawia FG IoU bez utraty clean IoU? |
| Kontrolowana zmiana proporcji i kształtów masek | Brak ablacji generatora | Obecnie 45% lokalny embed(crop), 45% full embed z przywróceniem clean regionu, 5% full i 5% empty. Zbadać małe/cienkie regiony przy stałych 2k i liczbie kroków. |
| Realistyczne edycje po znakowaniu | Brak porównania w pretrainingu 2k | JPEG, resampling, inpainting/outpainting i inne edycje jako kontrolowany domain gap; dotychczasowy generator nie odtwarza pełnej domeny konkursu. |
| Wkład SRM w fuzji | Brak matched ablacji D z zerowymi kanałami SRM | Oddzielić informację residual od RGB i dodatkowych parametrów, szczególnie na foldzie3. |
| Syntetyki razem z MoE / RGB / wiadomością | Brak zakończonej łącznej ablacji | Czy zyski są komplementarne? Dotychczasowe MoE używało real-only bazy STN, nie synthetic checkpointów. |
| Progresywny FPN/U-Net, fine-tuning ekstraktora | Brak zakończonego porównania takich wariantów w raportach | Decoder128 był już testowany i przegrał; nowa propozycja powinna wskazać inną hipotezę i koszt. Ekstraktory dotąd zamrożone. |
| Temperature/isotonic calibration | Brak zakończonego pomiaru | Progi globalne i zależne od gate już testowano. Nowa kalibracja musi być nested i oceniana obok rankingu. |
| Pełne sliding-window dopasowanie wiadomości | Brak osobnego zakończonego eksperymentu tego typu | Wykonane patche detektora, occlusion4×4 i okna ReMark to inne operacje. Sprawdzić lokalną informację i koszt przed treningiem. |
| Nowa niezależna walidacja i grupowanie po źródle | Brak nowego nietkniętego zbioru; źródłowe ID konkursu niedostępne | Hash/dHash ograniczają ryzyko duplikatów, ale nie wykluczają wspólnych źródeł po silnych edycjach. |

[Plan synthetic_scaling](../synthetic_scaling/README.md). W starszym audycie RGB/SRM, pełny trening po STN i syntetyki były opisane jako niewykonane — późniejsze raporty powyżej zastępują ten historyczny status. Podobnie kalibracja zależna od reżimu i eksperci MoE są już sprawdzeni.

## Pytania do dalszego researchu — kolejność robocza

To priorytety wynikające z naszych eksperymentów, do sprawdzenia w źródłach przez kolejnego agenta.

1. **Dane syntetyczne i transfer między rodzinami.** Ustalić stan synthetic_scaling; ocenić, jak oddzielić różnorodność coverów, budżet kroków i realizm lokalnego embedowania. Przed proponowaniem 10k+ wykorzystać zaplanowane kontrole 2k/4k.
2. **Zachowanie sygnału pretrain podczas dostrajania.** Zbadać uzasadnienie LR10× mniejszego, epoki0 i mieszania danych. Najtańszy nowy test może wykorzystać istniejące 2k i checkpointy bez kolejnej generacji.
3. **Lokalizacja małych regionów.** Poszukać metod uczenia lokalnej, zależnej od wiadomości reprezentacji przy stałym pretrained encoderze watermarku. Wyjaśnić, co miałoby działać inaczej niż przegrana losowa głowica bitów, patche i obecne sondy.
4. **Odzyskiwanie czystych fragmentów.** Sprawdzić, czy ekspert clean-large wnosi coś ponad syntetyki, prostą średnią ekspertów i dalszy trening. Każde połączenie ocenić matched ablacją.
5. **Architektura dopiero z konkretną diagnozą.** Ewentualny FPN, fine-tuning ekstraktora lub gałąź residual uzasadnić mierzalnym lokalnym sygnałem i kontrolą kosztu; sam większy model nie jest jeszcze wnioskiem z danych.

## Gotowe polecenie do przekazania agentowi

Zbadaj dalsze kierunki dla Task1: lokalizacji istniejących watermarków PixelSeal, TrustMark i StegaStamp. Przeczytaj ten brief, bieżący indeks task-1/experiments/README.md i przywołane raporty. Najpierw sprawdź, czy synthetic_scaling ma już nowe wyniki — jego stan w briefie to plan bez zakończonego raportu. Nasze najlepsze zapisane publiczne zgłoszenie to #1496=0,5235; aktywna geometria StegaStamp to inverse-STN. Najsilniejszy wynik rozwojowy to 2k syntetyków: matched OOF 0,46129→0,50719, ale bez poprawy foreground IoU. Synthetic-only=0,51153, a małe znaki nadal lokalizujemy słabo.

Przygotuj research oparty na publikacjach, oficjalnym kodzie i dostępnych checkpointach. Dla każdego kierunku podaj źródło, mechanizm, zgodność z naszymi encoderami i wiadomościami, potrzebne dane, koszt oraz różnicę względem wykonanych prób. Rozróżnij metody lokalizujące cudzy istniejący watermark od metod wymagających wspólnego treningu nowego embeddera. WAM/MaskWM były już inspiracją generatora; nie traktuj ich jako gotowych zamienników dekodera StegaStamp.

Wynikiem mają być trzy uszeregowane eksperymenty oraz jeden rekomendowany pierwszy test. Dla każdego zapisz hipotezę, jedną główną zmienną, matched kontrolę, podział danych, budżet, kryterium sukcesu i przerwania. Raportuj mIoU obu klas, macro AP/AUC, FG IoU dla GT≤5%, clean IoU dla GT>50% i granice. Top-K oznacz jako diagnostykę wykorzystującą GT. Zachowaj trzy outer-foldy, seedy42/7/123 i inner-only selekcję; porównuj na tym samym urządzeniu. Oddziel rozwój na znanych foldach od niezależnej walidacji i leaderboardu. Wyjaśnij, jak proponowany test odróżni rzeczywistą lokalizację od przewidywania dominującej klasy. Zakończ, gdy wszystkie trzy propozycje mają źródła, kontrolę, koszt i mierzalną decyzję; to zadanie researchowe, bez uruchamiania treningów i wysyłania zgłoszeń.

## Odtworzenie i utrzymanie briefu

Repozytorium: European-AI-Hackathon, katalog task-1. Źródło briefu: experiments/research_handoff/REPORT.md. Przed kolejnym użyciem sprawdź indeks eksperymentów, raport synthetic_scaling i najnowsze upload_receipt.json. Starsze raporty zachowują historyczne sformułowania „najlepszy model”; bieżący stan ustalaj z najnowszych potwierdzeń.

Checkpointy i duże dane są ignorowane przez Git, więc sam clone nie dostarcza artefaktów. [Reprodukcja eksperymentu 2k](../synthetic_pretrain_2k/REPRODUCE.md) opisuje istniejące środowiska i polecenia. Lokalne linki dashboardu działają na maszynie z serwerem lub przez tunel SSH; do agenta bez dostępu do tej maszyny przekaż pobrany tekst Markdown.
