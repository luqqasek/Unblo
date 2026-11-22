# Brief researchowy: problemy po submission #2409

> Historical snapshot: this report describes the named earlier submission. The latest recorded best is #3054 (public mIoU 0.6677); see the [documentation index](../DOCUMENTATION_INDEX.md).

Stan zweryfikowany 2026-11-22. Analiza istniejących predykcji, konfiguracji i kodu; bez nowego treningu, inferencji GPU lub submission. Potrzebujemy maksymalnie trzech konkretnych eksperymentów na A100 40 GB, dających pierwszą informację o jakości w około 30 minut. Jeden ustalony fold wystarczy do screeningu. Priorytetem jest poprawa rzeczywistej walidacji, nie syntetycznego lossu.

**Obecny punkt odniesienia.** Publiczny wynik #2409 to **0,6369**, poprzednio #2163: 0,6145. Serwer nie podaje wyników per rodzina. Poniższe liczby to walidacja rozwojowa na 170 obrazach/metodę, nie wyniki rodzin na publicznym teście ani estymacja ukrytego finału.

| Metoda | mIoU walidacji | Protokół |
|---|---:|---|
| PixelSeal | 0,585655 | message_localizer, oryginalne 3 foldy OOF |
| TrustMark | 0,652655 | OOF 0,593332 + zamrożony detektor propozycji i weryfikacja cropu wiadomością |
| StegaStamp | 0,694962 | wcześniejszy native RGB, synthetic-only; zachowany bez zmian |

Foldy są wielokrotnie używane rozwojowo; skale detektora dobierano adaptacyjnie. Nie przedstawiaj ich średniej jako niezależnego CV całego systemu. Publiczna poprawa obejmuje również refity PS/TM, więc nie izoluje samej korekty cropów.

**Aktualny model już wykorzystuje oficjalny, trenowalny ekstraktor, znaną wiadomość, dwie skale, skipy stride 4 oraz lokalny kontekst z pierwszym blokiem stride 1.** Lokalny kontekst otrzymuje RGB, Y, high-pass RGB i JND dla PS. Dopasowanie do wiadomości wchodzi później jako sześć map: dot/cosine/soft-accuracy dla dwóch skal. Nie proponuj tych elementów jako nieobecnych. Obecne embedders, siła watermarka, raw bits, kwantyzacja i oficjalny downsampling pozostają niezmienione.

**Gdzie nadal tracimy wynik.** Rozmiar oznacza udział odpowiedniej klasy w oficjalnym GT128. IoU w dwóch pierwszych wierszach dotyczy watermarka, w kolejnych dwóch klasy clean. „Całkowite pominięcie” oznacza zerowe przecięcie predykcji z tą klasą, nie tylko pustą/pełną maskę.

| Grupa | PixelSeal: IoU; n; całkowite pominięcia | TrustMark: IoU; n; całkowite pominięcia |
|---|---|---|
| Watermark ≤1% | 0,0000; 17; 17/17 | 0,0806; 9; 8/9 |
| Watermark 1–5% | 0,0462; 26; 22/26 | 0,5040; 36; 10/36 |
| Clean ≤1% | 0,0123; 12; 10/12 | 0,0057; 14; 11/14 |
| Clean 1–5% | 0,0945; 26; 14/26 | 0,1478; 25; 12/25 |

Te cztery rozłączne grupy obejmują 81/170 PS i 84/170 TM. Odpowiadają za **56,61% / 53,63% sumy błędów `1 − mIoU`** w rodzinie. To opis koncentracji błędu, nie prognoza odzyskiwalnych punktów.

Na obrazach z watermarkiem >50% (85/metodę) clean IoU wynosi tylko **0,2859 PS / 0,2447 TM**. Ta szersza grupa odpowiada za 48,63% / 60,32% błędu rodzin, ale nakłada się na małe clean z tabeli — nie sumować udziałów. Termin „dziury” obejmuje tu także czyste fragmenty przy brzegu i wiele komponentów, nie tylko topologicznie zamknięte otwory.

**1. PixelSeal: lokalna reprezentacja małych i cienkich osadzeń nadal nie działa.**

Na ≤1% aktualny AUC wynosi 0,4959, AP 0,0057, a nawet próg koszyka dobrany z GT daje FG IoU tylko 0,0064. Średnie maksimum prawdopodobieństwa w GT to 0,0329, poza GT 0,3373. Samo obniżenie progu nie daje poprawnej lokalizacji. Odczyt z idealnego GT-bbox: 54,14% bit accuracy, żaden z 17 obrazów nie osiąga 90%. Dla PS 1–5% bbox daje 64,62% i tylko 3/26 osiągają 90%. Próg pełnego odczytu stosowany w TM nie jest więc gotowym rozwiązaniem PS; brak odczytu nie dowodzi braku wykrywalnego watermarka.

Nowy szczegół z analizy geometrii: wszystkie 17 najmniejszych PS to prostokąty, **7/17 ma aspect ratio >4**, maksymalnie 14,25; mediana krótszego boku to **18 px**, minimum 9 px. Ostatni generator `tiny_data.py` losował aspect w zakresie 1/4–4, więc nie obejmował tej części geometrii mimo limitu zadania do 16. To konkretna luka, lecz nie dowód głównej przyczyny: model nie nauczył się też prawidłowej lokalizacji na swoim tiny-dev.

Sprawdzone: 512 nowych train/64 dev osadzeń, 200 dodatkowych aktualizacji ×32 widoki. Synthetic-only mIoU real: kontrola 0,56324, tiny 0,55957, tiny+residual 0,55959. Jeden fold po adaptacji: 0,55706 / 0,55958 / 0,55958; FG≤1% i 1–5% w tym foldzie pozostają 0. Na 48 częściowo oznakowanych tiny-dev FG IoU zaledwie 0,00302. Łatwe czyste kontrole zawyżają ogólne dev mIoU około 0,6195.

Residual auxiliary: L1 signed-Y, 16 subpikseli na komórkę128, pixel shuffle do512, słaba waga 0,1. MAE na niezerowym residualu 1,378549 vs 1,378558 dla predykcji samych zer. Kod podpina tę głowę do lokalnego kontekstu **przed warunkowaniem wiadomością**. W kontrolach embed64 51,45% pikseli ma dokładnie zerową zmianę RGB; w niskim JND 74,88% (tylko 16 coverów). Nie utożsamiać zerowej zmiany z etykietą clean.

Pytania: jak nauczyć lokalnego wykrywania PS bez wymagania odzyskania wszystkich 256 bitów? Czy lepszy pierwszy test to klasyfikacja obecności na małych patchach, nadzór cechami nauczyciela, czy cel residualu zależny od wiadomości/energii zamiast podpisanego L1? Jak sprawdzić, czy zero jest warunkową medianą tego celu i czy gradient auxiliary wnosi użyteczny sygnał? Jak pokryć cienkie prostokąty zgodnie z limitem zadania i zweryfikować transfer, zamiast dopasować sampler do 17 znanych GT?

**2. TrustMark: sukces dotyczy lokalnych prostokątów, nie małych pozostałości w ogólności.**

Obecny frozen NRGMark-Q proponuje regiony na skalach 512/1536/2560/4096. Odczyt następuje z ORYGINALNYCH pikseli512 przez niezmieniony decoder konkurencji, native245. Szukamy ustalonej siatki rozmiarów/przesunięć wokół propozycji; akceptujemy najlepszy crop przy ≥90% zgodności bitów i powierzchni ≤10%, następnie dodajemy jego prostokąt do maski bazowej. GT/typ edycji nie są wejściami.

W grupie 1–5%: 19 prostokątnych przypadków poprawiło FG IoU **0 → 0,8192**, ale 17 nieregularnych pozostaje na **0,1517**. Dopasowany bbox tych nieregularnych daje średnio 48,94% bit accuracy. Wymóg odczytu 90% może z definicji pomijać poprawnie oznakowane pozostałości, które nie zachowują pełnej wiadomości. Gałąź prostokątna nie odtwarza ich kształtu.

Dla ≤1% tylko 1/9 obrazów ma wśród przeszukanych kandydatów maskę z IoU≥0,5; właśnie ten obraz odzyskano. **Żaden z ośmiu pozostałych nie ma takiego kandydata**, więc samo obniżenie progu odczytu nie naprawi obecnego zestawu propozycji. Szczególnie jednoznaczny przykład: `a7876fef0ed7`, GT bbox [243,270,277,337], 34×67 px, daje **100% bit accuracy**, ale aktualna gałąź ma `search_count=0`. To oznacza brak małego kandydata po filtrach, niekoniecznie brak dowolnego raw box. Dla innych pominiętych tiny prostokątów odczyt bbox jest słabszy, więc poprawa propozycji nie wystarczy każdemu.

Pytania: jak podnieść recall małych propozycji bez GT przy ograniczonej liczbie wywołań dekodera — dostrojenie detektora na dokładnych małych osadzeniach, skanowanie wycinków czy wyszukiwanie rozmiaru? Oddzielić brak propozycji, zły rozmiar, nieudany odczyt oraz zły kształt finalnej maski. Nie zaczynać od kolejnego obniżania progu. Jak niezależnie wykrywać nieregularną pozostałość, której prawidłowy crop nie daje odczytu wiadomości?

**3. Czyste dziury: obecna udana korekta konstrukcyjnie ich nie naprawia.**

Operator `baseline OR accepted_rectangle` może dodawać watermark, ale nie usuwa fałszywie dodatniego watermarka z czystej dziury. Nie oczekuj poprawy clean IoU od dokładania kolejnych dodatnich propozycji. Mocny globalny odczyt wiadomości również nie wskazuje, które lokalne piksele zostały nadpisane.

Na clean≤1% macro clean AP z surowych prawdopodobieństw wynosi 0,0262 PS / 0,0374 TM; dla clean1–5% 0,2336 / 0,2731. W obu rodzinach 24/85 obrazów z przewagą watermarka ma zerowe przecięcie z prawdziwą klasą clean, mimo że tylko 6/85 PS i 9/85 TM przewiduje dosłownie pełną maskę. To nie wyłącznie kolaps do pełnego obrazu: model bywa też pewny czystych pikseli w niewłaściwym miejscu.

Pytania: jaki lokalny sygnał pozwala wykryć nadpisanie watermarka przy silnym sygnale w otoczeniu? Jak zapobiec przenoszeniu kontekstu watermarka przez receptive field na małą czystą dziurę? Zaproponuj bounded test jawnie uczący obie polaryzacje mniejszości, z identycznymi edycjami i różnym watermark provenance. Rozważ niezależny refiner clean/overwrite lub uczenie lokalnej obecności, ale uzasadnij, jak unika interpretacji naturalnie niskiego JND jako dziury. Dla tych grup potrzebna jest osobna diagnoza progu clean; dawny globalny oracle nie jest dowodem, że każda przyszła lokalna korekta będzie nieskuteczna.

**Co już było sprawdzone i czego nie powtarzać bez nowej hipotezy.**

- Obiektowe maski, transplantacje, clean additions i rozdzielenie embed rect/final mask już weszły do generatora. Po dopasowanej adaptacji: PS 0,58548 vs kontrola 0,58566; TM 0,59333 vs 0,58761. Poprawa bbox fill nie rozwiązała małych regionów. Liczba komponentów nadal odbiega od realnych, lecz nie mamy dowodu, że jej poprawa sama podniesie wynik.
- Test równopowierzchniowych par pozycyjnych nie przebił dopasowanej kontynuacji; tiny sampling i słaba residual L1 również nie dały przełomu. To krótkie testy konkretnych implementacji, nie dowód fundamentalnego sufitu.
- Transfer detektora NRGMark-Q z TM do PS nie pomógł. Nie oznacza to nieskuteczności detektora wytrenowanego konkretnie dla PS.
- Nie wracaj do tezy „brak lokalnego stemu / brak wiadomości / wszystko stride32”. Nie obiecuj 0,70–0,80 na podstawie AUC ani obliczenia idealnego odzysku klasy.

**Oczekiwany rezultat researchu.** Zaproponuj maksymalnie trzy eksperymenty, uszeregowane według prawdopodobnego zysku i kosztu. Dla każdego podaj: testowaną hipotezę, konkretną zmianę względem kodu, wymagane dane/weights, sposób wykrycia pozornego sukcesu, metryki oraz kryterium kontynuacji lub odrzucenia. Oddziel twierdzenia potwierdzone eksperymentem od hipotez; dla zewnętrznych technik podaj pierwotne źródła i ich rzeczywiste ograniczenia.

Screening: jeden stały fold, identyczne obrazy i zbliżony budżet, raport globalnego mIoU oraz IoU/AP właściwej klasy dla ≤1%, 1–5%, 5–20% i czystych dziur, liczby przypadków i całkowitych pominięć. Selekcja progu bez GT ocenianego foldu; GT-crop tylko diagnostycznie. Nie wymagamy dziewięciu refitów przed poznaniem kierunku. Udana zmiana nie może niszczyć już działających TM prostokątów. Najpierw sygnał jakości w ~30 minut; dopiero potem większy trening i ewentualne zamrożenie receptury do submission. Nie stroić na leaderboardzie.

Artefakty do analizy: [aktualny raport](REPORT.md), [wyniki per obraz i hashe źródeł nowej analizy](results/research_bottlenecks.json), skrypt CPU (`research_bottlenecks.py`, original local checkout), [diagnoza prawdopodobieństw](results/latest_oof_diagnosis.json), [cztery skale TM](results/finalscale_summary.json), [dane obiektowe](../region_data/REPORT.md), [historyczna diagnoza cropów](../small_region_diagnosis/REPORT.md). Ten ostatni raport opisuje starszą bazę #2163; aktualne mIoU brać z niniejszego briefu.

Uwaga o metrykach: `fixed_crop90_per_image.json` powstał z masek binarnych, więc jego AP/AUC nie oceniają rankingu pełnych prawdopodobieństw. Do diagnozy reprezentacji używać oryginalnych FP32 maps.npz. Reprodukcja nowego zestawienia: `python3 task-1/experiments/small_zoom/research_bottlenecks.py`.
