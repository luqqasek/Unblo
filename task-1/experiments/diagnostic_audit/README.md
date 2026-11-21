# Diagnostyka lokalizacji watermarków

[Pełny raport po polsku](final_research_report.md). [Galeria 510 obrazów OOF](results/examples/index.html).

Wyniki i nowe checkpointy diagnostyczne są odizolowane w `results/`. Skrypty nie otwierają test dataset, nie tworzą submission i nie zapisują do poprzednich eksperymentów. Nie generują syntetycznych watermarków. Geometryczne wzorce w testach służą tylko weryfikacji transformacji.

## Odtworzenie z katalogu repozytorium

Wymagane istniejące dane, wagi, cache frozen_extractor i head_ablation/features128, checkpointy original CV3 oraz stable_selection wszystkich trzech seedów. Środowisko: projektowe Windows CPU venv. Dodatkowe biblioteki: `pandas==2.2.3`, `scikit-learn==1.5.2`; zachowano `numpy==1.26.4`. Pełne wersje z wykonania są w `results/provenance.json`. `pip check` przechodzi.

Uruchamiać kolejno, nie równolegle:

```powershell
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/metrics_audit.py
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/execution_audit.py
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/controls.py
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/receptive_fields.py
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/probes.py
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/probes.py --mode tiny
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/confirm_alignment.py
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/enrich.py
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/report.py
task-1/.venv/Scripts/python.exe task-1/experiments/diagnostic_audit/verify.py
python task-1/dashboard/build.py
```

Sondy i tiny-set wznawiają ukończone wyniki; nie trenują ich ponownie przy obecnych JSON. Pełne powtórzenie treningów należy wykonać w kopii katalogu diagnostyki bez jego `results/`, zachowując położenie względem `task-1/experiments/`. Nie usuwać źródłowych wyników innych eksperymentów. Inference/metric scripts ponownie generują wyłącznie artefakty audytu.

## Artefakty

| Plik | Zakres |
|---|---|
| architecture_audit.md | rzeczywiste klasy, checkpointy, normalizacja, warstwy, stride, RF i tryb eval |
| preprocessing_audit.md | GT, oficjalna metryka, geometria i definicje wszystkich agregacji |
| stn_alignment_audit.md | rzeczywisty sampler, macierze, inverse geometry i wyniki kontroli |
| final_research_report.md / REPORT.md | pełny raport; identyczna treść dla pliku i dashboardu |
| results/per_image_metrics.csv | 1020 rekordów, 510 unikalnych obrazów; metryki OOF, geometria, undefined reasons |
| results/summary_metrics.csv | 6 kohort, macro/pooled, CIs i liczebności |
| results/size_stratified_metrics.csv | pięć binów area oraz osobna kategoria empty |
| results/feature_shapes.csv | native i cached wymiary potwierdzone wykonaniem |
| results/feature_statistics.csv | native statystyki, 6 obrazów/metodę |
| results/cached_feature_statistics.csv | pełne statystyki cache, 170 obrazów/metodę |
| results/feature_probe_results.csv | 28 screeningów, pierwszy fold |
| results/alignment_probe_confirmation.csv | 6 ocen raw/aligned, 3 foldy; cztery dodatkowe treningi outer z dopasowanymi inner |
| results/calibration_results.csv | operational, resubstitution, cross-OOF diagnostyczne i oracle progi |
| results/spatial_controls.csv | stała maska, shuffle, oracle empty/full |
| results/decoder_metrics.csv | raw bits, expected-message confidence i log-likelihood wszystkich obrazów |
| results/diagnostic_results.json | maszynowe podsumowanie i źródła checkpointów |
| results/provenance.json | SHA-256 wag, kodu, masek, splitów, cache; wersje bibliotek |
| results/figures/ | probabilities, area, thresholds, calibration, STN, features i gradient RF |
| results/examples/ | 510 paneli, index.html z filtrem, index.json z kategoriami |
| results/cache/ | wyłącznie nowe instrumented ONNX, diagnostyczne checkpointy i probabilities; niepublikowane przez dashboard |
| results/verification.json | przejście testów metryk, geometrii i spójności artefaktów |

## Ograniczenia metodologiczne

To diagnostyka rozwojowa, nie nowy niezależny benchmark. Original i stable selection są różnymi protokołami. Sondy mają inne loss/capacity niż pełna głowica; porównania raw/aligned i aligned/shuffled sond są dopasowane wewnętrznie. Tiny-set jest jawnie treningowy. Oracle-area, oracle threshold i oracle shift nie są deployable. Globalne optimum progu z tych samych OOF jest optymistyczne; także cross-OOF ma overlap treningowy. Bootstrap po obrazach jest warunkowy na zapisanych modelach.

## Podgląd

Po build: `http://127.0.0.1:8877/reports/diagnostic_audit/REPORT.html` i `http://127.0.0.1:8877/reports/diagnostic_audit/results/examples/index.html`, jeśli lokalny serwer dashboardu działa. Dashboard ma wersje PL i EN, źródłowy raport pozostaje po polsku. Nie publikuje cache ani plików modeli.
