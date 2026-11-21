# ReMark jako sygnał lokalizacji TrustMark

**Zakres:** 170 rzeczywistych obrazów TrustMark; zapisane 3-fold OOF i inner_validation. GPU RTX 3070, trustmark 0.8.0, Q, 100 bitów, bez ECC. Ekstrakcja i usuwanie na GPU, agregacja metryk na CPU. Wagi removera zamrożone. Użyto 51 map/wariantów skal i wygładzenia.

## Wynik porównania

| Metoda | mIoU | FG IoU | Clean IoU | AUC macro | AP macro | Top-K oracle | Internal BF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.4958 | 0.4793 | 0.5123 | 0.6435 | 0.5861 | 0.5612 | 0.0632 |
| B_full_selected | 0.3364 | 0.3075 | 0.3653 | 0.5764 | 0.5514 | 0.5380 | 0.0650 |
| C_local_selected | 0.3677 | 0.3428 | 0.3926 | 0.5922 | 0.5592 | 0.5436 | 0.0716 |
| R5_selected | 0.3207 | 0.2401 | 0.4012 | 0.5965 | 0.5476 | 0.5291 | 0.0670 |
| generic_selected | 0.2818 | 0.3375 | 0.2260 | 0.5615 | 0.5405 | 0.5242 | 0.0911 |

B: pełny residual; C: okna 128/256/384 i średnia skal; R5: RMS kanałów cech; generic: blur/resize/gradient. Mapa i smoothing wybierane wyłącznie na inner_validation maksymalizacją skalibrowanego mIoU. Osobno raportujemy AUC/AP/Top-K każdej mapy do oceny jakości przestrzennej. Wybór najlepszej mapy z tabeli OOF po obejrzeniu wyników byłby dodatkową selekcją rozwojową.

Najlepszy pojedynczy wariant lokalny według OOF AUC, **eksploracyjnie wskazany po wynikach**, to local_mean_s3: AUC 0.6048, AP 0.5579, mIoU 0.3461. Small Top-K lift wynosi 1.484, blisko, lecz poniżej uprzednio ustalonego minimum1,5. Nie traktujemy tej arbitralnej granicy jako dowodu braku jakiejkolwiek informacji; całościowo wyniki AP, masek i kontroli nie uzasadniają dalszego treningu tej receptury.

**Integracja D:** bramka użytecznej lokalizacji NIE została spełniona. Nie trenowano nowej głowicy; rekomendacja: zakończyć tę ścieżkę w przetestowanej postaci.

## Osiem pytań badawczych

1. **Czy zmiany są silniejsze w watermarked?** Dla surowego R1: średnio 0.006008 wewnątrz i 0.005086 poza GT; silniejszy sygnał wewnątrz w 115/170 obrazach. Dla czystych COCO średni residual wynosi 0.002903.
2. **Korelacja z lokalizacją?** Pełny wybierany residual ma AUC 0.5764; cechy R5 0.5965. To sygnał słabszy od baseline AUC 0.6435.
3. **Małe watermarki vs losowy ranking?** W 45 obrazach <=5%: AP full 0.0503, local 0.0506, generic 0.0784. Średnie Top-K/prevalence: full 2.19, local 1.89; referencja losowa =1. Top-K używa prawdziwej powierzchni, więc jest oracle, nie predykcją wdrożeniową.
4. **Czyste otwory?** Przy GT>50% clean IoU: full 0.0592, local 0.0833, baseline 0.1306. FPR w zamkniętych otworach: full 0.6111, local 0.6847. Na tych samych 59 obrazach z otworami średni score full w otworach to 0.009250, w FG 0.008184. Obszary clean połączone z ramką nie są nazywane otworami.
5. **Lokalne przetwarzanie?** AUC full 0.5764 → local 0.5922; AP 0.5514 → 0.5592. Wynik dla małych obszarów podano osobno; oracle cropy nie są używane do oceny metody wdrożeniowej.
6. **Komplementarność?** Prosta suma logiczna masek baseline OR residual daje mIoU full 0.4328, local 0.4439, wobec baseline 0.4958. To diagnostyka ustalonej reguły, nie nauczona integracja. Zapisano dodatkowe TP i FP per obraz; różne błędy nie wystarczają jako dowód użytecznej komplementarności.
7. **Specyficzność watermarku?** Kontrola matched-content wykazuje średni wzrost residualu wewnątrz minus poza lokalnym watermarkiem 0.000356. Usuwanie działa także na czystych obrazach; średni Spearman residual–tekstura -0.096, residual–krawędzie -0.145. Znak tych korelacji nie jest założony z góry. Porównanie z generic i parami nie uzasadnia traktowania całego residualu jako watermarku.
8. **Czy integrować?** Nie w tej postaci: residual nie przechodzi ustalonej bramki jakości, a koszty lokalnego przetwarzania są znaczne.

## Powierzchnia i czyste otwory

| Metoda | GT area | n | AP | Top-K | Clean IoU | Hole FPR | Obrazy z dziurami |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | le1pct | 9 | 0.0072 | 0.0000 | 0.9793 | nan | 0 |
| baseline | 1to5pct | 36 | 0.0708 | 0.0582 | 0.9554 | 0.0000 | 3 |
| baseline | 5to20pct | 25 | 0.2622 | 0.2390 | 0.8994 | 0.0000 | 2 |
| baseline | gt50pct | 85 | 0.9634 | 0.9350 | 0.1306 | 0.8945 | 59 |
| B_full_selected | le1pct | 9 | 0.0218 | 0.0212 | 0.7099 | nan | 0 |
| B_full_selected | 1to5pct | 36 | 0.0574 | 0.0429 | 0.7069 | 0.3333 | 3 |
| B_full_selected | 5to20pct | 25 | 0.1955 | 0.1865 | 0.7147 | 0.0000 | 2 |
| B_full_selected | gt50pct | 85 | 0.9258 | 0.9130 | 0.0592 | 0.6111 | 59 |
| C_local_selected | le1pct | 9 | 0.0155 | 0.0156 | 0.8098 | nan | 0 |
| C_local_selected | 1to5pct | 36 | 0.0594 | 0.0429 | 0.7105 | 0.6667 | 3 |
| C_local_selected | 5to20pct | 25 | 0.2105 | 0.1951 | 0.7277 | 0.0000 | 2 |
| C_local_selected | gt50pct | 85 | 0.9323 | 0.9148 | 0.0833 | 0.6847 | 59 |
| R5_selected | le1pct | 9 | 0.0143 | 0.0000 | 0.7501 | nan | 0 |
| R5_selected | 1to5pct | 36 | 0.0501 | 0.0263 | 0.7213 | 0.5349 | 3 |
| R5_selected | 5to20pct | 25 | 0.1774 | 0.1628 | 0.7167 | 0.0526 | 2 |
| R5_selected | gt50pct | 85 | 0.9282 | 0.9127 | 0.0992 | 0.4762 | 59 |
| generic_selected | le1pct | 9 | 0.0209 | 0.0113 | 0.3320 | nan | 0 |
| generic_selected | 1to5pct | 36 | 0.0928 | 0.0759 | 0.3718 | 0.9612 | 3 |
| generic_selected | 5to20pct | 25 | 0.1839 | 0.1594 | 0.4233 | 1.0000 | 2 |
| generic_selected | gt50pct | 85 | 0.9128 | 0.9066 | 0.0711 | 0.6585 | 59 |

Hole FPR macro po obrazach z co najmniej jednym zamkniętym otworem; pooled FPR i średni score w otworach są w CSV. NaN oznacza brak takich otworów. BF1 wyłącza ramkę obrazu i toleruje 2 komórki siatki128.

## Diagnostyka oracle i ramki obrazu

| Metoda | Kalibrowane mIoU | Oracle próg per obraz — niedeployable | AUC bez 16px brzegu | AP bez brzegu |
|---|---:|---:|---:|---:|
| baseline | 0.4958 | 0.5288 | 0.6533 | 0.5944 |
| B_full_selected | 0.3364 | 0.4863 | 0.5698 | 0.5525 |
| C_local_selected | 0.3677 | 0.4983 | 0.5851 | 0.5605 |
| R5_selected | 0.3207 | 0.4922 | 0.5884 | 0.5513 |
| generic_selected | 0.2818 | 0.4852 | 0.5590 | 0.5439 |

Oracle wybiera próg z 101 kwantyli score plus pełna/pusta maska, osobno na każdym ocenianym obrazie z użyciem jego GT. To diagnostyka, a nie dozwolona procedura inferencji. AUC bez brzegu dotyczy tylko obrazów z obiema klasami w pozostawionym wnętrzu; undefined są pomijane, nie zastępowane zerem.

## Usuwanie wiadomości i geometria

Globalna trafność surowych bitów na rzeczywistych obrazach: 0.7534 → 0.6385. Co najmniej 10/100 bitów zmieniło się na 146/170 obrazach. Wśród 87 z początkową trafnością >=0,75 spadek >=0,10 wystąpił na 56. Dla małych watermarków globalny odczyt może być losowy już przed usuwaniem.

API daje RGB512→RGB512 uint8. Sieć dostaje PIL bilinear256, tensor[-1,1]; zwraca poprawkę dodawaną do oryginalnego obrazu, z clip i feather1%. Wersja0.8.0 nie przyjmuje loadRemover — zawsze ładuje remover. Batch ścieżka lokalna różni się od pojedynczego API o maks. 1 poziom uint8 w 24 kontrolach; zapisano to jako efekt numeryczny GPU. Maks. przesunięcie fazowe na czystych kontrolach to 0.0129px; API nie ma transformacji przestrzennej poza resize poprawki.

Osobno oceniono AUC/AP po wycięciu 16px ramki. Kontrola resize512→256→512 mierzy odpowiedzi przy krawędziach, a float residual oddziela kwantyzację. Nie interpretujemy oracle cropów jako metody lokalizacji.

## Wizualizacje i artefakty

[Galeria 27 rzeczywistych obrazów — osiem paneli na obraz](results/examples/index.html). Dobór obejmuje wszystkie przedziały powierzchni, closed holes, porażki baseline i najlepsze rankingi residualu. Czerwony: FP, niebieski: FN; różnica RGB ×20 względem szarości0,5.

![Rozkłady residualu](results/trustmark_residual_examples.png)

[Per-image CSV](results/per_image_metrics.csv), [metryki według wielkości](results/summary_by_area.csv), [kalibracja](results/calibration.json), [model/provenance](results/provenance.json), [kontrole matched](results/controls/matched_pairs.csv), [bramka integracji](results/integration_gate.json), [weryfikacja](results/verification.json). Mapy i rekonstrukcje są w results/full, results/local, results/recovered. Hashe checkpointów i źródeł zachowano; brak konkursowego testu i submission.

## Odtworzenie

Środowisko CUDA `.venv-rgb-cuda`; uruchom z katalogu repo:
```powershell
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/extract.py --mode full
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/controls.py
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/extract.py --mode local
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/evaluate.py --include-local
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/report.py
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/remark_localization/verify.py
task-1/.venv/Scripts/python.exe task-1/dashboard/build.py
```

Badanie rozwojowe na znanych foldach; 170 unikalnych obrazów. Wielość map zwiększa ryzyko wyboru po wynikach. Oracle thresholds per-image, kwantyle101 oraz Top-K są jawnie niedeployable. Wyniki bez uczonej głowicy nie wykluczają wszystkich możliwych sposobów użycia ReMark, lecz wystarczają do decyzji o tej recepturze.

Źródła pierwotne: [Adobe TrustMark](https://github.com/adobe/trustmark), [publikacja ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html). Szczegóły implementacji i cytowania: [RESEARCH.md](RESEARCH.md).
