# Następny eksperyment po Etapie 0 — A100

Status: diagnoza wykonana lokalnie, treningi poniżej pozostają propozycją.
Wyniki i ograniczenia: [REPORT.md](REPORT.md). Zacznij od tego raportu i
`../RESEARCH_HANDOFF_2026-11-22.md`. Baza to aktualny `message_localizer`:
PS OOF 0,571916; TM 0,578721; publiczne zgłoszenie #2163 0,6145.

## Pierwsze zadanie: generator v2, jedna porównywalna ablacja

1. Na klastrze odczytaj oryginalne `encoded/{family}/pilot-train/encoded.jsonl`
   oraz odpowiadające pary PNG. Porównaj geometrię i Δ z lokalnym replay oraz
   kontrolami16 coverów. Potwierdź rzeczywiste źródła, kwantyzację i embedding rect.
   Ten audyt archiwalnych realizacji nie został wykonany lokalnie.
2. Przygotuj jawne `embedding_rect`, `final_mask`, `regime`, identyfikatory dawcy
   i odbiorcy, źródłowe hashe, wiadomość, geometrię cropu i politykę krawędzi.
   Przestrzegaj rozłączności źródeł train/dev; źródła locked pozostają zamknięte.
3. Zaimplementuj mieszankę A/B/C/D z instrukcji użytkownika. A: lokalne
   prostokąty, również48/64 px, max aspect16 PS /2 TM. B: transplantacja z innego
   obrazu z zachowaniem geometrii embeddingu. C: czysty obiekt z innego obrazu
   zasłania watermark. D: zachowaj25–35% same-image mixing. Proponowana stała
   mieszanka A/B/C/D =25/25/20/30%; zapisz ją przed oceną.
4. Dobieraj końcowe koszyki po utworzeniu maski i par: około20%≤1%,25%1–5%,
   25%5–20%,30%>20% z dziurami. Clean holes to nakładająca się kohorta;
   jawnie zapisz również udział czystych kontroli. Same-image nie oznacza
   wyłącznie pustych przykładów. Nie przypisuj pustego GT na podstawie Δ=0.
5. Obiektowe fill0,40–0,65 dotyczy przede wszystkim B/C. Zachowaj prostokąty A:
   wszystkie17 najmniejszych realnych PS mają fill1. Generator ma też nadmiar
   komponentów; samo obniżanie fill nie wystarczy. Początkowo użyj twardych
   krawędzi; feather0–2 px jako osobna późniejsza próba, z jawną definicją GT.
6. Sprawdź64 przykłady/metodę przed treningiem: poprawność masek i downsample,
   brak przypadkowego watermarka poza supportem, legalne aspect ratio,
   odmienne source/destination dla B/C, kontrola clean/marked same-content,
   ilościowy rozkład końcowych koszyków. Bez JPEG/blur i nowych transformacji
   geometrycznych. GT obszaru oznakowania wynika z pochodzenia pikseli.

## Budżet i porównanie

- Zachowaj architekturę, wiadomość, oficjalne ekstraktory PS/TM, obie skale,
  pretraining400 updates ×32 widoki,50 zamrożonych +350 joint, FP32, seed42,
  LR2e-5/3e-4 i obecny loss. Zmień tylko generator. Zachowaj2500 źródeł oraz
  politykę ich ponownego użycia; licz osobno wywołania embeddera i widoki.
- Użyj tego samego inicjalnego checkpointu oficjalnego, końcowych wag po stałym
  budżecie, a potem150 updates/fold z24synthetic+8real, LR2e-6/2e-5.
  Te same zapisane3 foldy, próg0,5, brak selekcji na outer-validation.
- Obie rodziny mają identyczny budżet kontroli i wariantu. Użyj accumulation,
  jeśli32 widoki nie mieszczą się na jednej A100; potwierdź równoważną redukcję
  lossu per image i par. Nie zmieniaj precyzji przy okazji generatora.
- Podawaj globalne per-image mIoU oraz FG IoU dla≤1%,1–5%,5–20%,>20%,clean holes,
  i dodatkowo clean IoU dziur. Zapisuj FP32 OOF maps, names, targets, split,
  per-image metryki, aktualizacje, widoki, czas i peak VRAM. Przedziały różnic
  licz parując te same obrazy; opisuj je jako rozwojowe, warunkowe na modelach.
- GPU tylko wewnątrz Slurm. Dotychczasowy zmierzony peak FP32 batch32 na klastrze
  wynosił około24,1GB PS i14,1GB TM. To dane starego przebiegu, nie profil nowego.

## Następne pojedyncze próby, po wyniku generatora

**TM zoom-in:** dokładny GT-bbox dla prostokątów1–5% daje99,05% bit accuracy,
nieregularne48,94%; okna128/192 blisko50%. Najpierw zmierz recall kandydatów
dostępnych bez GT, w tym dla pustych predykcji. Dopiero potem refiner; jawny
koszt inference. GT-bbox jest wyłącznie diagnostycznym sufitem.
Dodatkowy test tolerancji: przesunięcie o 8 px nadal daje 97,37%, a bbox
z idealnego GT128 daje 99,11%. Rozmiar cropu jest bardziej wrażliwy:
margines 4 px daje 98,63%, 8 px daje 72,42%, 16 px około przypadku.
Testuj propozycje o dobrym rozmiarze; nie zakładaj konieczności idealnej pozycji.

**PS residual head Y:** mały head na stride4, Δ z prawdziwej pary clean/marked
i poprawnej geometrii final image. Dla transplantacji czysty counterfactual
to identyczny composite z czystym dawcą, nie samo zdjęcie odbiorcy. Kontrola
zerowego residualu jest konieczna: w embed64 niskieJND daje74,88% zerowych pikseli.
Raportuj błąd na niezerowym Δ oraz realne FG/mIoU. Nie łącz tej zmiany z lossem
segmentacji lub nowym stemem w tym samym porównaniu.

**Loss / SRM / semantyka:** osobne ramiona. Obecna gałąź lokalna już zaczyna się
stride1 i ma wyjście stride4; istnieją skipy piramidy. Oracle postprocessing
daje najwyżej+0,00016/+0,00060 globalnego mIoU PS/TM na sprawdzonej siatce,
więc duży zysk wymaga zmiany wyuczonych cech lub danych, nie samego progu.

## Odtworzenie lokalnej diagnozy

Z katalogu `task-1`, po skopiowaniu pakietu i wyników bieżącego modelu:

```bash
python -m experiments.small_region_diagnosis.oof --require-maps
python -m experiments.small_region_diagnosis.decoder_probe --family pixelseal
python -m experiments.small_region_diagnosis.decoder_probe --family trustmark
python -m experiments.small_region_diagnosis.crop_tolerance --family pixelseal
python -m experiments.small_region_diagnosis.crop_tolerance --family trustmark
python -m experiments.small_region_diagnosis.signal --family pixelseal
python -m experiments.small_region_diagnosis.signal --family trustmark
python -m experiments.small_region_diagnosis.geometry
python -m experiments.small_region_diagnosis.report
python -m experiments.small_region_diagnosis.verify
```

Generator/signal korzystają z istniejącego lokalnego archiwum COCO val2017.
OOF domyślnie czyta `experiments/message_localizer/results/runs`; na klastrze
można podać `--runs /p/project1/training2666/pszenny1/native-localizer/message-runs`.
Nie są potrzebne nowe checkpointy ani inference aktualnego modelu.
