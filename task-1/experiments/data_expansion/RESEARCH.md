# Rozszerzenie danych: nowe fotografie i realistyczne czyste dziury

Stan źródeł: 22 listopada 2026. To badanie źródeł i propozycja eksperymentu, nie wynik treningu. W ramach tej analizy nie uruchomiono embeddowania, inferencji, treningu ani submission. Pobierano dokumentację, metadane i kilka podglądów obrazów; nie cały korpus. Osobne działania głównego agenta mogą być opisane w sąsiednich plikach.

## Rekomendacja

**Najbardziej użyteczne nowe źródło to `paint-by-inpaint/PIPE` razem z `paint-by-inpaint/PIPE_Masks`: najpierw 2–5 tys. różnych fotografii, z wyraźnym udziałem Open Images, potem kontrolowane porównanie tej samej maski z przywróceniem oryginalnych pikseli i z rzeczywistym inpaintingiem.** Daje to zarówno nowe treści, jak i czyste dziury po usunięciu obiektów. Nie jest gotowym zbiorem watermark-localization: dokładne etykiety trzeba wytworzyć przez własne operacje na pikselach po użyciu zamrożonych oficjalnych embedderów.

Więcej danych może poprawić generalizację, lecz obecne wyniki nie dowodzą, że ograniczeniem jest wyłącznie liczba zdjęć. `region_data` już dodało maski obiektów COCO, clean-object addition i marked-object transplantation. Po dopasowanej adaptacji PixelSeal zmienił się o −0.00017 mIoU, TrustMark o +0.00573; tiny PixelSeal nadal miał FG IoU 0. Zmiana samego kształtu maski nie była przełomem. Nowy test powinien izolować **nowe źródła** oraz **treść faktycznej edycji**. [Lokalny raport](../region_data/REPORT.md), [kod generatora](../region_data/data.py).

Aktualny punkt odniesienia to zaakceptowane #3188 / public 0.6736. Jego PixelSeal fold 1 0.581724 jest innym protokołem niż 170-obrazowe OOF. Istniejące 10 tys. kompozycji na rodzinę korzystały z około 2474 fotografii, nie z 10 tys. niezależnych źródeł. Żaden wynik w tej notatce nie zastępuje tej bazy ani nie przewiduje przyrostu leaderboardu. [Bieżący handoff](../RESEARCH_HANDOFF_2026-11-22.md).

## Co wiadomo o zgodności z zadaniem

Lokalny [Watermark Localization.pdf](../../../Watermark%20Localization.pdf) definiuje wejście RGB 512×512, wynik 128×128, trzy konkretne metody i cztery operacje: outpainting, removal/alteration, addition oraz transplantation. Osadzenie zachodzi w prostokącie; maksymalny aspect ratio to 16:1 dla PixelSeal/StegaStamp i 2:1 dla TrustMark. Etykieta oznacza pochodzenie watermarked pixels, nie maskę rozpoznanego obiektu. Oficjalne zmniejszanie maski stosuje co najmniej 50% dodatnich pikseli w komórce 4×4.

PDF **nie identyfikuje bazowego zbioru fotografii ani modelu/generatora edycji**. Nie ma podstaw, aby nazywać COCO, Stable Diffusion, LaMa lub DALL-E dokładnym generatorem konkursu. Są to kandydaci do kontrolowanego przybliżenia rozkładu.

Adres z PDF, `SprintML/Hackathon_Watermark_Localisation_Task_Paris`, w aktualnym [oficjalnym API HF](https://huggingface.co/api/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris?blobs=true) rozwiązuje się do `SprintML/Hackathon_Watermark_Localisation_Task`, revision `372b5ee8b1df43765096f027467037e3cf3c1023`. Sprawdzono obie nazwy: identyczny ID, revision i manifest sześciu plików; zbiór publiczny, bez gate. Wydanie zawiera README, requirements, starter kit, train.zip i test.zip, bez generatora ani czystych oryginałów. `train.zip` ma 190830424 B i SHA-256 `32ea0beaa173d2a8234574b893cab053cbad960a7b22d26b85e2b16ee782f783`. To weryfikacja metadanych Hub, nie ponowne pobranie archiwum. [Kanoniczne źródło](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task).

## Publiczne źródła — co rzeczywiście dostarczają

| Źródło | Dane, rozmiar i dostęp | Zgodność oraz ograniczenie |
|---|---|---|
| **PIPE + PIPE_Masks** | 888230 par train, 752 test; pary 512×512; oba repo publiczne, CC-BY-4.0. Cały PIPE około 80.46 GB. Maski mają osobne shardy, około 1.85 GB obecnych plików. | Oryginał z obiektem oraz inpainting po usunięciu; COCO/Open Images, anotacje LVIS; klucze źródła i obiektu. Najlepszy kandydat na nowe zdjęcia i realistyczne clean holes. To liczba par, nie zweryfikowana liczba unikalnych zdjęć. [PIPE](https://huggingface.co/datasets/paint-by-inpaint/PIPE), [maski](https://huggingface.co/datasets/paint-by-inpaint/PIPE_Masks). |
| **COCO train2017** | Oficjalnie 118 tys. zdjęć / około 18 GB, anotacje train+val około 241 MB. Pojedyncze JPEG można pobierać z `coco_url` w JSON, bez całego ZIP. | Najprostsza kontrola wpływu większej liczby źródeł przy starym generatorze. Nie daje gotowych edycji. Rozdzielczości są różne; przygotować clean 512 przed embedderem. Anotacje CC-BY-4.0, fotografie mają własne prawa/licencje — nie wszystkie są automatycznie CC-BY-4.0. [Download](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm), [terms](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/termsofuse.htm). |
| **Open Images V7** | Train: 944037 zdjęć z 2686666 maskami instancji, 350 klas. CVDF oferuje zdjęcia do 1024 px na dłuższym boku, train około 513 GB; oficjalny downloader pobiera listę ID. | Dużo nowych fotografii poza COCO, dobre źródło obiektów/donorów. Nie dostarcza edited pairs. Maski train powstawały interaktywnie; pomijano bardzo małe/niejednoznaczne obiekty. Adnotacje CC-BY-4.0, zdjęcia wymieniane jako CC-BY-2.0; zachować per-image attribution. [Opis/licencje](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), [subset download](https://storage.googleapis.com/openimages/web/download_v7.html), [CVDF rozmiary](https://github.com/cvdfoundation/open-images-dataset). |
| **TGIF / TGIF2** | Łącznie 271788 edytowanych wariantów, ale tylko 3124 oryginały, w tym 2440 train. Oryginały, maski i podzbiory edycji dostępne z serwera autorów; rozmiary do 1024×1024; CC-BY-SA-4.0. | Dobre źródło różnorodnych clean edits: SD2, SDXL, Firefly i FLUX. Preferować warianty `*-sp` oraz właściwe maski splicingu; `*-fr` regenerują także tło. Mała liczba niezależnych źródeł ogranicza wartość jako ekspansji fotografii. Nie zweryfikowano pełnej wielkości transferu. [Repo i download](https://github.com/IDLabMedia/tgif-dataset). |
| **MagicBrush** | Train 8807 tur / 4512 sesji; cały zbiór około 30.2 GB, train 25.61 GB w 51 shardach ~0.5 GB. Źródło, target i maska; target/maska zwykle 1024 px. HF publiczny, CC-BY-4.0. | Ręcznie przygotowywane instrukcje i różne zmiany, także removal/addition. `img_id` jest COCO ID; kolejne tury są zależne, więc grupować całą sesję. Nie traktować każdej tury jako nowej fotografii. Free-form edit mask nie jest dowodem dokładnego zasięgu zmian pikseli. [Oficjalna karta](https://huggingface.co/datasets/osunlp/MagicBrush), [projekt](https://osu-nlp-group.github.io/MagicBrush/). |

PIPE usuwa zbyt małe/duże obiekty i obiekty przy krawędziach, a następnie używa dilated masks i alpha blending. Jest więc lepszym kandydatem dla realistycznych dziur niż bezpośrednim rozwiązaniem tiny PixelSeal. Zachować dotychczasową małą geometrię i testować ją osobno; nie skalować już oznakowanych obiektów, aby sztucznie wytworzyć tiny cases. [Publikacja autorów, sekcja 3.1](https://arxiv.org/html/2404.18212v3).

**Outpainting:** najprostsze dokładnie etykietowane przybliżenie korzysta z nowego zdjęcia 512: watermark w prostokątnym cropie osadzonym później w nieoznaczonym otoczeniu. Symuluje geometrię, ale nie teksturę wygenerowanego otoczenia. GenCrop publikuje prawdziwy pipeline outpaintingu, lecz jego zdjęcia Unsplash wymagają uzyskania dostępu; to nie natychmiastowy gotowy bank par do pobrania. Nie ustalono publicznego zbioru jednocześnie zapewniającego dokładne pochodzenie zdjęć, gotowe outpainted pairs, maski i zgodność z ukrytym konkursem. [GenCrop](https://github.com/jhong93/gencrop).

**Transplantation:** nowe COCO/Open Images maski i donory wystarczą do dokładnej konstrukcji, ale sama ta operacja już istnieje w `region_data`. DEFACTO ma około 105 tys. splicing examples z maskami dawcy i odbiorcy oraz JSON graph donor identities, a także około 25 tys. removals; dostęp wymaga formularza. Na stronie nie potwierdzono otwartej licencji ani transferu bez oczekiwania, dlatego nie jest pierwszym wyborem. [Oficjalne DEFACTO](https://defactodataset.github.io/).

## PIPE: konkretny mały transfer i poprawne łączenie

Poniższe rozmiary i SHA-256 sprawdzono w oficjalnym API Hub; sumarycznie **803260478 bajtów**, czyli około 803 MB / 766 MiB. Nie pobierać wildcardu `data/train-*`, jeżeli celem jest mały debug subset.

| Repo / revision | Jeden plik train | Bajty / SHA-256 |
|---|---|---|
| `paint-by-inpaint/PIPE` / `54ce568a33948642fcfa7d2bac986a147952520a` | `data/train-00000-of-00162-4e513d2170c191e2.parquet` | 495913730 / `4230aec0e85f7128dec11e3ed880259179330dfaa48ed8d5b28473da6f2f1b12` |
| `paint-by-inpaint/PIPE_Masks` / `c69c460aed810376b1361b0b2ab949ee146689d7` | `data/train-00000-of-00006-8f1a1413a245c0d2.parquet` | 307346748 / `08455d992aae7b467149d097f604ea02c7659b7e60389d6b6c6032b400905798` |

Bezpośrednie, przypięte adresy: [shard obrazów](https://huggingface.co/datasets/paint-by-inpaint/PIPE/resolve/54ce568a33948642fcfa7d2bac986a147952520a/data/train-00000-of-00162-4e513d2170c191e2.parquet), [shard masek](https://huggingface.co/datasets/paint-by-inpaint/PIPE_Masks/resolve/c69c460aed810376b1361b0b2ab949ee146689d7/data/train-00000-of-00006-8f1a1413a245c0d2.parquet). Metadane źródłowe: [PIPE API](https://huggingface.co/api/datasets/paint-by-inpaint/PIPE?blobs=true), [PIPE_Masks API](https://huggingface.co/api/datasets/paint-by-inpaint/PIPE_Masks?blobs=true). Stara wartość `download_size` w karcie masek nie odpowiada sumie obecnych shardów; powyżej użyto aktualnych wielkości plików, bez dwóch niezwiązanych plików MP4 w repo masek.

Semantyka kolumn jest odwrotna do intuicyjnego „source=oryginał”:

- `target_img`: naturalny oryginał **z obiektem** — to clean cover do embeddowania.
- `source_img`: obraz **po usunięciu obiektu**, używany jako źródło przez model dodający obiekty — to czyste piksele edycji.
- `mask` w osobnym repo: maska użyta do utworzenia inpaintingu, **nie gotowa maska watermarked/clean**.
- Kandydat klucza join edycji: `(target_img_dataset, img_id, ann_id)`. Nie zakładać unikalności: w pobranym shardzie masek audyt wykrył duplikaty i zatrzymał początkowy strict-join. Grupować wystąpienia; wolno zwinąć maski identyczne bajtowo, konflikty odrzucać i raportować. Podział źródeł: `(target_img_dataset, img_id)` bez `ann_id`.

Te pola zachowują COCO numeric ID i Open Images hexadecimal ID. LVIS jest dodatkowym źródłem anotacji nad fotografiami COCO, więc nie tworzy niezależnego namespace fotografii. Oba repo wystawiają train i test, bez val; własny dev należy wydzielić grupami z train. Brak lokalnie ukończonego audytu wszystkich duplikatów train/test — nie zakładać rozłączności po wierszach. [Karta PIPE](https://huggingface.co/datasets/paint-by-inpaint/PIPE/blob/54ce568a33948642fcfa7d2bac986a147952520a/README.md), [karta masek](https://huggingface.co/datasets/paint-by-inpaint/PIPE_Masks/blob/c69c460aed810376b1361b0b2ab949ee146689d7/README.md).

### Zakres wykonanej weryfikacji loadera

Uruchomiono stdlib HTTP loader Dataset Viewer na pierwszych dwóch wierszach train obu repo. Join zgadzał się dla `COCO/339734/200890` oraz `COCO/537379/211233`. RGB i maski miały 512×512. Odczyt metadanych 1000 wierszy zakończył się HTTP 502, więc nie raportujemy liczby unikalnych źródeł z tej próby. Sprawdzony minimalny odczyt, **wyłącznie do inspekcji, nie do tworzenia danych treningowych**:

```python
import json
from urllib.request import urlopen

def inspect_rows(repo):
    url = ("https://datasets-server.huggingface.co/rows?dataset=" + repo
           + "&config=default&split=train&offset=0&length=2")
    with urlopen(url, timeout=30) as response:
        return [item["row"] for item in json.load(response)["rows"]]

images = inspect_rows("paint-by-inpaint/PIPE")
masks = inspect_rows("paint-by-inpaint/PIPE_Masks")
key_fields = ("target_img_dataset", "img_id", "ann_id")
key = lambda row: tuple(row[name] for name in key_fields)
mask_by_key = {key(row): row for row in masks}
for row in images:
    assert key(row) in mask_by_key
    print(key(row), row["source_img"]["width"], row["target_img"]["width"])
```

Podglądy RGB są JPEG. W podglądach maski miały wartości 0–255, a poza `mask>0` RGB różniły się odpowiednio w 29388 i 38889 pikselach. Tych wyników nie przenoszono na surowy Parquet: warstwa podglądu może zmieniać reprezentację. Nie należy pobierać viewer image URLs jako wzorcowych pikseli. Publiczny viewer pokazuje też zmienne rozmiary masek; pierwsze dwa przykłady nie dowodzą, że każda maska ma 512×512.

### Ukończony audyt surowego sharda na JURECA

Główny agent pobrał oba przypięte shardy, potwierdził SHA-256 i uruchomił [audit_pipe.py](audit_pipe.py) w izolowanym środowisku z pyarrow 19.0.1, NumPy 1.26.4, Pillow 11.3.0 i SciPy 1.15.3. Sprawdzony odczyt używa oryginalnych embedded `bytes`, dołącza maski po kluczach i grupuje ich duplikaty; nie opiera się na numerowaniu shardów ani podglądach HF. Przeczytano i podsumowano zapisany [raw_audit.json](results/raw_audit.json):

| Pomiar | Wynik |
|---|---:|
| Wiersze obrazów / masek | 5483 / 148039 |
| Różne source keys w shardzie obrazów | 5460 |
| Źródła COCO / Open Images | 1414 / 4046 |
| Wiersze obrazów z kluczem obecnym w shardzie masek | 5483 |
| Zduplikowane klucze w całym shardzie masek | 24 |
| Źródła już przypisane do train / dev / locked COCO | 607 / 15 / 31 |
| Źródła poza dotychczasowym assignment | 4807 |
| Pary objęte inspekcją raw bytes | 64 |

To **5460 różnych zdjęć**, a nie 5483 niezależne fotografie; pełnych 888230 par nadal nie policzono pod kątem unique sources. Powyższy overlap dotyczy istniejących przypisań COCO, nie dowodzi braku cross-dataset near-duplicates ani zgodności z konkursowym train.

We wszystkich 64 sprawdzonych parach RGB były JPEG 512×512 i nie wystąpił błąd join/conflicting masks. W 23 przypadkach maski też były JPEG 512×512; w pozostałych 41 były PNG o rozmiarach natywnych. Drift poza `mask>0` występował we wszystkich 23 parach o zgodnych wymiarach, od 10615 do 42575 pikseli. W dwóch pierwszych maski miały 222 i 233 różne wartości, a poza `mask>0` zmieniało się 28967 i 38842 pikseli. **Drift poza maską jest potwierdzony także na raw bytes**, nie tylko na podglądzie. Zapisana maska nie jest binarną mapą wszystkich zmian. Etykietę watermarka należy zdefiniować przez własny hard overwrite, np. jawnie deklarując `M = mask >= 128`, po sprawdzeniu wyrównania. Nie należy uznawać wszystkich różniących się pikseli za edycję.

**Wyrównanie native Open Images masks jest potwierdzone przez autora.** W [dyskusji PIPE_Masks #4](https://huggingface.co/datasets/paint-by-inpaint/PIPE_Masks/discussions/4) Navve Wasserman (`navvew`, członek organizacji autorów) podał 10 października 2025, że wszystkie obrazy zmniejszano przez `torchvision.transforms.Resize((512, 512))` i tę samą operację należy zastosować do masek. Oznacza to resize całego obrazu do kwadratu, **bez center crop**. Repo GitHub zawiera loader, lecz nie opublikowany generator; źródłem tego ustalenia jest odpowiedź autora, nie domysł z innego projektu. Domyślną interpolacją `Resize` jest bilinear; zastosować ją jawnie do maski, a dopiero potem ustalony próg128. To nasza deterministyczna binarna definicja support, nie twierdzenie o pierwotnej binarności maski. [Dokumentacja Resize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html).

Niezależna mała kontrola porównała fotografie z oficjalnego mirroru Open Images z podglądem PIPE dla `bcabc07fb161267a`, `96a31c58ef46de06`, `24463de8dd7e6225`: stretch512 z bicubic miał MSE 61.86/69.37/26.40, a center crop 7651.65/3052.95/2462.73. Potwierdza to geometrię stretch na tych trzech fotografiach; podgląd JPEG i mirror nie służą do wnioskowania o identyczności bajtowej ani dokładnym historycznym resamplerze. [Oficjalny mirror](https://github.com/cvdfoundation/open-images-dataset).

Reprodukcja wykonanego małego odczytu/audytu (CPU, bez embeddera i bez danych testowych konkursu), z katalogu `task-1` na JURECA:

```bash
/p/project1/training2666/pszenny1/native-localizer/data-expansion/.venv-audit/bin/python \
  experiments/data_expansion/audit_pipe.py \
  --root /p/project1/training2666/pszenny1/native-localizer --inspect 64
```

Na moment aktualizacji główny agent przygotowuje smoke subset 512 train / 64 dev / 64 locked wyłącznie z nowych source groups, z canonical 512 alignment przed embedderem. To etap weryfikacji danych, odrębny od proponowanego poniżej większego eksperymentu; nie ma jeszcze wyniku uczenia na PIPE.

## Jak utworzyć prawidłowe watermarked examples

To projekt własnego generatora, nie twierdzenie o ukrytym generatorze konkursowym. Oznaczmy: `C` — clean original 512, `E` — dopasowana edycja 512 bez watermarka, `R` — prostokąt osadzenia, `W` — dokładny uint8 wynik oficjalnego embeddera na `R`, `M` — zadeklarowany binarny obszar nadpisania.

1. **Wszystkie zmiany rozmiaru/cropy i wyrównanie pary wykonać przed embedderem.** Zachować oficjalny PixelSeal raw256, TrustMark Q raw100 i StegaStamp bez zmiany siły, checkpointów, bit order ani wrappera. Nie wykonywać resize/JPEG/blur na `W`.
2. Removal/alteration: osadzić `C`, następnie `X = W.copy(); X[M] = E[M]`. Ground truth 512 to `R & ~M`. Wymusić dokładne `X[~M] == W[~M]`, nie „prawie identyczne”. Jeśli maska źródła jest miękka lub w innym rozmiarze, najpierw ustalić i zapisać deterministyczną regułę binarnej maski oraz poprawne mapowanie; odrzucać brak alignment. Nie wyprowadzać watermark GT z `E != C`.
3. Addition: można odwrócić parę PIPE — embedder otrzymuje clean obraz bez obiektu, a oryginalny obiekt z nieoznaczonego `target_img` nadpisuje `M`. To osobny reżim, ponieważ clean cover zawiera wcześniej wygenerowane tło; nie mieszać tego bez oznaczenia z removal na naturalnym oryginale.
4. Transplantation: najpierw embedder na prostokącie zdjęcia dawcy, później kopiowanie oznakowanego fragmentu na clean background. GT wynika z przecięcia maski obiektu z zakresem osadzenia dawcy. Dopuszczalna dokładna translacja całkowitopikselowa; skala/rotacja ewentualnie przed embedderem. Dawca i tło muszą należeć do tej samej source partition.
5. Outpainting: embedder na oryginalnym prostokątnym cropie, potem umieszczenie go bez interpolacji we wcześniej przygotowanym nieoznaczonym 512 canvas. GT to wklejony prostokąt. Generowane otoczenie musi zostać ograniczone do obszaru poza nim.
6. Zawsze liczyć końcową maskę 128 przez `wmloc.utils.downsample_mask`. Etykieta wynika z operacji kopiowania pikseli nawet tam, gdzie watermark po kwantyzacji przypadkiem daje zerowy residual. Sprawdzić PNG roundtrip, oba bieguny maski, kształty, histogram udziału clean/marked i tolerancję 0 pikseli poza support.

Hard overwrite w punkcie 2 tworzy dokładną etykietę naszego przykładu także wtedy, gdy gotowy `E` ma inne piksele poza maską: te piksele nigdy nie trafiają do `X`. Jeśli maska ucina realistyczną granicę edycji, próbkę odrzucić albo zmienić jawnie zdefiniowany support przed kompozycją; nie maskować tego miękkim blendowaniem watermarka. Original i edited należy obejrzeć wraz z granicą na małej galerii przed skalowaniem.

W provenance zapisać repo/revision, shard SHA-256, source key, ann_id, źródłowy i końcowy mask hash, transformację do 512, rzeczywisty support 512/128, message, rectangle, wrapper/model hashes, liczbę i reżim wywołań embeddera. Kolumny źródłowe i instrukcja nie są wejściem modelu.

## Najbliższy kontrolowany eksperyment: 2500 różnych źródeł PIPE

**Etap przygotowania CPU, bez uczenia:** z pierwszego sharda zebrać do 2500 unikalnych source keys, preferując znaczący udział Open Images; zachować liczby per-source, a nie obiecywać dokładnej końcowej liczebności przed filtrowaniem. Cel: 2000 train, 250 dev, 250 locked. Audyt znalazł 4807 source keys poza istniejącym assignment, więc ten shard ma wystarczającą pulę kandydatów; ostateczny yield zależy od jakości masek i deduplikacji. Jeśli po odrzuceniach nie wystarczy, jawnie raportować mniejszy zbiór lub dopiero wtedy pobrać następny shard obrazów. Ta sama para shardów waży 803 MB.

Przed podziałem odrzucić wszystkie źródła z istniejącego replay oraz z już ustalonych dev/locked COCO. Znormalizować `COCO` i LVIS do COCO ID. Wszystkie edycje, cropy, różne anotacje obiektów oraz rodziny watermarków danego zdjęcia muszą zostać razem. Powiązać near-duplicate zdjęcia także pomiędzy COCO/Open Images; użyć hash pliku i zdekodowanych pikseli, pHash oraz crop-aware candidate screen z kontrolą niepewnych par. Obecny `prepare_sources.py` używa SHA i pHash≤4 i sam zaznacza, że nie jest to pełne wykrywanie transformowanych duplikatów. Sprawdzić wyłącznie 510 konkursowych train, bez otwierania konkursowych test pixels. [Istniejący split/screen](../native_localizer/prepare_sources.py).

**Najmniejszy test, który odpowiada na przyczynę: trzy ramiona, jeden z góry ustalony budżet.**

| Ramię | Ten sam udział starego replay | Nowe przykłady | Co mierzy porównanie |
|---|---|---|---|
| A | 100% dotychczasowego replay | brak | Kontynuacja kontrolna przy identycznej liczbie aktualizacji. |
| B | 50% dotychczasowego replay | 50% nowe PIPE originals + PIPE supports, z przywróceniem `C[M]` | B−A mierzy łącznie nowe źródła i geometrię; **nie izoluje** ich osobno. |
| C | jak B | Te same source/message/rectangle/maski/cohorty co B, ale `E[M]` zamiast `C[M]` dla clean holes | C−B izoluje wartość realistycznej treści edycji. |

Nie zmieniać architektury, lossu, batcha, LR, schedulerów, progu, share real/synthetic ani geometrii B/C. Zachować dotychczasowe small-positive i small-hole replay; nowe PIPE nie muszą mieć dość przykładów ≤1%, żeby je zastąpić. Proponowany krótki budżet to 200 synthetic updates, a następnie 150 kroków adaptacji per fold, dopasowane do protokołu wcześniejszego `region_data`; checkpoint i konfigurację zamrozić przed oglądaniem wyników. Nie startować fold-validation z checkpointu refitowanego na wszystkich 170 real labels. Użyć wspólnego synthetic-only checkpointu lub właściwych niezanieczyszczonych checkpointów per-fold. Dopiero wygrane ramię przenieść do protokołu aktualnej najlepszej architektury.

Jeżeli celem jest **wyłącznie** efekt nowych fotografii, dodać odrębne ramię `new originals + identyczny stary generator` albo wykonać je jako wcześniejszy test. Nie przedstawiać B−A jako czystego efektu liczby zdjęć.

Ocena: te same trzy istniejące foldy i te same real-image draws, oficjalny mIoU, FG IoU≤1% /1–5%, clean IoU dla hole≤1% /1–5%, liczba kompletnych pominięć, per-image delta oraz bootstrap warunkowy po obrazach. Są to wielokrotnie używane development folds, nie niezależny test. Przed startem przykładowa bramka: wzrost mIoU≥0.005, dodatni przedział paired delta i brak regresji>0.01 w tiny-FG/clean-hole grupach; małe n raportować jawnie. Zmiana progu/oracle/hysteresis należy do osobnej osi eksperymentu, aby przyrostu nie przypisać błędnie danym.

Zachować ten sam łączny budżet optymalizacji między ramionami i raportować: unique photos, unique object edits, exact embed calls oraz generated views oddzielnie. Nie trenować na konkursowym teście, nie dobierać mieszanki na public leaderboardzie. Jeśli poprawia się tylko synthetic dev, a real OOF stoi lub spada, zakończyć skalowanie i zbadać mismatch maski, tekstury i reżimu osadzenia.

## Niewiadome, które decydują o dalszym kroku

- Nieznane pozostają oficjalne photo source i editor; zgodność PIPE jest hipotezą opartą na typie operacji.
- Raw-parquet audit ukończono dla metadanych całego pierwszego sharda i 64 par pikselowych. Przed treningiem pozostaje walidacja wszystkich wybranych par, obejrzenie wyrównanych masek oraz deduplikacja nowych source groups; liczba 4807 jest pulą kandydatów, nie obietnicą finalnego yield.
- PIPE może poprawić clean holes, lecz filtruje małe obiekty. Dla tiny PixelSeal nadal trzeba rozdzielić obecność sygnału, skalę osadzenia, ranking i próg; więcej ogólnych par nie gwarantuje poprawy.
- Największą nowością względem zakończonego `region_data` są nowe fotografie Open Images i naturalne↔inpainted pairs, a nie ponowne użycie maski COCO pod inną nazwą.
