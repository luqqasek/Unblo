"""Build the Polish stage-0 report directly from inspectable measured artifacts."""
import numpy as np
from PIL import Image, ImageDraw

from .common import *

LABELS = {"le1": "≤1%", "1to5": "1–5%", "5to20": "5–20%", "gt20": ">20%", "clean_holes": "clean holes", "all": "wszystkie"}


def fmt(value, n=4):
    return "—" if value is None else f"{value:.{n}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in rows])+"\n"


def figures(oof):
    dest = OUT/"gallery"; dest.mkdir(exist_ok=True)
    meta = metadata(); decoder = {f: read(OUT/"decoder"/f/"per_image.json") for f in FAMILIES}
    for family in FAMILIES:
        candidates = [r for r in decoder[family] if r["view"] == "bbox256"]
        chosen = []
        for group in ("le1", "1to5"):
            rr = sorted([r for r in candidates if r["bucket"] == group], key=lambda r: r["bit_accuracy"])
            chosen.extend((rr[0], rr[-1]))
        maps = {}
        for fold in (1, 2, 3):
            with np.load(RUNS/family/f"matched-v1-adapt/fold{fold}/maps.npz") as z:
                maps.update({str(n): (p.squeeze(), y.squeeze().astype(bool)) for n, p, y in zip(z["names"], z["probabilities"], z["targets"])})
        canvas = Image.new("RGB", (6*192, len(chosen)*222), "white"); draw = ImageDraw.Draw(canvas)
        for j, r in enumerate(chosen):
            name = r["image_name"]; p, gt = maps[name]
            rgb = Image.open(ROOT/f"data/train/images/{name}.png").convert("RGB")
            t = oof[family]["groups"][r["bucket"]]["oracle_shared_threshold"]["threshold"]
            heat = np.stack([p, np.zeros_like(p), 1-p], axis=-1)
            panels = [rgb, Image.fromarray(np.uint8(gt)*255), Image.fromarray(np.uint8(heat*255)),
                      Image.fromarray(np.uint8(p >= .5)*255), Image.fromarray(np.uint8(p >= t)*255), rgb.crop(r["box"])]
            labels = [f"{name} / {r['bucket']}", "GT", "P: blue=0 red=1", "Prediction t=0.5", f"Bucket oracle t={t:.4f}", f"GT crop bits={r['bit_accuracy']:.3f}"]
            for k, (im, label) in enumerate(zip(panels, labels)):
                canvas.paste(im.convert("RGB").resize((192, 192)), (k*192, j*222+30))
                draw.text((k*192+3, j*222+6), label, fill="black")
        canvas.save(dest/f"example_{family}.png")


def main():
    raw = read(OUT/"oof.json"); oof = raw["families"]; geo = read(OUT/"geometry.json")
    dec = {f: read(OUT/"decoder"/f/"summary.json") for f in FAMILIES}
    signals = {f: read(OUT/"signal"/f/"summary.json") for f in FAMILIES}
    tolerance = {f: read(OUT/"crop_tolerance"/f/"summary.json") for f in FAMILIES}
    text = ["# Etap 0: małe regiony aktualnego message_localizer\n",
            "Pomiar 2026-11-22, lokalna RTX 3070 8 GB. 170 obrazów PixelSeal i 170 TrustMark, ten sam seed42 i trzy foldy co baza #2163. **Bez nowego treningu i submission.** Najlepszy zaakceptowany wynik publiczny pozostaje 0,6145.\n",
            "## Wniosek\n",
            "**Samo przesunięcie progu lub hysteresis nie naprawia małych znaków.** Globalny oracle mIoU poprawia się najwyżej o około 0,0002 PS / 0,0006 TM w sprawdzonej rodzinie postprocessingu. Dla PS≤1% nawet próg dobierany osobno dla każdego obrazu daje średnie FG IoU tylko 0,0089. W TM jest więcej sygnału rankingowego, ale za mało do dużego zysku samą kalibracją.\n",
            "**Dopasowanie cropu ma znaczenie:** TM 1–5% z niemal prostokątną maską daje 99,05% bit accuracy po wycięciu bbox, a nieregularne maski 48,94%. Stałe okna128/192 pozostają blisko przypadku. Uzasadnia to osobne reżimy lokalnego osadzania i zachowanych fragmentów oraz kontrolowany test dopasowanego zoom-in. Nie jest to dowód odtworzenia historii edycji każdego obrazu.\n",
            "**Sygnał PS jest słaby i zależny od JND:** w kontrolach embed64×64 odsetek pikseli z dokładnym ΔRGB=0 to 51,45%; w dolnym kwartylu JND 74,88%. Chroma ma niewielką energię względem Y. To uzasadnienie testu pomocniczego residual head w Y, nie dowód jego przyszłej poprawy mIoU.\n",
            "## 1. Wyniki bazowe i kontrola protokołu\n",
            "Sześć map FP32 sprawdzono po nazwach, foldach i oficjalnym GT128; każdy obraz występuje dokładnie raz, outer-train i outer-validation są rozłączne. mIoU, FG/clean IoU, AUC, AP i puste predykcje odtworzone z map zgadzają się z zapisanymi metrykami. Hashe plików są w `results/oof.json`.\n",
            "Koszyki: (0,1%], (1%,5%], (5%,20%], >20%, według oficjalnego GT128. Clean holes to `full_watermark=true` i niejednorodna maska; jest dodatkową, nakładającą się kohortą. Dla niej raportujemy też IoU klasy clean, bo samo wysokie FG IoU ukrywa niewykrywanie dziur.\n"]
    rows = []
    for f in FAMILIES:
        for b in ("all",)+BUCKETS:
            g = oof[f]["groups"][b]
            rows.append([f, LABELS[b], g["n"], fmt(g["miou"], 5), fmt(g["fg_iou"], 5), fmt(g["clean_iou"], 5)])
    text.append(table(["Metoda", "Koszyk", "n", "mIoU", "FG IoU", "Clean IoU"], rows))
    text.extend(["## 2. Czy model widzi sygnał?\n", "Wszystkie średnie są macro po obrazach. PR-AUC(AP) oznacza nieinterpolowane average precision; dodatkową całkę trapezową zapisano w JSON/CSV. Max GT i max tła to średnie z maksimum w danym obszarze każdego obrazu, nie średnia wszystkich pikseli.\n"])
    rows = []
    for f in FAMILIES:
        for b in ("le1", "1to5"):
            g = oof[f]["groups"][b]
            rows.append([f, LABELS[b], g["n"], fmt(g["roc_auc"]), fmt(g["fg_ap"]), fmt(g["empty"]*100, 1)+"%", fmt(g["max_prob_gt"]), fmt(g["max_prob_bg"]), fmt(g["true_area"])])
    text.append(table(["Metoda", "Koszyk", "n", "ROC-AUC", "PR-AUC(AP)", "Puste t=.5", "Max GT", "Max poza GT", "Udział FG / losowy AP"], rows))
    text.append("### Oracle dla każdego koszyka\n\nPróg wspólny przeszukano dokładnie po wszystkich różnych wartościach prawdopodobieństwa, maksymalizując średnie FG IoU. Hysteresis: 8-sąsiedztwo, 85 wcześniej ustalonych par low/high; maksimum na tej siatce nie jest ścisłym supremum wszystkich par. Osobno pokazano silniejszy oracle progu per obraz. **Każdy oracle korzysta z GT ocenianych obrazów; nie jest wynikiem walidacyjnym nowej metody.** Optimum FG może drastycznie pogarszać clean IoU i oficjalne mIoU.\n")
    rows = []
    for f in FAMILIES:
        for b in BUCKETS:
            g = oof[f]["groups"][b]; t = g["oracle_shared_threshold"]; h = g["oracle_hysteresis_grid"]
            rows.append([f, LABELS[b], fmt(g["fg_iou"]), fmt(t["threshold"]), fmt(t["fg_iou"]), fmt(t["miou"]), f"{h['low']}/{h['high']}", fmt(h["fg_iou"]), fmt(h["miou"]), fmt(g["per_image_oracle_fg"])])
    text.append(table(["Metoda", "Koszyk", "FG t=.5", "Próg oracle", "FG oracle", "mIoU oracle", "Hyst low/high", "FG hyst", "mIoU hyst", "FG oracle per obraz"], rows))
    text.append("### Jedna polityka dla całych 170 obrazów\n\nTutaj progi maksymalizują **globalne oficjalne mIoU**, a następnie te same parametry stosujemy do wszystkich koszyków. Nadal jest to optymistyczna selekcja na całym OOF. Nie wybieramy polityki według nieznanego podczas inference rozmiaru GT.\n")
    rows = []
    for f in FAMILIES:
        for label, policy in oof[f]["policies"].items():
            g = policy["groups"]
            rows.append([f, label, fmt(g["all"]["miou"], 6), *[fmt(g[b]["fg_iou"]) for b in BUCKETS], fmt(g["clean_holes"]["clean_iou"])])
    text.append(table(["Metoda", "Polityka", "Globalne mIoU", "FG ≤1%", "FG 1–5%", "FG 5–20%", "FG >20%", "FG holes", "Clean holes"], rows))
    text.extend(["## 3. Zamrożony oficjalny dekoder i cropy\n", "Wszystkie43 małe PS i45 TM. Pełny obraz trafia bezpośrednio do wrappera. Crop bbox/margines/tło: bilinear antialias do256, potem niezmieniony wrapper. Natywna skala TM to245; osobny bbox_native sprawdza bezpośredni resize. Losowy crop128 tła wybierany wyłącznie spośród okien bez GT; kontrola64 losowych wiadomości daje około50%. GT-bbox jest diagnostycznym oracle, nie dostępną cechą testową.\n"])
    rows = []
    for f in FAMILIES:
        for b in ("le1", "1to5"):
            g = dec[f]["groups"][b]
            rows.append([f, LABELS[b], g["full"]["n"], *[fmt(g[v]["bit_accuracy"]*100, 2)+"%" for v in ("full", "bbox256", "margin128_to256", "margin192_to256", "background128_to256")], g["bbox256"]["n_ge90"]])
    text.append(table(["Metoda", "Koszyk", "n", "Pełny", "BBox→256", "128→256", "192→256", "Tło→256", "BBox ≥90%: n"], rows))
    text.append("Okna128/192 są wycentrowane na bbox, ale przy wydłużonym GT mogą ucinać znak: średni zachowany GT dla1–5% to PS83,7%/90,7%, TM85,6%/93,1%. To pomiar ograniczenia tych okien, nie test pełnego cropu z marginesem dla każdego wydłużonego obiektu. Bezpośredni bbox→245 dla TM daje68,22%/75,67%, wobec67,78%/75,39% przez256.\n")
    margin_rows = []
    for f in FAMILIES:
        for b in ("le1", "1to5"):
            g = dec[f]["groups"][b]
            margin_rows.append([f, LABELS[b], *[fmt(g[v]["bit_accuracy"]*100, 2)+"%" for v in ("bbox256", "bbox_margin16_to256", "bbox_margin32_to256")]])
    text.append("Dodatkowa kontrola po wyniku stałych okien: rozszerzenie każdej krawędzi bbox o 16 lub 32 px (ograniczone brzegiem obrazu), zachowujące 100% GT. Słaby wynik marginesu nie wynika więc wyłącznie z ucięcia watermarka.\n")
    text.append(table(["Metoda", "Koszyk", "Dokładny bbox", "BBox + 16 px", "BBox + 32 px"], margin_rows))
    rows = []; subgroups = []
    for f in FAMILIES:
        rr = read(OUT/"decoder"/f/"per_image.json")
        for b in ("le1", "1to5"):
            for rect in (True, False):
                subset = [r for r in rr if r["bucket"] == b and r["view"] == "bbox256" and (r["crop_gt_fraction"] > .99) == rect]
                if not subset:
                    continue
                rows.append([f, LABELS[b], "fill>0,99" if rect else "fill≤0,99", len(subset), fmt(mean(subset, "bit_accuracy")*100, 2)+"%"])
                subgroups.append({"family": f, "bucket": b, "near_rectangle": rect, "n": len(subset), "bit_accuracy": mean(subset, "bit_accuracy")})
    text.append(table(["Metoda", "Koszyk", "Kształt GT", "n", "Bit accuracy bbox"], rows))
    text.append("Podział według fill-ratio jest analizą następczą po wyniku cropów. Odzysk wiadomości wskazuje na zgodną geometrię osadzenia. Brak odzysku nie dowodzi transplantacji: kontrolne lokalne osadzanie PS48/64 również często nie daje pełnego odczytu.\n")
    text.append("### Tolerancja błędu położenia i rozmiaru cropu\n\nTest następczy po słabym wyniku marginesów: te same 17 wariantów na każdym z 88 małych obrazów. Poniżej niemal prostokątne GT (fill>0,99). Przesunięcia to średnia czterech kierunków ±x/±y, z ograniczeniem do obrazu, nie najlepszy kierunek. BBox z siatki128 pochodzi z **idealnego GT128**, a nie predykcji modelu.\n")
    tolerance_rows = []
    for f in FAMILIES:
        for b in ("le1", "1to5"):
            gs = {r["variant"]: r for r in tolerance[f]["groups"] if r["bucket"] == b and r["shape"] == "near_rectangle"}
            tolerance_rows.append([f, LABELS[b], gs["exact"]["n_images"], *[fmt(gs[v]["bit_accuracy"]*100, 2)+"%" for v in ("exact", "oracle_grid128", "expand_4", "expand_8", "shift_4", "shift_8")]])
    text.append(table(["Metoda", "Koszyk", "n", "Dokładny bbox", "GT128 bbox", "Margines4", "Margines8", "Przesunięcie4", "Przesunięcie8"], tolerance_rows))
    text.append("**TM ma praktyczną tolerancję przesunięcia:** w prostokątach 1–5% przesunięcie o 8 px nadal daje 97,37%, a bbox z GT128 99,11%. Silniejszy problem to rozmiar: margines4 daje 98,63%, margines8 już72,42%. Nie trzeba więc od razu zakładać subpikselowej lokalizacji. Pozostaje niewykonany test uzyskania takich propozycji bez GT. [Wyniki tolerancji PS](results/crop_tolerance/pixelseal/summary.json), [TM](results/crop_tolerance/trustmark/summary.json).\n")
    text.append("### Kontrole: lokalne osadzanie kontra fragment pełnego obrazu\n\n16 tych samych źródeł COCO i wiadomości per metoda. Oficjalny embedder, domyślna siła, FP32 i dokładna kwantyzacja uint8. To nowy mały zestaw kontrolny, nie oryginalne realizacje z treningu na klastrze.\n")
    rows = []
    for f in FAMILIES:
        for side in (48, 64, 128, 512):
            cs = {r["regime"]: r["bit_accuracy"] for r in signals[f]["controls"] if r["side"] == side}
            rows.append([f, side, *[fmt(cs[k]*100, 2)+"%" for k in ("local_embedding", "large_source_fragment", "clean")]])
    text.append(table(["Metoda", "Bok cropu", "Lokalne osadzenie", "Fragment embed512", "Czysty crop"], rows))
    text.append("## 4. Fizyka sygnału po uint8\n\nΔ mierzone w prostokącie osadzenia, względem identycznego czystego covera. Jednostki0–255. Zero pixel: wszystkie trzy kanały mają Δ=0. Y/Cb/Cr to liniowe reszty BT.601, bez offsetów. Niski JND oznacza dolny kwartyl oryginalnej mapy na cropie embeddera; wyniki średnie po16 coverach.\n")
    rows = []
    for f in FAMILIES:
        for r in signals[f]["signal"]:
            if r["side"] in (64, 512) and r["stratum"] in ("all", "low_jnd_q25", "high_jnd_q75", "smooth_q25"):
                rows.append([f, r["side"], r["stratum"], fmt(r["mae_rgb_uint8"]), fmt(r["zero_pixel_fraction"]*100, 2)+"%", fmt(r["mae_Y"]), fmt(r["mae_Cb"]), fmt(r["mae_Cr"])])
    text.append(table(["Metoda", "Bok", "Obszar", "Średnie |ΔRGB|", "Piksele Δ=0", "|ΔY|", "|ΔCb|", "|ΔCr|"], rows))
    text.append("Lokalny kod `unet_base_yuv_quant` wytwarza jednokanałowy sygnał dodawany do RGB, z JND i clippingiem. Zmierzone małe reszty chromy po uint8 oznaczają, że nie należy narzucać matematycznie zerowego ΔCb/Cr na końcowym obrazie. TrustMark też ma przewagę energii Y; sama przewaga Y nad chromą nie jest jeszcze sprawdzonym klasyfikatorem watermarka.\n")
    text.append("## 5. Geometria: dane realne i obecny generator\n\nMaski realne512 dla340 obrazów. Generator:1024 widoki na metodę, odtworzenie geometrii pierwszych32 batchy po32 według obecnego kodu i RNGseed42, z clean twins/complement/move. To **replay kodu**, nie odczyt archiwalnych masek treningowych. Pierwsze24 grupy porównano z oryginalnym `pair_views` na siatce128. Nie porównujemy do historycznego generatora starego ConvNeXt.\n")
    rows = []
    for f in FAMILIES:
        for pop in ("real", "generator_replay"):
            for pol in ("foreground", "minority"):
                r = geo["families"][f][pop][pol]["all"]
                rows.append([f, pop, pol, r["n"], *[fmt(r[k]["mean"]) for k in ("fill_ratio", "perimeter_over_area", "components8", "width_ridge_median", "edge_alignment")]])
    text.append(table(["Metoda", "Zbiór", "Polaryzacja", "n", "Fill bbox", "Obwód/pole", "Komponenty8", "Grubość px", "Zgodność krawędzi"], rows))
    text.append("Fill/thickness pomijają puste maski. Grubość to2×EDT na lokalnych maksimach3×3, przybliżenie szerokości fragmentów w pikselach512. Zgodność krawędzi: odsetek wewnętrznej granicy maski w odległości≤2px od górnego kwintyla Sobela. **To proxy krawędzi obrazu, nie semantyczna adnotacja obiektu.** Replay używa16 lokalnych coverów COCO, nie oryginalnych zdjęć klastra; interpretacja tej kolumny jest ograniczona. Pełne p10/mediana/p90 w JSON.\n")
    rows = []
    for f in FAMILIES:
        for pop in ("real", "generator_replay"):
            v = geo["families"][f][pop]["foreground"]
            for b in ("le1", "1to5", "5to20"):
                r = v[b]
                rows.append([f, pop, LABELS[b], r["n"], fmt(100*r["n"]/v["all"]["n"], 1)+"%", fmt(r["fill_ratio"]["mean"]), fmt(r["components8"]["mean"], 2), fmt(r["width_ridge_median"]["mean"], 1)])
    text.append(table(["Metoda", "Zbiór", "Koszyk", "n", "% widoków", "Fill bbox", "Komponenty", "Grubość px"], rows))
    text.append("**Nie obniżać globalnie fill-ratio do0,40–0,65:** wszystkie17 najmniejszych realnych PS mają pełny prostokąt (fill1). Dla TM1–5% niemal prostokątne19/36 dają prawie idealny odczyt. Generator ma równocześnie zbyt wiele rozrzuconych komponentów (minority średnio11,6/12,6 vs2,6/2,8 real). Kształt trzeba dopasować warunkowo do reżimu, wielkości i polaryzacji. W replay udział≤1% to7,3%/7,5% wszystkich widoków, a nie deklarowane20% minority źródeł; balansować po utworzeniu par i ostatecznych masek.\n")
    text.append("## 6. Kolejność dalszych ablacjii\n\n1. **Generator v2:** jawne embed rect i final mask; lokalne małe prostokąty oraz osobny tryb transplantacji z innego obrazu / clean holes z obiektu. Zachować25–35% same-image mixing. Balans koszyków mierzyć na końcowych widokach. Porównać pojedynczą zmianę do aktualnego generatora przy identycznych400 krokach pretrainingu i150 adaptacji/fold.\n2. **TM: dopasowany crop/refiner:** najpierw test kandydatów dostępnych bez GT. GT-bbox daje mocny sufit dla prostokątów; stałe128/192 nie uzasadniają same z siebie pełnego sliding-window. Raportować recall propozycji dla pustych predykcji bazowych.\n3. **PS: pomocniczy residual head w Y:** porównać L1 z kontrolą predykcji Δ=0; przy wielu zerach zwykłe średnie L1 może promować zerowy head. Sprawdzać residual error na niezerowych pikselach oraz właściwe realne FG IoU.\n4. **Loss:** kontrolowana pojedyncza zmiana nadal może poprawić wyuczone cechy/recall; słaby oracle nie dowodzi, że retraining lossu nie pomoże. Nie oczekiwać dużego zysku z samego obniżenia progu.\n5. **SRM/Bayar i semantyka:** oddzielne ablacjie po powyższych. Aktualny model już ma lokalną gałąź512 z pierwszym blokiem stride1, dalszym wyjściem stride4 i skipami piramidy. Zmiana filtrów to ablacjia reprezentacji, a nie naprawa nieobecnego skipu. Brak dotąd testu przewagi semantyki dla PS.\n\nEtapy1–2 **nie zostały wytrenowane w tym zadaniu**. [Brief do dalszej pracy na A100](HANDOFF_A100.md).\n")
    text.append("## 7. Artefakty, koszt i ograniczenia\n")
    elapsed = raw["seconds"]+geo["seconds"]+sum(dec[f]["seconds"]+signals[f]["seconds"]+tolerance[f]["seconds"] for f in FAMILIES)
    text.append(f"Suma czasów ostatnich wykonań skryptów wynosi około {elapsed:.1f} s (bez pisania kodu, wcześniejszych prób, transferu i budowy dashboardu). Diagnostyka OOF {raw['seconds']:.1f} s; geometria {geo['seconds']:.1f} s. Maksymalna pamięć zaalokowana przez CUDA w testach dekodera i kontrolach pozostawała poniżej 0,5 GB, więc ten etap nie wymagał A100. Nie jest to pomiar kosztu treningu nowego modelu.\n")
    text.append("OOF jest wielokrotnie używanym zbiorem rozwojowym, nie nowym niezależnym testem. Oracle z GT służy diagnozie. Kontrole sygnału obejmują16 źródeł, nie pełną pulę treningową. Oryginalne syntetyczne pary i ich manifesty pozostają na klastrze; dokładny audyt archiwalnych realizacji wymaga ich odczytu. Brak semantycznych adnotacji konkursowych ogranicza ocenę granic obiektów.\n")
    text.append("- [OOF i progi JSON](results/oof.json), [per-image CSV](results/oof_per_image.csv).\n- [Geometria i rozkłady](results/geometry.json), [PS CSV](results/geometry_pixelseal.csv), [TM CSV](results/geometry_trustmark.csv).\n- [Cropy PS](results/decoder/pixelseal/per_image.csv), [cropy TM](results/decoder/trustmark/per_image.csv).\n- [Sygnał PS](results/signal/pixelseal/summary.json), [sygnał TM](results/signal/trustmark/summary.json).\n- [Weryfikacja](results/verification.json), [pochodzenie i hashe](results/provenance.json).\n")
    figures(oof)
    text.append("Galerie wybierają najniższą i najwyższą bit accuracy bbox w każdym małym koszyku; nie są losową próbką.\n\n![PixelSeal](results/gallery/example_pixelseal.png)\n\n![TrustMark](results/gallery/example_trustmark.png)\n")
    save(OUT/"crop_shape_summary.json", subgroups)
    document = "\n".join(text)
    replacements = {"ablacjii": "ablacji", "ablacjie": "ablacje", "ablacjia": "ablacja", "Wszystkie43": "Wszystkie 43",
                    "i45": "i 45", "około50": "około 50", "kontrola64": "kontrola 64", "crop128": "crop 128", "do256": "do 256",
                    "to245": "to 245", "Okna128": "Okna 128", "okna128": "okna 128", "dla1": "dla 1", "PS83": "PS 83", "TM85": "TM 85",
                    "daje68": "daje 68", "wobec67": "wobec 67", "przez256": "przez 256", "COCO16": "COCO 16",
                    "embed64": "embed 64", "PS48": "PS 48", "embed512": "embed 512", "Jednostki0": "Jednostki 0", "po16": "po 16",
                    "Maski realne512": "Maski realne 512", "dla340": "dla 340", "Generator:1024": "Generator: 1024", "pierwszych32": "pierwszych 32",
                    "RNGseed42": "RNG seed 42", "Pierwsze24": "Pierwsze 24", "siatce128": "siatce 128", "to2×EDT": "to 2×EDT",
                    "maksimach3×3": "maksimach 3×3", "pikselach512": "pikselach 512", "odległości≤2px": "odległości ≤2 px", "używa16": "używa 16",
                    "wszystkie17": "wszystkie 17", "TM1–5%": "TM 1–5%", "prostokątne19/36": "prostokątne 19/36",
                    "średnio11": "średnio 11", "vs2": "vs 2", "to7,3": "to 7,3", "deklarowane20": "deklarowane 20",
                    "Zachować25": "Zachować 25", "identycznych400": "identycznych 400", "i150": "i 150", "gałąź512": "gałąź 512",
                    "stride1": "stride 1", "stride4": "stride 4", "Etapy1": "Etapy 1", "około{": "około {", "poniżej0,5GB": "poniżej 0,5 GB",
                    "obejmują16": "obejmują 16", "do0,40": "do 0,40", "fill1": "fill 1", "stałe128": "stałe 128", "udział≤1%": "udział ≤1%"}
    for before, after in replacements.items():
        document = document.replace(before, after)
    (HERE/"REPORT.md").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
