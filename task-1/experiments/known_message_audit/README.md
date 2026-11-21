# Znana wiadomość: kontrolowany audyt nowych map

Plan zapisany przed nowymi pomiarami, 2026-11-22.

Wcześniejsze `decoder_evidence` obejmuje projekcję pre-GAP i korektę per-bit;
nie powtarzamy twierdzenia, że wiadomości nigdy nie używano.

Etap bez treningu: wszystkie 170 realnych obrazów PixelSeal oraz TrustMark.
Gęste logity w natywnej rozdzielczości (256 / 245) i 512: dot, cosine,
soft accuracy, norma. Prawdziwa wiadomość i dwie niezależne losowe wiadomości
z deterministycznym seedem; różnica względem średniej losowych kontroli.
Blur-IG: rozmycie Gaussian sigma=3 w oryginalnych współrzędnych 512, cztery
prawe końce przedziałów ścieżki, surowy signed-logit score, suma wszystkich
kanałów RGB, average pooling 4 do 128. IG dla natywnej rozdzielczości ekstraktora;
dodatkowy test 16 kroków na pierwszych ośmiu obrazach kontroluje kwadraturę.
Bez obietnicy dokładnej kompletności przy skończonej liczbie kroków.

Keep/Drop: istniejące OOF ensemble trzech seedów z synthetic_scaling,
PS n4k_s2500 / TM n4k_s5000. Progi .1... .9 plus oryginalny próg OOF.
Dwa ustalone kryteria: maksymalny score keep minus drop; oraz najmniejsza
maska zachowująca >=95% przyrostu score obrazu nad rozmyciem i pozostawiająca
<=5% tego przyrostu poza maską. Gdy przyrost <=0 lub brak kandydata,
drugi wariant zachowuje oryginalny próg. Identyczne kryteria dla losowej wiadomości.
GT nie wybiera progu. Maksimum mIoU po kandydatach raportowane wyłącznie jako oracle.

Ocena: macro ROC-AUC i AP per obraz, osobno małe GT <=5% i duże GT >50%;
paired bootstrap po obrazach dla efektu prawdziwej wiadomości i zmian mIoU.
Istniejące foldy są wielokrotnie użytym zbiorem rozwojowym. Brak niezależnego
testu i brak submission. Norma logitów nie jest warunkowana wiadomością.
Orientacja map ustalona dodatnio; nie odwracamy znaków po obejrzeniu GT.

Kandydat do dalszej fuzji musi mieć macro AUC >=.60, dodatni dolny CI95
różnicy AUC względem losowej wiadomości i nie pogarszać small AP względem
tej kontroli. To bramka diagnostyczna, nie dowód poprawy segmentatora.

Przed uruchomieniem dodatkowo ustalono test re-embedding NCC dla wszystkich
170 obrazów każdej z trzech metod: oryginalny wrapper embeddera, różnica
G(I,b)-G(I,1-b), HPF I-Gaussian(sigma=3), lokalna wycentrowana korelacja
w oknach 16 i 32, luminancja dla PS i RGB dla TM/SS. Dwie losowe wiadomości
z tym samym seedem per obraz co wyżej; raportować oba rozmiary okien oraz
podział według lokalnego/globalnego osadzania (tylko diagnostyka z metadanych).
To niezależny test nowej rodziny cech, bez zmiany wcześniej ustalonych testów.

Po pomiarze map (adaptacyjny etap rozwojowy): natywne cosine dla PS/TM
przeszły bramkę sygnału. Sprawdzamy korektę logits zamrożonego, istniejącego
segmentatora seed42. Wejście korekty: cos/soft-accuracy/norma dla obu skal
oraz logit bazowy; mały CNN 7->8->1 z zerowym wyjściem początkowym.
Kontrola: ta sama korekta uczona z jedną fałszywą wiadomością; norma identyczna.
Wewnętrzny model bazowy odtworzony z syntetycznego checkpointu i zapisanego
wyboru epoki, wyłącznie na inner-train. Statystyki map liczone tylko na train.
AdamW .001, weight decay .0001, batch8, loss .5BCE+.5Dice, maks.30 epok,
patience8, epoka0 dopuszczona; próg .3/.4/.5/.6/.7 z inner-validation.
Refit korekty na outer-train przy bazie z odpowiedniego checkpointu outer.
Screening fold1 seed42; pozostałe foldy tylko jeśli delta mIoU >=.005
nad bazą i nad kontrolą wrong. Dodatkowo podmiana wiadomości w inference.
To świadomie test małej fuzji, nie pełnego end-to-end finetuningu ekstraktora.

Uruchomienie z katalogu repo:

```powershell
task-1/.venv-rgb-cuda/Scripts/python.exe -u task-1/experiments/known_message_audit/probe.py
task-1/.venv-rgb-cuda/Scripts/python.exe -u task-1/experiments/known_message_audit/reembed_probe.py
task-1/.venv-rgb-cuda/Scripts/python.exe -u task-1/experiments/known_message_audit/reembed_controls.py
task-1/.venv-rgb-cuda/Scripts/python.exe -u task-1/experiments/known_message_audit/fusion.py
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/known_message_audit/report.py
task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/known_message_audit/verify.py
python task-1/dashboard/build.py
```

Skrypty wznawiają istniejące artefakty w `results/`; po zmianie protokołu użyć
nowego katalogu wyników zamiast mieszać stare i nowe pomiary. `verification.json`
zawiera hashe aktualnego kodu i wyniki odtworzenia metryk oraz checkpointów.
Kontrola re-embedding po wynikach: osiem pierwszych nazw zdjęć z istniejącego
COCO val2017.zip, centralny czysty otwór, ta sama wiadomość/pozycja/skala.
Nie jest to niezależny benchmark; sprawdza działanie metody w zgodnej geometrii.
