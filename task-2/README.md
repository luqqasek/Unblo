# Task 2 — Model Equality

Workspace do European AI Championship Paris 2026. Id zadania na serwerze:
`33-model-equality`. Przygotowanie i losowy baseline nie wymagają GPU ani zapytań do modeli.

## Status przygotowania — 2026-11-21

Pobrano 11 oficjalnych plików (w tym metadane Git i setup JURECA), sprawdzono
ich SHA-256, utworzono lokalne środowisko Python 3.12 i zainstalowano zależności.
Walidacja CSV: 33 endpointy, 3 losowe rodziny, 9 losowych modeli.
Scorer praktyczny na odpowiedziach wzorcowych: 1.000 (sanity check).

Wysłano dokładnie jedno zgłoszenie: **ID 1098**, HTTP 200, `status=success`,
**public score 0.0**. Czas próby: 2026-11-21 16:20:34 UTC / 17:20:34 Warszawa.
Potwierdzenie: `outputs/submission_20261121T162034Z.json`.
SHA-256 CSV: `cfcd77c19ebe43594a2e3e84636e4d69cb5216b8eacc998537b32a7f03663dd6`.
W tym przygotowaniu wykonano **0 zapytań do modeli**, w tym referencyjnych.
Nie sprawdzano wcześniejszego wykorzystania budżetu przez zespół.

## Zasady i kontekst

33 deploymenty (`ep-01`–`ep-33`) należy podzielić na rodziny i dokładne modele.
Rodzina oznacza wspólny model bazowy. Dokładna tożsamość wymaga tego samego
checkpointu, kwantyzacji i adaptera LoRA. Zmiana watermarku, system promptu
lub dekodowania nie zmienia tożsamości. Kwantyzacja lub inny adapter ją zmieniają.

CSV zawiera dokładnie 33 wiersze i kolumny `endpoint,family,model`.
Etykiety są dowolne, ale wspólna etykieta modelu wymaga wspólnej rodziny.
Klasteryzacja zapewnia przechodniość: A=B i B=C implikuje A=C.

Wynik: `0.1 * MCC_family + 0.9 * MCC_model`, z ujemnym MCC obciętym do zera.
MCC to `(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))`.
Rodziny oceniane są na parach endpointów (łącznie 528), modele tylko na parach
należących do tej samej prawdziwej rodziny. Publiczna i końcowa ocena używają
różnych podzbiorów par. Nie stroić metody do leaderboardu; wynik losowy służy
wyłącznie sprawdzeniu ścieżki zgłoszenia.

Limity: 100 odpowiedzi na każdy oceniany endpoint na cały konkurs, maksymalnie
256 tokenów wyjścia, 1024 tokeny wejścia i 4 równoległe requesty na zespół.
Każda odpowiedź w żądaniu zużywa osobną jednostkę. Temperature i seed są
ignorowane, wiadomości systemowe odrzucane. Błędne zapytania nie zużywają
budżetu odpowiedzi. Zgłoszenie CSV można wykonać raz na 30 minut; błąd zgłoszenia
także uruchamia cooldown.

**Na polecenie użytkownika nie odpytywać modeli ani endpointów referencyjnych
w ramach przygotowania.** Nie uruchamiano też endpointu `/models`.
`workflow.py` nie zawiera wywołań inference: jego jedyne żądanie sieciowe to
wysyłka CSV na `/submit/33-model-equality` po jawnej komendzie `submit`.
Oryginalne skrypty z `official/` mogą odpytywać modele — nie uruchamiać ich
do samego generowania losowego pliku.

## Pliki

- `official/`: komplet plików SprintML/model_equality bez modyfikacji.
- `official/hackathon/`: instrukcja i skrypty JURECA; pobrane, nie uruchamiane na Windows.
- `official/manifest.json`: adresy przypięte do rewizji, rozmiary i SHA-256 pobrań.
- `workflow.py`: generowanie offline, walidacja i wysłanie CSV.
- `outputs/random_seed_2026.csv`: losowe predykcje, bez wnioskowania o modelach.
- `outputs/submission_*.json`: odpowiedź serwera i hash wysłanego pliku.
- `.env`: lokalny klucz, ignorowany przez Git; wzór w `.env.example`.
- `requirements-windows.lock.txt`: wersje pakietów sprawdzone na tej maszynie.

W oficjalnym repo nie ma wag ani korpusu odpowiedzi do pobrania. Odpowiedzi do
analizy trzeba byłoby zebrać później, w ramach osobno zatwierdzonego budżetu.

## Uruchamianie (PowerShell, z folderu task-2)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe workflow.py generate
.\.venv\Scripts\python.exe workflow.py validate
# Tylko wysyłka CSV, nie odpytuje modeli:
.\.venv\Scripts\python.exe workflow.py submit
```

Na Linux/macOS użyj `.venv/bin/python`. Generator domyślnie używa seeda 2026,
trzech losowych rodzin i do trzech modeli na rodzinę, jak oficjalny przykład.
To założenie techniczne, nie oszacowanie liczby rzeczywistych rodzin/modeli.
Nie nadpisuje istniejącego pliku. Inny plik: `--file outputs/other.csv --seed 42`.
Walidator akceptuje nagłówek lub jego brak. Klucz odczytywany jest z
`MODEL_EQUALITY_API_KEY` w środowisku lub lokalnym `.env`.

Skrypt zapisuje próbę przed wysłaniem, blokuje kolejną lokalną próbę przez 30 minut
i nie ponawia żądań przy błędzie. Cooldown obejmuje cały zespół: lokalny zapis
nie wykrywa zgłoszeń z innych maszyn. Nie usuwać zapisu próby w celu ponowienia.

## Część praktyczna i dalsza praca

`ref-01`–`ref-09` używają osobnego małego modelu. Według oficjalnego szablonu
nie mają limitu zapytań i nie są oceniane na serwerze; tutaj ich nie odpytywano.
Wszystkie należą do rodziny P. Ten sam model S: ref-01, ref-03, ref-05,
ref-06, ref-08 (plain, system prompt, persona, watermark, sampling).
Pozostałe to różne modele: ref-02 FP8, ref-04 LoRA 0.7,
ref-07 LoRA 1.5, ref-09 LoRA 0.3. Nie przenosić tych etykiet na ep-*.

Lokalny scorer nie wykonuje żądań sieciowych:

```powershell
.\.venv\Scripts\python.exe official/score_practice.py official/practice_answers.csv
```

Powyżej jest sanity check na odpowiedziach wzorcowych, nie ocena opracowanej metody.
Scorer praktyczny liczy tylko MCC tożsamości modelu na 36 parach jednej rodziny;
nie odtwarza pełnej ważonej oceny rodzin i modeli dla ep-*.

Dalszy plan: przygotować prompty rozróżniające modele, mierzyć zmienność na
powtórzeniach, oceniać odporność na system prompts/watermark/dekodowanie na
praktyce, następnie klasteryzować rodziny i modele. Dla właściwych endpointów
planować budżet przed zebraniem danych, zachowywać odpowiedzi i rezerwę na
niepewne pary. Sam styl wypowiedzi ani nazwa zadeklarowana przez model nie
stanowią dowodu tożsamości. To plan, nie zaimplementowany algorytm.

## Źródła

- Oficjalne pliki: https://huggingface.co/datasets/SprintML/model_equality
- Setup JURECA: https://huggingface.co/datasets/SprintML/hackathon
- Leaderboard: http://34.147.229.55/leaderboard_page
- Opis konkursu przekazany przez użytkownika w rozmowie 2026-11-21.
- Lektury wskazane przez organizatorów (niezweryfikowane w tym przygotowaniu):
  MET https://arxiv.org/pdf/2410.20247,
  RUT https://arxiv.org/pdf/2506.06975,
  AgentProv https://arxiv.org/pdf/2609.00052.
