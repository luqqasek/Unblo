# Task 3 — Unlearning

European AI Championship Paris 2026. Id serwerowe: `31-unlearning`.
Celem jest zmiana dostarczonego modelu tak, by zachowywał się jak model
wytrenowany bez 3500 wskazanych obrazów, z zachowaniem jakości na reszcie danych.
Przygotowany baseline to niezmieniony oryginał, zgodnie z oficjalnym szablonem.
Nie jest to metoda unlearningu; oczekiwany wynik to około zera.

## Status przygotowania — 2026-11-21

Pobrano checkpoint i forget set, sprawdzono opublikowane SHA-256.
Skonfigurowano osobne środowisko Python 3.12 CPU; `pip check` bez błędów.
Checkpoint: 176 tensorów, 85 893 988 parametrów, 343 617 075 bajtów,
float32, bez NaN/Inf, poprawne `strict=True`. Dane: 3500 obrazów uint8,
224x224 RGB, bez etykiet. Jedna lokalna predykcja CPU zwróciła skończone
logity `[1,100]`; raport w `outputs/local_verification.json`.

**Zgłoszenie przyjęte: ID 1106, HTTP 200, status success, public score 0.0.**
Próba rozpoczęta 2026-11-21 16:29:07 UTC / 17:29:07 Warszawa.
Potwierdzenie: `outputs/submission_20261121T162907Z.json`.
Wysłano raz, bez ponawiania. Baseline jest identyczny bajtowo z oryginałem:
SHA-256 `485881a03ad064b5e6defc245ae3f45fd96714bf065ff96543a5eeca9f642571`.
Odpowiedź serwera nie zawiera szczegółów utility gate ani składowych wyniku.
Nie wykonano zapytań do zdalnych modeli.

## Zasoby i format

- `data/model.pt`: oryginalny weights-only state dict, około 344 MB.
- `data/forget_images.npz`: `images`, uint8 `[3500,224,224,3]`, RGB, bez etykiet.
- `official/`: niezmienione skrypty organizatorów i manifest SHA-256/revision.
- `official/hackathon/`: kopia oficjalnych instrukcji i skryptów JURECA z task-2.
- `outputs/baseline.pt`: dokładna kopia wag oryginału do testowego zgłoszenia.
- `workflow.py`: generowanie baseline, walidacja i wysyłka state dict.
- `verify_local.py`: walidacja danych i jedna lokalna predykcja CPU.
- `.env`: `UNLEARNING_API_KEY`, lokalny sekret ignorowany przez Git.

Architektura: timm `vit_base_patch16_224`, `pretrained=False`, `num_classes=100`,
`init_values=1e-4` (LayerScale). Obrazy: NHWC uint8 -> NCHW float32,
`(pixels/255 - 0.5)/0.5`, bez dodatkowego resize lub normalizacji ImageNet.

Zgłoszenie musi zawierać sam state dict `.pt` lub `.pth`, ładowalny z
`strict=True`, float32, bez NaN/Inf, maksymalnie 450 MB. Adaptery/LoRA należy
scalić z oryginalnymi wagami. Serwer używa eval i fp16 autocast podczas inference;
nie zmienia to wymogu float32 dla przesłanych wag.

## Metryka i ograniczenia

`orig` = oryginał, `gold` = kilka modeli retrenowanych bez forget set,
`sub` = zgłoszenie. Rozkłady predykcji liczone są z softmax(logits / 2).
Prawdziwe etykiety, gold, retain/control i validation nie są udostępnione.
Nie da się lokalnie policzyć pełnej metryki ani sprawdzić utility gate.

1. Utility gate: accuracy na ukrytej walidacji poniżej 70% zeruje wynik.
2. Dla forget set i ukrytych zmodyfikowanych kopii osobno:
   `d_m = abs(mean(CE_m)-mean(CE_gold)) + abs(std(CE_m)-std(CE_gold))`;
   `loss_score = 1 - clip(d_sub/d_orig, 0, 1)`;
   `prediction_score = 1 - clip(mean KL(p_gold||p_sub) / mean KL(p_gold||p_orig), 0, 1)`.
3. Uśrednić osobno loss_score i prediction_score po obu grupach (równe wagi),
   następnie pomnożyć te średnie: `forget_score`.
4. `retain_score = clip(acc_sub(control)/acc_orig(control),0,1)`
   razy `1/(1+mean KL(p_orig||p_sub))` na control set.
5. `final_score = forget_score * retain_score`.

Gold loss mean/std to średnie tych statystyk po modelach gold; p_gold to
średni rozkład predykcji. Opis nie precyzuje obsługi zerowych mianowników.
Samo obniżenie accuracy na forget set albo dopasowanie średniego loss nie wystarcza:
liczy się rozkład odpowiedzi dla poszczególnych obrazów i ich ukrytych modyfikacji.
Nie używać naiwnego gradient ascent jako dowodu skutecznego zapominania.

Public split to około 30%, końcowy około 70%, deterministycznie według hasha ID;
zmodyfikowana kopia dziedziczy split oryginału. Leaderboard pokazuje najlepszy
wynik zespołu, odpowiedź na zgłoszenie wynik bieżący. Nie stroić do public score.

## Uruchomienie (PowerShell, z task-3)

```powershell
python download_resources.py
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe workflow.py generate
.\.venv\Scripts\python.exe verify_local.py
.\.venv\Scripts\python.exe workflow.py submit
```

CPU wystarcza do przygotowania i smoke testu. Środowisko GPU na JURECA wymaga
dobrania PyTorch do CUDA; nie instalować tam bezrefleksyjnie wersji CPU.
`requirements-windows.lock.txt` zapisuje przetestowane lokalnie wersje;
do odtworzenia użyć go zamiast requirements.txt (dla torch CPU dodać
`--extra-index-url https://download.pytorch.org/whl/cpu`).

Generator nie nadpisuje istniejącego pliku. Walidacja dowolnego checkpointu:
`python workflow.py validate --file outputs/candidate.pt`.
Wysyłka: `python workflow.py submit --file outputs/candidate.pt`.
Klucz odczytywany z otoczenia lub `.env`, nigdy nie zapisany w kodzie.
Wysyłka zapisuje hash i odpowiedź serwera w `outputs/submission_*.json`, bez
automatycznego ponawiania. Zachowano lokalny 30-minutowy odstęp między próbami
jako ostrożną blokadę; opis task-3 nie podaje takiego limitu serwera.

Nie są wykonywane zapytania do API modeli task-2. Lokalna predykcja na
pobranym ViT nie zużywa tamtego budżetu. Jedynym wywołaniem serwera konkursowego
w tym workflow jest wysłanie checkpointu do oceny.

## Punkt startowy do właściwej metody

Zachować oryginał jako punkt odniesienia. Pseudolabels z niego nie są etykietami
prawdziwymi. Metoda musi działać tylko z oryginalnymi wagami i forget images;
brak retain set uniemożliwia zwykły retraining na czystych danych. Rozważać
ograniczone aktualizacje z regularizacją do oryginału oraz stabilność na
augmentacjach, jawnie oddzielając lokalne proxy od ukrytej metryki konkursowej.
Ten workspace przygotowuje dane i ścieżkę zgłoszenia; nie trenuje nowej metody.

## Źródła

- https://huggingface.co/datasets/SprintML/unlearning
- https://huggingface.co/datasets/SprintML/hackathon
- http://34.147.229.55/leaderboard_page
- Opis zadania przekazany przez użytkownika w rozmowie 2026-11-21.
- Lektury organizatorów (niezweryfikowane w tym przygotowaniu):
  https://arxiv.org/abs/2302.09880, https://arxiv.org/abs/2201.05629,
  https://arxiv.org/abs/2310.12508, https://arxiv.org/abs/2404.05868.
