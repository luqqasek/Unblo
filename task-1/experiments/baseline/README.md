# Baseline bez dekodera

Trzy punkty odniesienia: same zera, same jedynki oraz przestrzenna częstość
watermarku, osobno dla każdej metody. Ostatni wariant oznacza komórkę jako 1,
jeśli co najmniej połowa masek uczących ma tam watermark. Nie analizuje RGB ani
wiadomości; mierzy, ile daje sam rozkład masek. Nie wymaga wag ani GPU.

Podział: 408 obrazów do wyznaczenia częstości, 102 do walidacji (34 na metodę).
Nazwy są sortowane przed losowaniem; seed 42. Próg 0.5 jest ustalony z góry.
Wynik to oficjalne mIoU po dwóch klasach, następnie po obrazach, również osobno
dla metod. Nie wolno porównywać go z wynikiem na całym train ze startera.

Z katalogu repozytorium:

```powershell
python task-1/experiments/baseline/run.py
python task-1/experiments/baseline/run.py --submission
```

Ewaluacja wymaga NumPy oraz pobranych danych w `task-1/data`. Eksport wymaga też
PyTorch. `--data`, `--out`, `--seed` pozwalają nadpisać ustawienia. Wyniki trafiają
do `results/metrics.json` (metryki i dokładne nazwy podziału) i `priors.npz`.

Najlepszy wariant wybieramy na walidacji. Jego wynik jest wynikiem selekcji,
nie niezależnym oszacowaniem po wyborze modelu. Submission wykorzystuje częstości
ponownie obliczone na całym train. Powstaje lokalny, walidowany plik uint8
`[1020,128,128]`, poniżej 20 MB; skrypt niczego nie wysyła.

Kolejny eksperyment: `../sliding_window/` z porównywaniem wiadomości dekodowanych
z fragmentów obrazu ze znaną wiadomością. Następnie `../wykorzystanie_embeddingu/`:
mapy cech PixelSeal/TrustMark i prosty klasyfikator komórek. StegaStamp domyślnie
udostępnia tylko globalne logity. Każdy eksperyment powinien zachować ten sam
podział i zapisywać konfigurację, wyniki per metoda oraz koszt obliczeń.

## Wyniki — 2026-11-21

Walidacja: 102 obrazy, seed 42, bez zmiany progu na podstawie walidacji.

| Wariant | mIoU | PixelSeal | TrustMark | StegaStamp |
| --- | ---: | ---: | ---: | ---: |
| Same zera | 0.241116 | 0.192271 | 0.286991 | 0.244086 |
| Same jedynki | 0.258884 | 0.307729 | 0.213009 | 0.255914 |
| Częstość per metoda | **0.262779** | 0.231726 | 0.272808 | 0.283805 |

Wybrano częstość per metoda. Submission po ponownym dopasowaniu na 510 maskach:
`results/submission.pt`, 1020 masek uint8, 16 734 635 bajtów. Wynik walidacyjny
nie jest wynikiem leaderboardu.

Submission wysłany 2026-11-21 i zaakceptowany przez serwer (HTTP 200).
ID: `1065`. Publiczne mIoU: **0.2934**. Wynik dotyczy publicznego podzbioru;
wynik prywatny pozostaje ukryty. Odpowiedź zapisana w `results/upload_receipt.json`.

## Wysyłka na leaderboard

Ustawić `WMLOC_API_KEY` w środowisku albo w ignorowanym `task-1/.env`
(`WMLOC_API_KEY=klucz_zespolu`). Opcjonalnie `WMLOC_BASE_URL` zmienia serwer.
Z katalogu repozytorium:

```powershell
.\task-1\.venv\Scripts\python.exe task-1/experiments/baseline/upload.py
```

Skrypt waliduje plik, wysyła go jeden raz i zapisuje odpowiedź serwera
w `results/upload_receipt.json`. Nie ponawia automatycznie próby po timeout.
