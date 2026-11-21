# Audyt lokalnego checkoutu — 2026-11-21, Windows

## Dostępne

- `wmloc/data.py`: loader obrazów, metod, wiadomości oraz obu masek train.
- `wmloc/watermarks.py`: wrappery PixelSeal, TrustMark Q i StegaStamp;
  dekodowanie i dostęp do cech (StegaStamp domyślnie tylko logity).
- `wmloc/utils.py` i `src/watermark_localization/metrics.py`: próg >= 50%,
  średnia IoU obu klas per obraz, pusta unia = 1.
- `src/watermark_localization/submission.py`: kontrola pełnego zbioru nazw,
  unikalności, kształtu i binarności; zapis uint8 i limit 20 MB.
- Skrypty instalacji, pobierania danych i wag, testy zgodności metryk.
- `experiments/baseline/`: powtarzalne porównanie prostych masek bez dekodera.

## Stan przed naprawą

- `data/` i `checkpoints/` zawierają wyłącznie `.gitkeep`.
- Brak pobranego `starter-kit/` i środowiska `.venv`.
- Launcher `py` nie znajduje instalacji Pythona 3.11 wymaganej przez setup.ps1.
  Dostępny `python` to zewnętrzne środowisko 3.12.12, bez PyTorch.
- Nie można jeszcze potwierdzić zawartości 510/1020 obrazów, masek, wiadomości,
  zgodności downsamplingu ani działania trzech dekoderów na tej maszynie.
- `solution.py` pozostaje pustym punktem wejścia (`NotImplementedError`).
- Szablon przewiduje zera, nie tworzy katalogu wyjściowego i ma placeholdery
  adresu serwera/API key. Nowy baseline korzysta z walidowanego eksportu.

Pierwotny `SETUP_STATUS.md` dokumentował wcześniejsze środowisko. Dodano w nim
aktualną weryfikację Windows. Dane i modele są celowo ignorowane przez Git.

## Uruchomienie

Pełne środowisko: z `task-1` wykonać
`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1`.
Skrypt tworzy `.venv`; przy braku `py -3.11` instaluje lokalny interpreter przez
uv, używając istniejącego polecenia `python`. Pobiera dane i modele.
Dla baseline'u potrzebne są tylko dane i NumPy;
PyTorch jest potrzebny dopiero do eksportu submission. Potem uruchomić
`experiments/baseline/run.py` interpreterem wybranego środowiska.

## Naprawa i wyniki

- Utworzono projektowe `.venv` z Pythonem 3.11.18 i PyTorch 2.5.1 CPU.
- Instalacja TrustMark wymaga `PYTHONUTF8=1` na polskim Windows; setup ustawia tę zmienną.
- Dodano sprawdzone ograniczenia wersji `requirements-windows.lock.txt`.
- Pobrano przypiętą wersję danych i startera; trzy sumy SHA-256 są zgodne z manifestem.
- Wczytano 510/1020 obrazów, potwierdzono liczebności per metoda i zgodność
  wszystkich 510 downsamplowanych masek z `masks_128.npz`.
- Baseline na walidacji: zera 0.241116, jedynki 0.258884, częstość 0.262779.
- Zapisano submission częstości: 1020 masek, 16 734 635 bajtów, bez wysyłania.
- Pięć testów repozytorium i dwa testy baseline'u przeszły; `pip check` bez błędów.
- Cały `scripts/setup.ps1` zakończył się kodem 0. Wszystkie trzy dekodery
  zwróciły skończone logity o poprawnych wymiarach i oczekiwane mapy cech na CPU.
- Pobieranie modeli na Windows używa curl z ponawianiem i limitami czasu.
  Wagi TrustMark są pobierane wcześniej i weryfikowane sumami z pakietu.

Test syntetyczny sprawdza
rozłączność podziału, brak wykorzystania masek walidacyjnych przy uczeniu priory,
kolejność metod, próg remisu i kształt wyjść. Nie zastępuje audytu datasetu.
