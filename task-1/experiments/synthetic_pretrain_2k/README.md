# Syntetyczny pretrening: 2000 przykładów StegaStamp

Plan ustalony przed oceną nowych modeli na rzeczywistych foldach:

- Pierwszy test dotyczy StegaStamp, aktywnej bazy inverse-STN. Nie zmieniamy ekstraktora ani głowicy.
- 2000 syntetycznych obrazów treningowych z 2000 różnych zdjęć COCO val2017; dodatkowe 200 niezależnych zdjęć do kontroli domeny syntetycznej. „val2017” to nazwa źródła COCO, a nie walidacja konkursowa.
- Prawdziwy lokalny encoder StegaStamp i losowe wiadomości 100-bitowe. 45% lokalnego embed(crop), 45% embed(full) z odtworzeniem czystego fragmentu, 5% całkowicie czystych i 5% całkowicie oznakowanych. Maski prostokątne, eliptyczne i obiektowe COCO. Kompozycja zawsze z tym samym czystym zdjęciem: bez widocznych szwów pochodzących od innego obrazu.
- Rozmiary/geometria wybrane z góry, bez dopasowania do masek walidacji konkursowej. Żadnych konkursowych obrazów testowych ani obrazów walidacyjnych jako materiału generacji. Kontrola duplikatów i zapis źródeł.
- Pretrening: dokładnie 10 epok na 2000 syntetykach, ten sam Head i loss 0.5 BCE + 0.5 Dice; brak selekcji na walidacji konkursowej. Trzy seedy 42/7/123.
- Fine-tuning: dokładnie istniejące zewnętrzne i wewnętrzne splity. Maks. 50 epok, patience 8, AdamW 0.001, batch 8. Trzy wewnętrzne foldy wybierają epokę i próg; świeży refit od tego samego syntetycznego checkpointu na całym outer-train. Optymalizator resetowany przed fine-tuningiem.
- Kontrola bez pretreningu uczona tym samym wykonawcą, urządzeniem i seedami. Historyczne checkpointy CPU podajemy dodatkowo, nie podstawiamy ich za kontrolę GPU.
- Porównanie wszystkich trzech foldów i seedów. Główna miara: parowane OOF oficjalne mIoU; dodatkowo foreground IoU, macro AP/AUC, delta względem empty, wyniki przy GT <=5%. Przedziały bootstrap po obrazach, uśredniając seedy; osobno rozrzut seedów.
- Dodatkowa kontrola syntetycznej walidacji i predykcji synthetic-only pozwoli odróżnić brak nauki syntetycznej od przesunięcia domeny / zapominania po fine-tuningu.
- Nie modyfikujemy istniejących checkpointów, podziałów, predykcji testowych ani zgłoszeń. Równoległy eksperyment RGB/SRM działa niezależnie.

To mały test jednej receptury syntetycznej i jednej rodziny znaków, nie dowód skuteczności/nieskuteczności wszystkich syntetyków. Brak generatywnego inpaintingu w tym przebiegu jest świadomym ograniczeniem domeny.

## Uruchomienie

Z katalogu repozytorium: `task-1/.venv/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/download.py`, następnie `prepare.py` (CPU / ONNX), `run.py` środowiskiem `.venv-rgb-cuda` na wolnym GPU 1, `report.py` i `verify.py`, a na końcu `task-1/dashboard/build.py`.

Literatura i uzasadnienie: [RESEARCH.md](RESEARCH.md). Wszystkie duże artefakty w ignorowanym `results/`.
