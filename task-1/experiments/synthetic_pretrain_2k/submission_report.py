"""Record the explicitly requested deployment separately from the earlier CV audit."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results/full_train'
MARKER = '\n## Zgłoszenie po fine-tuningu — aktualizacja\n'

def main():
    prediction = json.loads((OUT / 'prediction_summary.json').read_text())
    train = json.loads((OUT / 'training_summary.json').read_text())
    receipt_path = OUT / 'upload_receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    response = json.loads(receipt['response']) if receipt else None
    if receipt:
        with (OUT / 'submission.pt').open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['sha256'] == prediction['submission_sha256']
    lines = ['# Zgłoszenie: syntetyki → fine-tuning → test', '',
             'Po zakończeniu eksperymentu użytkownik polecił przygotować i wysłać predykcje testowe, a następnie jawnie wybrał wariant po fine-tuningu. Wcześniejszy audyt nie korzystał z testu; poniższa operacja jest oddzielnym, późniejszym etapem.', '',
             '## Model i wybór parametrów', '',
             '- StegaStamp: zapisany pretrain na 2000 syntetyków, seed 42, 10 epok; świeży fine-tuning na wszystkich 170 rzeczywistych obrazach.',
             f'- Fine-tuning: **{prediction["epochs"]} epok**, próg **{prediction["threshold"]}**; mediany epok 7/5/10 i progów 0.5/0.6/0.6 z trzech foldów seedu 42.',
             '- Niezmieniona głowica, zamrożony ekstraktor, rzeczywista korekta inverse-STN. AdamW LR=0.001, weight decay=0.0001, batch 8, float32 CUDA, TF32 off; optymalizator zresetowany po pretreningu.',
             '- Domyślny seed 42, bez wyboru najlepszego seedu. Test nie wybierał modelu, epoki, progu ani postprocessingu.',
             '- PixelSeal i TrustMark: zachowane 680 masek z najlepszego wcześniejszego zgłoszenia #1202. Nowe są wszystkie 340 predykcji StegaStamp.', '',
             'Wynik CV wariantu po fine-tuningu: 0.50719 po trzech seedach, 0.51514 dla seedu 42. Synthetic-only osiągał średnio 0.51153, ale nie został wysłany. Obecne zgłoszenie nie jest porównaniem obu wariantów na leaderboardzie.', '',
             '## Walidacja pliku', '',
             f'**PASS:** 1020 unikalnych wymaganych nazw, tensor uint8 1020×128×128, wyłącznie 0/1; {prediction["bytes"]:,} bajtów. Ponowne wczytanie checkpointu dało identyczne prawdopodobieństwa. Sprawdzono przypisanie masek po nazwie i niezmienność pozostałych 680 masek.', '',
             f'SHA-256 zgłoszenia: `{prediction["submission_sha256"]}`.', '',
             f'SHA-256 nowego checkpointu: `{prediction["checkpoint_sha256"]}`.', '',
             f'Udział pikseli dodatnich StegaStamp: {prediction["watermark_fraction_stegastamp"]:.5f}. Zmieniono {prediction["changed_pixels_vs_previous"]:,} pikseli względem wcześniejszego StegaStamp. To statystyki predykcji, nie miary jakości na nieznanym GT.', '',
             '## Odpowiedź leaderboardu', '']
    if response and response.get('status') == 'success':
        lines += [f'**Zaakceptowano #{response["submission_id"]}: publiczne mIoU {response["score"]:.4f}.** Poprzedni najlepszy wynik {response["previous_best"]:.4f}; zmiana **{response["delta"]:+.4f}**.', '',
                  'Serwer potwierdził nowy najlepszy wynik.' if response.get('improved') else 'Poprzedni najlepszy wynik pozostaje wyższy.', '',
                  'To wynik publicznego podzbioru. Końcowy ranking zależy od ukrytego podzbioru ujawnianego po zamknięciu zgłoszeń. Wykonano jeden POST; nie dostrajano modelu na podstawie tej odpowiedzi.', '']
    else:
        lines += ['Brak potwierdzonego przyjęcia — zobacz zapis odpowiedzi serwera.', '']
    lines += ['[Odpowiedź serwera](results/full_train/upload_receipt.json) · [Walidacja predykcji](results/full_train/prediction_summary.json) · [Konfiguracja pełnego treningu](results/full_train/training_summary.json)', '',
              '## Odtworzenie predykcji', '', '```powershell',
              'task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/deploy.py fit',
              'task-1/.venv/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/deploy.py extract',
              'task-1/.venv-rgb-cuda/Scripts/python.exe task-1/experiments/synthetic_pretrain_2k/deploy.py predict', '```', '',
              '`fit` i `extract` wznawiają zgodne artefakty; `predict` chroni istniejący plik zgłoszenia przed nadpisaniem. Wysyłka jest osobnym poleceniem i nie stanowi części odtwarzania predykcji. Checkpointy oraz plik submission znajdują się lokalnie w `results/full_train/`.', '']
    (HERE / 'SUBMISSION_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    main_report = HERE / 'REPORT.md'
    original = main_report.read_text(encoding='utf-8').split(MARKER)[0].rstrip()
    outcome = (f'Zaakceptowane zgłoszenie **#{response["submission_id"]}**, publiczne mIoU **{response["score"]:.4f}**, zmiana względem wcześniejszego najlepszego **{response["delta"]:+.4f}**.'
               if response and response.get('status') == 'success' else 'Predykcje przygotowane; status wysyłki w osobnym raporcie.')
    main_report.write_text(original + '\n' + MARKER + '\nNa późniejsze, jawne polecenie użytkownika wykonano fine-tuning na wszystkich 170 realnych obrazach StegaStamp i predykcję testową. ' + outcome + '\n\n[Pełny raport zgłoszenia](SUBMISSION_REPORT.md). Wcześniejsze stwierdzenia o nieużywaniu testu opisują etap eksperymentalny przed tą operacją.\n', encoding='utf-8')
    print(outcome)

if __name__ == '__main__': main()
