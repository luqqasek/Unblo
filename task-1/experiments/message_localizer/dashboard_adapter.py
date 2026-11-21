"""Publish measured message-localizer validation; no tensors or test claims."""
import json
import shutil

def add_rows(experiments, dist, row):
    from polish import COPY
    labels = {
        'Message-conditioned synthetic model': 'Model z wiadomością, trening syntetyczny',
        'Synthetic model and incumbent, fixed equal ensemble': 'Model syntetyczny i dotychczasowy, stała średnia',
        'Message-conditioned real adaptation': 'Model z wiadomością po adaptacji',
        'Adapted model and incumbent, fixed equal ensemble': 'Model po adaptacji i dotychczasowy, stała średnia',
        'Fixed 3-fold development OOF, 170 images': 'Stałe 3-fold rozwojowe OOF, 170 obrazów',
        'Synthetic-only transfer, 170 real images': 'Transfer bez realnego treningu, 170 obrazów',
        'message localizer': 'Lokalizacja ze znaną wiadomością',
    }
    COPY.update(labels)
    base = experiments / 'message_localizer/results'
    path = base / 'comparison.json'
    if not path.exists(): return
    variants = {
        'synthetic_only': 'Message-conditioned synthetic model',
        'synthetic_only_fixed_equal_ensemble': 'Synthetic model and incumbent, fixed equal ensemble',
        'adapted': 'Message-conditioned real adaptation',
        'adapted_fixed_equal_ensemble': 'Adapted model and incumbent, fixed equal ensemble',
    }
    for method, result in json.loads(path.read_text(encoding='utf-8')).items():
        for key, value in result['variants'].items():
            protocol = 'Fixed 3-fold development OOF, 170 images' if key != 'synthetic_only' else 'Synthetic-only transfer, 170 real images'
            lo, hi = value['paired_bootstrap95']
            row('message_localizer', method, variants[key], protocol, value['miou'],
                f"Delta {value['delta_vs_incumbent']:+.5f}; paired image bootstrap 95% [{lo:+.5f}, {hi:+.5f}]. Development data; no new test score.")
    target = dist / 'reports/message_localizer/results'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target / path.name)

    submission=base/'submission'
    if submission.exists():
        dest=target/'submission';dest.mkdir(parents=True,exist_ok=True)
        for path in submission.glob('*.json'):shutil.copyfile(path,dest/path.name)
