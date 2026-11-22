"""Rebuild and verify the repository dashboard; never run training or GPU probes."""
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[2]

def main():
    subprocess.run([sys.executable, str(ROOT / 'dashboard/build.py')], check=True)
    dist = ROOT / 'dashboard/dist'
    data = json.loads((dist / 'data.json').read_text(encoding='utf-8'))
    rows = [r for r in data['rows'] if r['experiment'] == 'pcmv_validation']
    stopped = [r for r in data['rows'] if r['experiment'] == 'rtx_all_information']
    assert len(rows) == 3 and len(stopped) == 4
    for name in ('index.html', 'index-en.html'):
        page = (dist / name).read_text(encoding='utf-8')
        for row in rows + stopped:
            assert str(row['miou']) in page
        assert 'pcmv_validation' in page and 'rtx_all_information' in page
    assert (dist / 'reports/pcmv_validation/REPORT.html').exists()
    print('Verified PL/EN dashboards: 3 PCMV reference/oracle rows and 4 stopped-experiment rows.')

if __name__ == '__main__':
    main()
