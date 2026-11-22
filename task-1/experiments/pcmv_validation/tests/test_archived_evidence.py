"""CPU-only checks: no model imports, training, downloads or archive mutations."""
import ast
import runpy
from unittest.mock import patch
import json
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).resolve().parents[1]

def read(name):
    return json.loads((HERE/'results'/name).read_text(encoding='utf-8'))

class ArchivedEvidenceTests(unittest.TestCase):
    def test_oracle_against_exhaustive_pixel_edits(self):
        # Load only the pure function; do not import SAM2 or execute module setup.
        tree=ast.parse((HERE/'proposal_screen.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='oracle')
        namespace={'np':np}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<oracle>','exec'),namespace)
        def score(p,y):
            return np.mean([((p==k)&(y==k)).sum()/((p==k)|(y==k)).sum() if ((p==k)|(y==k)).any() else 1. for k in [0,1]])
        rng=np.random.default_rng(42)
        for _ in range(50):
            y=rng.random((4,5))>.5;p=rng.random((4,5))>.5;m=rng.random((5,4,5))>.75
            got=namespace['oracle'](p,y,m);one=two=score(p,y)
            for a in range(2*len(m)):
                edited=p.copy();edited[m[a//2]]=a%2==0;one=max(one,score(edited,y))
                for b in range(2*len(m)):
                    if (m[a//2]&m[b//2]).any():continue
                    pair=edited.copy();pair[m[b//2]]=b%2==0;two=max(two,score(pair,y))
            self.assertAlmostEqual(got['oracle_one'],one,places=12)
            self.assertAlmostEqual(got['oracle_two'],max(one,two),places=12)

    def test_training_coordinators_are_inert_on_import(self):
        for filename in ['run.py','prepare_all.py']:
            with patch('subprocess.run',side_effect=AssertionError('Unexpected training launch')) as launch, patch('os.chdir') as chdir:
                runpy.run_path(str(HERE.parent/'rtx_all_information'/filename),run_name='review_import')
                launch.assert_not_called()
                chdir.assert_not_called()

    def test_geometry_summary_matches_per_image_evidence(self):
        d=read('proposal_screen.json');rows=d['per_image']
        self.assertEqual(len(rows),57)
        self.assertEqual(len({r['image_name'] for r in rows}),57)
        for key in ['base','oracle_one','oracle_two']:
            self.assertAlmostEqual(d['summary']['all'][key],np.mean([r[key] for r in rows]),places=12)
        for row in rows:self.assertLessEqual(row['base'],row['oracle_one']);self.assertLessEqual(row['oracle_one'],row['oracle_two'])
        self.assertFalse(d['trained_model']);self.assertFalse(d['PCMV_gain_measured'])

    def test_message_null_and_selection_correction(self):
        d=read('signal_oracle.json')
        for group,summary in d['summary'].items():
            rows=[r for r in d['per_image'] if r['kind']==group and 'scores' in r]
            self.assertEqual(len(rows),summary['n_evaluated'])
            self.assertEqual(summary['n_total']-len(rows),summary['n_excluded'])
            null=[];true=[]
            for key,stats in summary['methods'].items():
                v=np.array([r['scores'][key]['paired_ranking_true'] for r in rows]);w=np.array([r['scores'][key]['paired_ranking_wrong'] for r in rows])
                self.assertEqual(w.shape,(len(rows),256))
                self.assertAlmostEqual(v.mean(),stats['same_image_matched_ranking_auc'],places=12)
                self.assertAlmostEqual(w.mean(),stats['wrong_message_mean'],places=12)
                null.append(w.mean(0));true.append(v.mean())
            adjusted=(1+(np.max(null,axis=0)>=max(true)).sum())/257
            self.assertEqual(adjusted,summary['max_over12_methods_wrong_p'])
        self.assertTrue(d['labels_used'])

if __name__=='__main__':unittest.main()
