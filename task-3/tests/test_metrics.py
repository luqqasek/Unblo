import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from metrics import score, forget_group

class MetricTests(unittest.TestCase):
    def setUp(self):
        self.o=np.array([[6.,0.],[0.,5.],[4.,0.]])
        self.g=np.array([[2.,0.],[0.,1.],[0.,1.]])
        self.y=np.array([0,1,0])
    def run_score(self, sub, val=.8):
        group=(self.o,sub,self.g[None],self.y)
        return score(group, group, self.o,self.o,self.y,val)
    def test_original_zero(self):
        self.assertAlmostEqual(self.run_score(self.o)['final_score'],0.)
    def test_single_gold_perfect(self):
        self.assertAlmostEqual(self.run_score(self.g)['final_score'],1.)
    def test_gate(self):
        self.assertEqual(self.run_score(self.g,.699)['final_score'],0.)
        self.assertAlmostEqual(self.run_score(self.g,.7)['final_score'],1.)
    def test_gold_statistics_per_model(self):
        gs=np.stack([self.g,self.g*2])
        r=forget_group(self.o,self.g,gs,self.y)
        from metrics import ce
        self.assertAlmostEqual(r['gold_std'],np.mean([ce(g,self.y).std() for g in gs]))
    def test_average_components_before_multiply(self):
        a=(self.o,self.g,self.g[None],self.y)
        b=(self.o,self.o,self.g[None],self.y)
        r=score(a,b,self.o,self.o,self.y,.8)
        self.assertAlmostEqual(r['final_score'],.25)
    def test_undefined_denominator(self):
        with self.assertRaises(ValueError):
            forget_group(self.o,self.o,self.o[None],self.y)
    def test_invalid_labels_and_nonfinite_logits(self):
        from metrics import ce, log_softmax
        with self.assertRaises(ValueError):
            ce(self.o, np.array([-1, 1, 0]))
        with self.assertRaises(ValueError):
            log_softmax(np.array([[float('nan'), 1.]]))

if __name__=='__main__': unittest.main()
