"""Independent exhaustive-label checks for installed moment formulas."""
from itertools import combinations, product
import unittest
import numpy as np
from meq import features
from meq import moments as exact


def assignments(n, a):
    for left in combinations(range(n), a):
        yield list(left), [j for j in range(n) if j not in left]


def enumerate_mmd(k, a):
    values = []
    for left, right in assignments(len(k), a):
        aa = [k[i, j] for i in left for j in left if i != j]
        bb = [k[i, j] for i in right for j in right if i != j]
        ab = [k[i, j] for i in left for j in right]
        values.append(np.mean(aa)+np.mean(bb)-2*np.mean(ab))
    return np.array(values)


def enumerate_style(rows, a):
    return np.array([rows[left].mean(axis=0)-rows[right].mean(axis=0)
                     for left, right in assignments(len(rows), a)])


class ExactMoments(unittest.TestCase):
    def test_mmd_all_partitions(self):
        rng = np.random.default_rng(2718)
        for a, b in [(2, 2), (2, 3), (3, 3), (2, 5), (3, 4), (5, 2)]:
            for kind in ['random', 'duplicate', 'constant']:
                rows = rng.normal(size=(a+b, 3))
                if kind == 'duplicate': rows[::2] = rows[0]
                k = np.exp(-np.square(rows[:, None]-rows[None, :]).sum(axis=2))
                if kind == 'constant': k[:] = .7
                np.fill_diagonal(k, 0.)
                values = enumerate_mmd(k, a)
                with self.subTest(a=a, b=b, kind=kind):
                    np.testing.assert_allclose(exact.mmd_moments(k, a), [values.mean(), values.var()], atol=1e-13, rtol=1e-10)

    def test_mmd_shift_scale_and_order_invariance(self):
        rng = np.random.default_rng(314)
        k = rng.normal(size=(7, 7)); k=(k+k.T)/2; np.fill_diagonal(k, 0.)
        expected = exact.mmd_moments(k, 3)[1]
        for scale, shift in [(1., 100.), (2., -5.), (.001, 0.)]:
            changed = k*scale+shift; np.fill_diagonal(changed, 0.)
            order=rng.permutation(7)
            self.assertAlmostEqual(exact.mmd_moments(changed[order][:, order], 3)[1], expected*scale**2, places=10)

    def test_style_all_partitions(self):
        rng = np.random.default_rng(1618)
        for a, b in [(1, 1), (1, 3), (2, 2), (2, 3), (3, 3), (4, 3), (5, 2)]:
            for dimension in [1, 4]:
                for kind in ['random', 'duplicate', 'constant']:
                    rows=rng.normal(size=(a+b, dimension))
                    if kind=='duplicate': rows[::2]=rows[0]
                    if kind=='constant': rows[:]=2.5
                    vectors=enumerate_style(rows, a)
                    squared=np.sum(vectors*vectors, axis=1)
                    mean,var,cov=exact.style_moments(rows,a)
                    with self.subTest(a=a,b=b,dimension=dimension,kind=kind):
                        np.testing.assert_allclose([mean,var],[squared.mean(),squared.var()],atol=1e-12,rtol=1e-10)
                        np.testing.assert_allclose(cov,vectors.T@vectors/len(vectors),atol=1e-12,rtol=1e-10)

    def test_style_independent_blocks(self):
        rng=np.random.default_rng(99)
        for specs in [[(2,2),(2,3)],[(1,1),(1,2),(2,2)]]:
            blocks=[(rng.normal(size=(a+b,3)),a) for a,b in specs]
            vectors=[enumerate_style(rows,a) for rows,a in blocks]
            squared=[np.sum(np.mean(draw,axis=0)**2) for draw in product(*vectors)]
            np.testing.assert_allclose(exact.style_null(blocks),[np.mean(squared),np.var(squared)],atol=1e-12,rtol=1e-10)

    def test_api_observed_statistics_match(self):
        raw_a={'m0':{i:s for i,s in enumerate(['red ball','blue ball','red cat'])},'m1':{0:'a',1:'b'}}
        raw_b={'m0':{i:s for i,s in enumerate(['red ball','blue cat','green cat','green ball'])},'m1':{0:'c',1:'a'}}
        a,b=features.represented(raw_a),features.represented(raw_b)
        self.assertAlmostEqual(features.mmd(a,b)['stat'],exact.mmd(a,b)['stat'],places=14)
        self.assertEqual(exact.mmd(a,b,1,3),exact.mmd(a,b,400,99))
        sa,sb=features.style_sets(raw_a),features.style_sets(raw_b)
        scale=features.style_scale([sa,sb])
        self.assertAlmostEqual(features.style_test(sa,sb,scale)['shift'],exact.style_test(sa,sb,scale)['shift'],places=14)
        self.assertEqual(exact.style_test(sa,sb,scale,1,3),exact.style_test(sa,sb,scale,400,99))

    def test_empty_and_degenerate(self):
        self.assertEqual(exact.mmd({}, {})['z'],0.)
        self.assertEqual(exact.style_test({}, {}, (0.,1.))['z'],0.)
        reps=features.represented({'m0':{0:'same',1:'same',2:'same'}})
        self.assertEqual(exact.mmd(reps,reps)['z'],0.)
        style={'m0':[[1.,2.],[1.,2.]]}
        self.assertEqual(exact.style_test(style,style,(0.,1.))['z'],0.)


if __name__=='__main__':
    unittest.main(verbosity=2)
