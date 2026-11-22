"""Offline regression checks for reproducible hillclimbing evidence."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from meq import decide, features
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
import hillclimb


class PermutationTests(unittest.TestCase):
    def test_null_method_switch_keeps_independent_caches(self):
        data = self.example_data()
        ev = decide.Evidence(data, permutations=17, null_method='montecarlo')
        def measurements(e):
            return [e.z('a', 'b'), e.style('a', 'b'), e.against('a', ['b', 'c']), e.on('a', 'b', 'm')['z']]
        sampled = measurements(ev)
        ev.use_null_method('exact')
        deterministic = measurements(ev)
        self.assertNotEqual(sampled, deterministic)
        self.assertEqual(deterministic, measurements(decide.Evidence(data, permutations=1, seed=555, null_method='exact')))
        ev.use_null_method('montecarlo')
        self.assertEqual(measurements(ev), sampled)
        # An old Evidence pickle has no null_method field and must retain its
        # original Monte Carlo semantics and cache entries.
        del ev.null_method
        self.assertEqual(measurements(ev), sampled)

    def test_invalid_null_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown null method'):
            decide.Evidence({}, null_method='unknown')

    def test_supplied_families_include_hidden_endpoints_and_never_cross(self):
        ev = SimpleNamespace(names=['a-hidden', 'b-plain', 'c-other-family'])
        supplied = {'a-hidden': 'A', 'b-plain': 'A', 'c-other-family': 'B'}
        seen = []

        def model_stage(evidence, members, params):
            seen.append(list(members))
            return {e: 1 for e in members}, {e: 'model stage' for e in members}

        with patch.object(decide, 'nameless', return_value={'a-hidden': 'hidden prompt'}), \
                patch.object(decide, 'models_in_family', side_effect=model_stage):
            explanations = {}
            labels = decide.labels_from(ev, explain=explanations, known_families=supplied)
        self.assertEqual(labels['a-hidden'], labels['b-plain'])
        self.assertNotEqual(labels['a-hidden'][0], labels['c-other-family'][0])
        self.assertEqual(seen, [['a-hidden', 'b-plain'], ['c-other-family']])
        self.assertEqual(set(explanations), set(ev.names))

    def test_supplied_families_must_cover_every_endpoint(self):
        with self.assertRaisesRegex(ValueError, 'Missing supplied families for: missing'):
            decide.labels_from(SimpleNamespace(names=['missing']), known_families={})

    def test_duplicate_kernel_matrix_is_exact(self):
        replies = [features.represent('m1-task', x) for x in
                   ['red blue', 'one two', 'red blue', 'blue red', '', 'one two', '']]
        expected = np.zeros((len(replies), len(replies)))
        for i in range(len(replies)):
            for j in range(i + 1, len(replies)):
                expected[i, j] = expected[j, i] = features.kernel(replies[i], replies[j])
        actual = features.kernel_matrix(replies)
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(actual[0, 2], 0)
        self.assertEqual(actual[0, 0], 0)

    @staticmethod
    def example_data():
        return {'a': {'m1-task': {0: 'red green blue', 1: 'green blue', 2: 'red blue'}},
                'b': {'m1-task': {0: 'one two three', 1: 'two three four', 2: 'one four'}},
                'c': {'m1-task': {0: 'yes yes yes yes', 1: 'yes yes', 2: 'yes'}}}

    def test_pool_member_order_does_not_change_test(self):
        data = self.example_data()
        first = decide.Evidence(data, permutations=60).against('a', ['b', 'c'])
        second = decide.Evidence(data, permutations=60).against('a', ['c', 'b'])
        self.assertEqual(first, second)

    def test_pair_scaling_ignores_unrelated_endpoints(self):
        data = self.example_data()
        before = decide.Evidence(data, permutations=60, style_normalization='pair').style('a', 'b')
        data['unrelated'] = {'m1-task': {0: 'HUGE! ' * 1000, 1: 'small'}}
        after = decide.Evidence(data, permutations=60, style_normalization='pair').style('a', 'b')
        self.assertEqual(before, after)

    def test_packed_agreement_metrics_use_their_declared_support(self):
        a = {('p', 1): ['yes', 'yes'], ('p', 2): ['shared']}
        b = {('p', 1): ['no', 'no'], ('p', 2): ['shared']}
        result = features.pair(a, b)
        self.assertEqual(result['matched_cosine'], 0.0)
        self.assertEqual(result['hist_cosine'], 0.5)
        self.assertEqual(features.pair(a, a)['hist_cosine'], 1.0)

    def test_vectorized_matches_scalar(self):
        rng = np.random.default_rng(24)
        a = {p: rng.normal(size=(3, 14)).tolist() for p in ['z', 'a', 'm']}
        b = {p: rng.normal(size=(4, 14)).tolist() for p in ['m', 'z', 'a']}
        scale = features.style_scale([a, b])
        mean, spread = scale
        blocks = [((np.vstack([a[p], b[p]]) - mean) / spread, len(a[p])) for p in sorted(a)]
        def statistic(orders):
            total = sum(rows[o[:n]].mean(axis=0) - rows[o[n:]].mean(axis=0)
                        for (rows, n), o in zip(blocks, orders))
            return float(np.sum((total / len(blocks)) ** 2))
        rng = np.random.default_rng(42)
        observed = statistic([np.arange(len(rows)) for rows, _ in blocks])
        null = np.array([statistic([rng.permutation(len(rows)) for rows, _ in blocks]) for _ in range(300)])
        result = features.style_test(a, b, scale, seed=42)
        self.assertAlmostEqual(result['z'], (observed - null.mean()) / null.std(), places=12)
        self.assertAlmostEqual(result['shift'], observed, places=12)

    def test_process_hash_seed_does_not_change_evidence(self):
        program = '''
import json
import numpy as np
from meq import features
rng=np.random.default_rng(5)
a={p:rng.normal(size=(3,14)).tolist() for p in ['alpha','beta','gamma','delta']}
b={p:rng.normal(size=(4,14)).tolist() for p in ['delta','beta','alpha','gamma']}
print(json.dumps(features.style_test(a,b,features.style_scale([a,b]),permutations=60,seed=17),sort_keys=True))
'''
        outputs = [subprocess.check_output([sys.executable, '-c', program],
                   env={**os.environ, 'PYTHONHASHSEED': str(seed)}, text=True) for seed in [1, 2, 77]]
        self.assertEqual(len(set(outputs)), 1, outputs)

    def test_no_shared_prompts(self):
        scale = (np.zeros(14), np.ones(14))
        self.assertEqual(features.style_test({}, {}, scale), {'prompts': 0, 'z': 0.0, 'shift': 0.0})

    def test_empty_style_pool(self):
        mean, spread = features.style_scale([{}, {}])
        self.assertEqual(mean.shape, (features.LAYOUT,))
        self.assertTrue(np.isfinite(spread).all())


class SearchTests(unittest.TestCase):
    def test_unreliable_audit_does_not_mark_endpoint_as_disguised(self):
        data = {'a': {'a1-audit': {0: 'YES', 1: 'YES'}},
                'b': {'a1-audit': {0: 'NO', 1: 'NO'}},
                'c': {'a1-audit': {0: 'NO', 1: 'NO'}}}
        ev = SimpleNamespace(data=data, profiles={e: {'self': 0.4} for e in data})
        params = {**decide.PARAMS, 'audit_min_self': 0.5}
        self.assertEqual(decide.prompted(ev, list(data), params), {})
        ev.profiles = {e: {'self': 0.8} for e in data}
        self.assertEqual(set(decide.prompted(ev, list(data), params)), {'a'})

    def test_retired_openai_is_not_evaluated_or_exposed(self):
        self.assertEqual(set(hillclimb.SETS), {'practice', 'testbed'})
        self.assertEqual(set(hillclimb.LIVE), {'practice', 'testbed'})
        self.assertEqual(hillclimb.WEIGHTS, {'testbed': 0.625, 'practice': 0.375})
        with self.assertRaisesRegex(ValueError, 'Inactive dataset'):
            hillclimb.evidence('openai', False)
        with patch.object(hillclimb, 'quick', return_value=0.8) as quick:
            weighted, _ = hillclimb.objective({}, full=False)
        self.assertAlmostEqual(weighted, 0.8)
        self.assertEqual([call.args[0] for call in quick.call_args_list], ['testbed', 'practice'])

    def test_weighted_gain_cannot_hide_practice_regression(self):
        def objective(settings, full=False):
            value = settings.get('candidate', 0)
            weighted, practice = {0: (0.4, 0.5), 1: (0.8, 0.3), 2: (0.6, 0.5)}[value]
            return weighted, {name: {'score': practice if name == 'practice' else weighted}
                              for name in hillclimb.WEIGHTS}
        with patch.object(hillclimb, 'GRID', {'candidate': [1, 2]}), patch.object(hillclimb, 'objective', objective):
            settings, score = hillclimb.climb(log=lambda _: None)
        self.assertEqual(settings, {'candidate': 2})
        self.assertEqual(score, 0.6)


if __name__ == '__main__':
    unittest.main()
