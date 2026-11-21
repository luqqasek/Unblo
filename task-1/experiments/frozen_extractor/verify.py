"""Recompute validation outputs from saved heads and frozen caches; verify ONNX weights."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import onnx
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('experiment_b',HERE/'run.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def main():
    torch.set_num_threads(4)
    original=onnx.load(module.ROOT/'checkpoints/stegastamp/stega_stamp_logits.onnx')
    derived=onnx.load(module.OUT/'stegastamp_features.onnx')
    assert [v.SerializeToString() for v in original.graph.initializer]==[v.SerializeToString() for v in derived.graph.initializer]
    for method in ['pixelseal','trustmark','stegastamp']:
        cache=torch.load(module.OUT/f'{method}_features.pt',weights_only=True)
        saved=torch.load(module.OUT/f'{method}_head.pt',weights_only=True)
        report=json.loads((module.OUT/f'{method}_metrics.json').read_text())
        val=np.load(module.OUT/f'{method}_validation.npz')
        head=module.Head(saved['channels']); head.load_state_dict(saved['state_dict']); head.eval()
        ids=[cache['names'].index(n) for n in report['val_names']]
        with torch.inference_mode():
            prob=torch.cat([head([f[ids[j:j+8]] for f in cache['features']]).sigmoid() for j in range(0,34,8)]).numpy()
        np.testing.assert_allclose(prob,val['probability'],atol=1e-6)
        result=module.metrics(prob,val['target'],report['selected']['threshold'])
        assert abs(result['miou']-report['selected']['miou'])<1e-12
        assert len(list(module.OUT.glob(f'{method}_*_example.png')))==10
        print(method,'checkpoint, metric and 10 visualizations verified')
    print('StegaStamp pretrained initializers unchanged')


if __name__=='__main__': main()
