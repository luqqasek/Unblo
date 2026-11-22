"""Explicit, strict loader for the experimental architecture-surgery checkpoints."""
import torch
from torch import nn
from experiments.manual_weights.nonlinear_surgery import SurgicalHead
from experiments.manual_weights.mixture_surgery import MixtureHead


def make_core(state):
    old_network=None
    if any(k.startswith('old_network.') for k in state):
        from experiments.native_localizer.model import block
        old_network=nn.Sequential(block(len(state['mean']),128),nn.Conv2d(128,1,1))
    core=SurgicalHead(state['mean'],state['scale'],state['projection'],state['prototypes'],state['weight'],state['original'],old_network)
    core.load_state_dict(state,strict=True)
    return core


def replacement(saved):
    state=saved['replacement_head'];kind=saved['replacement_module'].split('.')[-1]
    if kind=='SurgicalHead':return make_core(state)
    if kind=='MixtureHead':
        core=make_core({k.removeprefix('core.'):v for k,v in state.items() if k.startswith('core.')})
        s=saved['selection'];head=MixtureHead(core,state['experts'],state['router_mean'],state['router_scale'],state['router'],s['temperature'],s['blend'],s['hard'])
        head.load_state_dict(state,strict=True);return head
    raise ValueError('Unsupported replacement head: '+kind)


def attach(model,saved):
    if 'replacement_head' not in saved:return model
    path=saved['replace_path'];parent,name=path.rsplit('.',1)
    model.get_submodule(path)  # Reject misspelled paths rather than create an unused module.
    setattr(model.get_submodule(parent),name,replacement(saved))
    return model
