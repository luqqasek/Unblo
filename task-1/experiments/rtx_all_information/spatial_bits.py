"""Retain PixelSeal's complete native per-bit spatial output before any validation."""
import json,time,shutil,gc
import numpy as np
import torch
from .prepare import ROOT,OUT,Evidence,module,mmap,save

def prepare_native_bits():
    folder=OUT/'data/pixelseal';complete=folder/'native_bits_complete.json'
    if complete.exists():return
    assert (folder/'complete.json').exists()
    images=np.load(folder/'rgb.npy',mmap_mode='r');messages=np.load(folder/'message.npy',mmap_mode='r')
    source=module('allinfo_spatial_source',ROOT/'experiments/synthetic_pretrain_2k/prepare.py');e=Evidence('pixelseal',source)
    progress=folder/'native_bits_progress.json';done=json.loads(progress.read_text())['completed'] if progress.exists() else 0
    array=None;start=time.monotonic()
    for i in range(done,len(images)):
        _,bits,_=e.extract(np.array(images[i]).transpose(1,2,0),np.array(messages[i]),bit_size=None)
        if array is None:
            path=folder/'bits_native.npy'
            if not path.exists():assert shutil.disk_usage(folder).free>len(images)*bits.nbytes+10_000_000_000
            array=mmap(path,(len(images),*bits.shape),bits.dtype)
            print('NATIVE_BITS_SHAPE',array.shape,'GB',array.nbytes/1e9,flush=True)
        array[i]=bits
        if (i+1)%100==0 or i+1==len(images):
            array.flush();save(progress,{'completed':i+1,'total':len(images),'seconds':time.monotonic()-start})
            print('NATIVE_BITS',i+1,len(images),'seconds',round(time.monotonic()-start,1),flush=True)
    if array is None:array=np.load(folder/'bits_native.npy',mmap_mode='r')
    save(complete,{'shape':list(array.shape),'bytes':array.nbytes,'seconds':time.monotonic()-start,'note':'Both native decoder input scales retained at maximum native bit-map spatial resolution; no spatial downsampling.'})
    for hook in e.hooks:hook.remove()
    del e
    gc.collect();torch.cuda.empty_cache()
