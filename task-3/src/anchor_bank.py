"""Class-balanced sampling of validated synthetic anchors; no retain-accuracy claim."""
import torch

class AnchorBank:
    def __init__(self, path, seed=0):
        data=torch.load(path,map_location='cpu',weights_only=True)
        images,labels,valid=data['images'],data['targets'],data['valid']
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError('Anchor images must be NHWC uint8')
        if len(images)!=len(labels) or len(images)!=len(valid) or valid.dtype!=torch.bool:
            raise ValueError('Invalid anchor index arrays')
        self.images=images[valid]; self.labels=labels[valid]
        if not len(self.images):raise ValueError('No accepted synthetic anchors')
        self.classes=self.labels.unique(sorted=True)
        self.indices=[(self.labels==c).nonzero().flatten() for c in self.classes]
        self.g=torch.Generator().manual_seed(seed)
    def sample(self,n):
        classes=torch.randint(len(self.indices),(n,),generator=self.g)
        ids=torch.stack([self.indices[c][torch.randint(len(self.indices[c]),(),generator=self.g)] for c in classes])
        return self.images[ids]
