"""Reuse PIQ's explicitly supported feature_extractor argument; keep its code path."""
def install():
    from piq import FID
    from piq.feature_extractors import InceptionV3
    original=FID.compute_feats
    def cached(self,loader,feature_extractor=None,device='cuda'):
        if feature_extractor is None:
            # Store off nn.Module registration: feature extractor stays a separate official module.
            if not hasattr(self,'_shared_feature_extractor'):
                object.__setattr__(self,'_shared_feature_extractor',InceptionV3())
            feature_extractor=self._shared_feature_extractor
        return original(self,loader,feature_extractor=feature_extractor,device=device)
    FID.compute_feats=cached
