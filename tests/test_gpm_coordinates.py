import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.models.odr_lora import ODRLoRAConv2d
from src.scripts.audit_gpm_coordinates import sampled_patches,fit_basis,coverage


class CoordinateAuditTests(unittest.TestCase):
    def test_matched_memory_sources_share_counts_and_merge_rule(self):
        raw=ODRLoRAConv2d(2,2,3)
        freq=ODRLoRAConv2d(2,2,3)
        x=torch.randn(2,2,16,16)
        counts=[]
        for layer,kind in [(raw,'raw'),(freq,'frequency')]:
            layer.gpm_statistics=kind
            layer.gpm_update_rule='bounded_svd'
            layer.gpm_sample_columns=512
            with patch.object(layer.dual_gpms['global'],'update') as update, patch.object(
                    layer.dual_gpms['global'],'update_preserving') as preserving:
                layer.update_gpm(x,0)
                update.assert_called_once()
                preserving.assert_not_called()
                counts.append(update.call_args.args[0].shape[1])
        self.assertEqual(counts[0],counts[1])
        self.assertLessEqual(counts[0],512)

    def test_raw_sampling_matches_real_convolution_layout(self):
        layer=ODRLoRAConv2d(2,3,3,stride=2)
        x=torch.randn(1,2,23,25)
        h=sampled_patches(layer,x,32)
        actual=F.conv2d(x,layer.weight,None,stride=2,padding=1).flatten(2)[0]
        index=torch.linspace(0,actual.shape[1]-1,32).long()
        torch.testing.assert_close(layer.weight.flatten(1)@h,actual[:,index])

    def test_bounded_preserving_keeps_span_and_matches_column_budget(self):
        x=torch.randn(2,2,16,16)
        sizes=[]
        for rule in ('bounded_svd','bounded_preserving'):
            layer=ODRLoRAConv2d(2,2,3)
            layer.gpm_statistics='raw';layer.gpm_update_rule=rule;layer.gpm_sample_columns=32
            layer.dual_gpms['global'].max_basis=2
            old=torch.eye(18)[:,:2]
            layer.dual_gpms['global'].basis=old.clone()
            layer.update_gpm(x,0);sizes.append(layer.gpm_column_history)
            if rule=='bounded_preserving':
                new=layer.dual_gpms['global'].basis
                self.assertLess(float((old-new@(new.T@old)).norm()),1e-5)
                self.assertEqual(new.shape[1],2)
        self.assertEqual(sizes[0],sizes[1])

    def test_frequency_orthogonality_need_not_protect_raw_patches(self):
        # Haar maps a checkerboard to a constant HH band. Constant 3x3 HH
        # patches and alternating-sign 3x3 image patches have different spans.
        layer=ODRLoRAConv2d(1,1,3,spatial_blocks=(2,2),freq_level=2)
        axis=torch.arange(64)
        x=((axis[:,None]+axis[None,:])%2*2-1).float()[None,None]
        freq=torch.cat([h for _,_,h in layer._extract_statistics(x)],1)
        m=fit_basis(freq,1)
        raw=F.unfold(x,3,padding=0)[0,:,100:101]
        self.assertGreater(coverage(m,freq),.99)
        self.assertLess(coverage(m,raw),.03)
        delta=raw.T-(raw.T@m)@m.T
        self.assertLess(float((delta@m).norm()),1e-5)
        self.assertGreater(float((delta@raw).abs().item()),8.)


if __name__=='__main__':unittest.main()
