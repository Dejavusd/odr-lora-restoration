import unittest
import torch
from src.models.dual_gpm import DualGPM


class BalancedMemoryTests(unittest.TestCase):
    def memory(self):
        m=DualGPM(6,max_basis=4)
        m.basis=torch.eye(6)[:,:4]
        h=torch.diag(torch.tensor([1.,2.,3.,4.,9.,8.]))
        return m,h

    def test_half_quota_orthogonal_capacity_and_current_modes(self):
        m,h=self.memory();m.accumulate_balanced(h);info=m.finalize_balanced(.5)
        self.assertEqual(info['retained_old_rank'],2)
        self.assertEqual(info['current_residual_rank'],2)
        self.assertEqual(m.basis.shape,(6,4))
        torch.testing.assert_close(m.basis.T@m.basis,torch.eye(4),atol=1e-6,rtol=1e-6)
        # Lowest-current-energy old directions e0/e1 plus current e4/e5.
        expected=torch.eye(6)[:,[0,1,4,5]]
        torch.testing.assert_close(m.basis@m.basis.T,expected@expected.T,atol=1e-6,rtol=1e-6)

    def test_full_old_quota_preserves_old_span(self):
        m,h=self.memory();old=m.basis.clone();m.accumulate_balanced(h);i=m.finalize_balanced(1.)
        self.assertEqual(i['current_residual_rank'],0)
        self.assertLess(float((old-m.basis@(m.basis.T@old)).norm()),1e-6)

    def test_aggregate_order_and_old_basis_rotation_invariant(self):
        a,h=self.memory();b,_=self.memory()
        q=torch.linalg.qr(torch.randn(4,4)).Q;b.basis=b.basis@q
        a.accumulate_balanced(h[:,:3]);a.accumulate_balanced(h[:,3:])
        b.accumulate_balanced(h[:,3:]);b.accumulate_balanced(h[:,:3])
        ia=a.finalize_balanced(.5);ib=b.finalize_balanced(.5)
        self.assertEqual(ia['covariance_sha256'],ib['covariance_sha256'])
        torch.testing.assert_close(a.basis@a.basis.T,b.basis@b.basis.T,atol=1e-5,rtol=1e-5)

    def test_zero_data_keeps_old_memory_and_invalid_quota_rejected(self):
        m,h=self.memory();old=m.basis.clone()
        m.accumulate_balanced(h*0);info=m.finalize_balanced(.5)
        self.assertTrue(info['no_current_energy'])
        torch.testing.assert_close(m.basis,old)
        m.accumulate_balanced(h)
        with self.assertRaises(ValueError):m.finalize_balanced(1.2)
        with self.assertRaises(FloatingPointError):m.accumulate_balanced(h*float('nan'))


if __name__=='__main__':unittest.main()
