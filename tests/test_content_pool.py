"""ContentPool 的 GAP 初始化、训练梯度和混合精度回归检查。

运行：python -m unittest discover -s tests -p 'test_content_pool.py'
"""

import math
import unittest

import torch

from model.attn_pool import ContentPool


class ContentPoolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_initial_output_matches_gap_on_noncontiguous_features(self):
        pool = ContentPool(8, rank=3)
        feat = torch.randn(2, 8, 3, 5).transpose(2, 3)
        for training in (True, False):
            pool.train(training)
            torch.testing.assert_close(pool(feat), feat.mean(dim=(2, 3)))
            self.assertAlmostEqual(pool._last_entropy, math.log(15), places=5)

    def test_attention_can_leave_gap_and_all_parameters_can_learn(self):
        pool = ContentPool(8, rank=3)
        feat = torch.randn(2, 8, 3, 5, requires_grad=True)
        optimizer = torch.optim.SGD(pool.parameters(), lr=0.5)

        pool(feat).square().mean().backward()
        self.assertGreater(pool.q.grad.norm().item(), 0)
        self.assertEqual(pool.wk.weight.grad.norm().item(), 0)
        self.assertGreater(feat.grad.norm().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        output = pool(feat)
        self.assertGreater((output - feat.mean(dim=(2, 3))).abs().max().item(), 1e-6)
        output.square().mean().backward()
        for parameter in pool.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)

    def test_single_position_is_identity(self):
        pool = ContentPool(8, rank=1)
        feat = torch.randn(2, 8, 1, 1)
        with torch.no_grad():
            pool.q.fill_(3)
        torch.testing.assert_close(pool(feat), feat[:, :, 0, 0])
        self.assertEqual(pool._last_entropy, 0)

    def test_float64_precision_is_preserved(self):
        pool = ContentPool(8, rank=3).double()
        feat = torch.randn(2, 8, 3, 5, dtype=torch.float64)
        output = pool(feat)
        self.assertEqual(output.dtype, feat.dtype)
        torch.testing.assert_close(output, feat.mean(dim=(2, 3)), rtol=1e-12, atol=1e-12)

    def test_bfloat16_autocast_has_finite_output_entropy_and_gradients(self):
        for low_precision_input in (False, True):
            with self.subTest(low_precision_input=low_precision_input):
                pool = ContentPool(8, rank=3)
                feat = torch.randn(2, 8, 3, 5)
                if low_precision_input:
                    pool = pool.bfloat16()
                    feat = feat.bfloat16()
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    initial_output = pool(feat)
                self.assertEqual(initial_output.dtype, feat.dtype)
                torch.testing.assert_close(initial_output, feat.mean(dim=(2, 3)))
                with torch.no_grad():
                    pool.q.fill_(1000)
                    pool.wk.weight.fill_(1)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    output = pool(feat)
                self.assertEqual(output.dtype, feat.dtype)
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(math.isfinite(pool._last_entropy))
                output.float().square().mean().backward()
                for parameter in pool.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_dimensions_have_clear_errors(self):
        for channels, rank in ((0, 3), (8, 0), (-1, 3), (8, -1)):
            with self.subTest(channels=channels, rank=rank):
                with self.assertRaises(ValueError):
                    ContentPool(channels, rank)
        pool = ContentPool(8, rank=3)
        for shape in ((2, 8, 3), (2, 7, 3, 5), (2, 8, 0, 5), (2, 8, 3, 0)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    pool(torch.empty(shape))

    def test_backbone_global_pool_integration(self):
        from model.backbone import build_backbone

        model = build_backbone("resnet18", pretrained=False, content_pool=True,
                               content_rank=4).eval()
        self.assertIsInstance(model.global_pool, ContentPool)
        feat = torch.randn(2, model.num_features, 3, 5)
        output = model.forward_head(feat)
        torch.testing.assert_close(output, feat.mean(dim=(2, 3)))
        output.square().mean().backward()
        self.assertGreater(model.global_pool.q.grad.norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
