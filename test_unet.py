"""Regression checks for U-Net training, masking, inference and the Hough interface."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from unet_segmentation import UNet,masked_loss,predict,probability_to_evidence,load_manifest
from televiewer import detect,curve,params,ridge_orientation

class UNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_output_matches_odd_image_shape(self):
        model=UNet(4)
        x=torch.rand(1,4,65,71)
        self.assertEqual(model(x).shape,(1,1,65,71))

    def test_ignored_labels_do_not_affect_loss_or_gradient(self):
        logits=torch.zeros(1,1,16,16,requires_grad=True)
        target=torch.zeros_like(logits); valid=torch.ones_like(logits)
        valid[:,:,:4]=0
        a=masked_loss(logits,target,valid)
        changed=target.clone(); changed[:,:,:4]=1
        b=masked_loss(logits,changed,valid)
        self.assertAlmostEqual(float(a.detach()),float(b.detach()),places=6)
        a.backward()
        self.assertEqual(float(logits.grad[:,:,:4].abs().sum()),0.)

    def test_optimizer_reduces_loss_on_one_example(self):
        torch.manual_seed(30)
        m=UNet(4)
        x=torch.rand(1,4,32,40)
        target=torch.zeros(1,1,32,40); target[:,:,13:17]=1
        valid=torch.ones_like(target)
        opt=torch.optim.Adam(m.parameters(),lr=.005)
        before=float(masked_loss(m(x),target,valid).detach())
        for _ in range(12):
            opt.zero_grad()
            loss=masked_loss(m(x),target,valid); loss.backward(); opt.step()
        after=float(masked_loss(m(x),target,valid).detach())
        self.assertLess(after,before*.9)

    def test_tiled_inference_covers_tail_and_masks_missing_pixels(self):
        class Constant(torch.nn.Module):
            def forward(self,x):
                return torch.zeros(x.shape[0],1,*x.shape[-2:],device=x.device)
        rgb=np.full((113,47,3),100,np.uint8); valid=np.ones((113,47),bool)
        valid[30:40,5:10]=False
        p=predict(Constant(),rgb,valid,48,12)
        np.testing.assert_allclose(p[valid],.5,atol=1e-6)
        self.assertTrue(np.all(p[~valid]==0))
        self.assertEqual(p.shape,valid.shape)

    def test_probability_mask_reaches_sinusoidal_hough(self):
        h,w=220,96; y=np.arange(h)[:,None]
        truth=[110.,19.,-8.]
        p=np.exp(-.5*((y-curve(truth,w))/1.2)**2).astype(np.float32)
        valid=np.ones((h,w),bool)
        ev,mask=probability_to_evidence(p,valid)
        self.assertLess(np.count_nonzero(ev),mask.sum())
        ds=detect(ev,valid,{"max_amplitude_px":40},orientation=ridge_orientation(p,valid))
        self.assertEqual(len(ds),1)
        self.assertLess(np.mean(np.abs(curve(params(ds[0]),w)-curve(truth,w))),1.)

    def test_borehole_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.fromarray(np.zeros((16,16,3),np.uint8)).save(root/"image.png")
            Image.fromarray(np.zeros((16,16),np.uint8)).save(root/"mask.png")
            samples=[dict(borehole="BH01",channel="optical",split=s,image="image.png",mask="mask.png") for s in ("train","val")]
            (root/"manifest.json").write_text(json.dumps(dict(dataset_kind="rgl",samples=samples)))
            with self.assertRaisesRegex(ValueError,"Borehole leakage"):
                load_manifest(root/"manifest.json")

    def test_annotated_training_input_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"manifest.json"
            path.write_text(json.dumps(dict(dataset_kind="rgl",samples=[
                dict(borehole="BH01",channel="acoustic",split="train",annotated_input=True)])))
            with self.assertRaisesRegex(ValueError,"expert-pick"):
                load_manifest(path)

    def test_border_augmentation_ignores_labels_and_zeros_input(self):
        from unittest.mock import patch
        from unet_segmentation import TraceDataset
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.fromarray(np.full((32,40,3),100,np.uint8)).save(root/"image.png")
            Image.fromarray(np.full((32,40),255,np.uint8)).save(root/"mask.png")
            ds=TraceDataset([dict(image=str(root/"image.png"),mask=str(root/"mask.png"))],32,True)
            with patch("unet_segmentation.random.randrange",return_value=3):
                x,y,v=ds[0]
            self.assertTrue(torch.all(x[:,:3]==0))
            self.assertTrue(torch.all(x[:,-3:]==0))
            self.assertTrue(torch.all(v[:,:3]==0))
            self.assertTrue(torch.all(y[:,:3]==1))

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            probability_to_evidence(np.full((16,16),np.nan),np.ones((16,16),bool))

if __name__=="__main__":
    unittest.main(verbosity=2)

