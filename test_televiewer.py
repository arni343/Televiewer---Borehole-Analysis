"""Deterministic regression checks; synthetic success is not field accuracy."""
import unittest
import numpy as np
from televiewer import (prepare, detect, params, curve, geometry, evaluate,
                        circular_gap, calibration, deduplicate, ridge_orientation)

def synthetic(parameters, h=420, w=120, seed=8, gap=None):
    rng=np.random.default_rng(seed)
    y=np.arange(h)[:,None]
    image=np.full((h,w),.72)+rng.normal(0,.008,(h,w))
    for p in parameters:
        trace=curve(p,w)
        contrast=.43*np.exp(-.5*((y-trace)/1.1)**2)
        if gap:
            contrast[:,gap[0]:gap[1]]=0
        image-=contrast
    return np.repeat((np.clip(image,0,1)*255).astype(np.uint8)[:,:,None],3,axis=2)

def recover(rgb,settings=None):
    ev,valid,_,gray=prepare(rgb)
    return detect(ev[:1],valid,{"max_amplitude_px":48,"amplitude_step_px":4,
                                "tile_rows":160}|(settings or {}),orientation=ridge_orientation(gray,valid))

class TeleviewerTests(unittest.TestCase):
    def test_blank_and_flat(self):
        for value in (0,128,255):
            self.assertEqual(recover(np.full((220,100,3),value,np.uint8)),[])

    def test_noisy_sinusoid_and_partial_trace(self):
        truth=[180.,24.,16.]
        for gap in (None,(20,43)):
            pred=recover(synthetic([truth],gap=gap))
            errors=[np.mean(np.abs(curve(params(d),120)-curve(truth,120))) for d in pred]
            self.assertTrue(errors)
            self.assertLess(min(errors),1.)
            self.assertEqual(len(pred),1)

    def test_crossing_curves(self):
        truth=[[190.,35.,0.],[190.,-35.,0.]]
        pred=recover(synthetic(truth))
        for p in truth:
            self.assertLess(min(np.mean(np.abs(curve(params(d),120)-curve(p,120))) for d in pred),1.5)
        self.assertEqual(len(pred),2)

    def test_tile_boundary(self):
        pred=recover(synthetic([[160.,23.,-10.]]))
        self.assertEqual(len(pred),1)
        self.assertLess(abs(pred[0]["center_row_px"]-160),1)

    def test_horizontal_bands_do_not_create_waves(self):
        rgb=synthetic([[140.,0,0],[152.,0,0],[163.,0,0],[178.,0,0]])
        pred=recover(rgb)
        self.assertEqual(len(pred),4)
        self.assertTrue(all(d["amplitude_px"]<1 for d in pred))

    def test_random_noise(self):
        rng=np.random.default_rng(172)
        rgb=np.repeat(rng.integers(70,190,size=(420,120),dtype=np.uint8)[:,:,None],3,axis=2)
        self.assertEqual(recover(rgb),[])

    def test_white_gap_never_is_evidence(self):
        rgb=synthetic([[180.,24.,16.]])
        rgb[100:250]=255
        self.assertEqual(recover(rgb),[])

    def test_geometry_known_plane_and_resize(self):
        d=dict(center_row_px=100.,amplitude_px=50.,sin_coefficient_px=0.,cos_coefficient_px=50.)
        s=dict(depth_anchors=[[0,0],[100,1]],diameter_m=1.,image_frame_to_ned=np.eye(3).tolist())
        g=geometry(d,s)
        self.assertAlmostEqual(g["borehole_relative_dip_deg"],45)
        self.assertAlmostEqual(g["true_dip_deg"],45)
        self.assertAlmostEqual(g["true_dip_azimuth_deg"],0)
        d2={k:2*v for k,v in d.items()}
        s2=s|{"depth_anchors":[[0,0],[200,1]]}
        self.assertEqual(g,geometry(d2,s2))
        self.assertIsNone(geometry(d,s|{"diameter_m":None})["borehole_relative_dip_deg"])

    def test_inclined_frame(self):
        t=np.deg2rad(30)
        rotation=[[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]]
        d=dict(center_row_px=0,amplitude_px=0,sin_coefficient_px=0,cos_coefficient_px=0)
        s=dict(depth_anchors=[[0,0],[100,1]],diameter_m=.1,image_frame_to_ned=rotation)
        self.assertAlmostEqual(geometry(d,s)["true_dip_deg"],30)
        self.assertAlmostEqual(geometry(d,s)["true_dip_azimuth_deg"],180)

    def test_matching_one_to_one_and_review_scope(self):
        def d(y):
            return dict(center_row_px=y,sin_coefficient_px=10,cos_coefficient_px=0)
        m=evaluate([d(100),d(102),d(350)],[d(100),d(200)],[[50,250]],100)
        self.assertEqual((m["tp"],m["fp"],m["fn"]),(1,1,1))
        self.assertEqual(m["precision"],.5)

    def test_wrap_gap(self):
        a=np.ones(100,bool); a[:10]=False; a[-10:]=False
        self.assertEqual(circular_gap(a),.2)


    def test_reference_points_recover_known_curve(self):
        from fit_reference import fit_points
        x=np.linspace(0,149,12)
        y=320+21*np.sin(2*np.pi*x/150)-13*np.cos(2*np.pi*x/150)
        d=fit_points(x,y,150)
        self.assertAlmostEqual(d["center_row_px"],320,places=6)
        self.assertAlmostEqual(d["sin_coefficient_px"],21,places=6)
        self.assertAlmostEqual(d["cos_coefficient_px"],-13,places=6)
        with self.assertRaises(ValueError):
            fit_points(np.arange(6),np.arange(6),150)

    def test_annotation_ink_is_excluded(self):
        from televiewer import annotation_mask
        rgb=synthetic([[180.,24.,16.]])
        rgb[170:174,20:95]=[255,0,255]
        ev,valid,ink,_=prepare(rgb,annotated=True)
        self.assertTrue(ink[171,30])
        self.assertFalse(valid[171,30])
        self.assertTrue(np.all(ev[:,~valid]==0))

    def test_invalid_rotation_rejected(self):
        d=dict(center_row_px=100,amplitude_px=50,sin_coefficient_px=0,cos_coefficient_px=50)
        with self.assertRaises(ValueError):
            geometry(d,dict(depth_anchors=[[0,0],[100,1]],diameter_m=.1,
                            image_frame_to_ned=[[2,0,0],[0,1,0],[0,0,1]]))

    def test_undefined_metrics_are_not_zero_error(self):
        d=dict(center_row_px=100,sin_coefficient_px=10,cos_coefficient_px=0)
        m=evaluate([], [d], [[0,200]],100)
        self.assertEqual(m["fn"],1)
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["trace_mae_px"])
        self.assertEqual(m["recall"],0)

    def test_wrong_calibration_rejected(self):
        with self.assertRaises(ValueError):
            calibration({"depth_anchors":[[100,2],[0,1]]})

if __name__=="__main__":
    unittest.main(verbosity=2)


