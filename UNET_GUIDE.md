# U-Net segmentation followed by sinusoidal Hough fitting

Both stages are implemented. The default rgl_config.json still selects the classical baseline because no RGL-trained checkpoint is available. The existing results/ folder contains those classical results.

The learned route is:

**Clean full-circumference image -> U-Net pixel scores -> binary mask -> skeleton -> sinusoidal Hough voting -> robust curve fit -> calibrated geometry -> reference validation.**

## Preferred entry point: televiewer.py

Keep the original televiewer.py Hough detector and curve fitting. U-Net replaces its segmentation evidence, with no changes to those fitting rules. FINAL_BOREHOLE_ANALYSIS.py is a separate experimental detector and is not the preferred baseline.

From the workspace root:

```powershell
python -B televiewer_pipeline/televiewer.py
python -B televiewer_pipeline/televiewer.py --channels optical --unet-checkpoint RESULTS/models/rgl_optical/best.pt
```

The second command requires an actual RGL-trained optical checkpoint at the given example path. A missing, synthetic-only, wrong-modality or annotated-input selection is rejected explicitly. Without a checkpoint the original classical baseline runs. Per-track checkpoint configuration remains supported for separate optical/acoustic models.

Default analysis outputs now go into a dated RESULTS/televiewer_* folder. To train into RESULTS, use:

```powershell
python -B televiewer_pipeline/unet_segmentation.py train --manifest labels/manifest.json --output RESULTS/models/rgl_optical --epochs 50
```

The manifest must contain real expert labels, as described below. The complete supplied synthetic U-Net-to-Hough demonstration was rerun in RESULTS/televiewer_unet_verified. Nine U-Net regression tests passed. Synthetic results establish integration, not field accuracy.

## Implementation

- unet_segmentation.py: trainable PyTorch U-Net, dataset loading, training, checkpoint loading, tiled inference, pixel evaluation and mask thinning.
- televiewer.py: directly calls the U-Net when a track has unet_checkpoint, then feeds the thinned segmentation to seeds()/detect() and robust fitting.
- test_unet.py: learning, dimensions, ignored-label gradients, inference coverage, split leakage and the Hough interface.
- requirements-unet.txt: optional dependencies, in addition to the classical pipeline.

The U-Net has three encoder/decoder levels, skip connections, GroupNorm and a single output logit per pixel. Input is RGB in [0,1] plus a validity channel. Convolutions wrap horizontally around the borehole seam and replicate at depth boundaries. Training uses binary cross-entropy with logits plus Dice loss, excluding unknown labels. It uses cyclic azimuth and masked-border augmentation, AdamW, gradient clipping, validation checkpoint selection and early stopping.

Inference blends overlapping depth tiles at native pixel dimensions. Sigmoid outputs are model scores, not calibrated geological confidence. Thresholding and periodic skeletonization turn the segmentation into thin Hough evidence. The Hough transform searches sinusoids, not straight lines.

## Run the complete saved synthetic demonstration

From the televiewer_pipeline folder:

~~~powershell
python -m pip install -r requirements-unet.txt
python -B televiewer.py --config unet_demo/pipeline_config.json --output unet_demo/results
python -B unet_segmentation.py evaluate --manifest unet_demo/data/manifest.json --checkpoint unet_demo/model_v2/best.pt --split test --allow-synthetic --output unet_demo/test_metrics.json
~~~

Open unet_demo/results/index.html. Each track contains:
- segmentation_probability.npy: full-resolution model scores.
- segmentation_mask.png: thresholded segmentation.
- segmentation_model.json: checkpoint hash and training provenance.
- evidence.npz: skeleton evidence passed to Hough, plus validity masks.
- candidates.csv, validation.json and page images.

The supplied demo checkpoint was trained on synthetic narrow dark traces, with separate synthetic train/validation/test groups. It is not trained on RGL geology. The demo explicitly enables allow_synthetic_checkpoint. Do not interpret its performance as field accuracy. Demo curve references are least-squares fits to the known binary target masks, so they also include pixel quantization.

## Train a model using expert RGL masks

Use CLEAN cropped image tracks containing a full azimuth revolution. Masks must have the same dimensions:
- 0: reviewed background.
- 255: the target feature pixels.
- 128: unknown or ignored pixels, including unreviewed regions.

Define the target consistently with the supervisor: visible fracture traces, all planar traces, or some other explicitly labelled class. Bedding and veins are not automatically fractures. Training currently implements binary segmentation, not six-class fracture classification.

Create a manifest like this (paths are relative to the manifest):

~~~json
{
  "dataset_kind": "rgl",
  "label_definition": "Expert-labelled visible fracture traces",
  "samples": [
    {
      "borehole": "BH_TRAIN",
      "channel": "optical",
      "split": "train",
      "image": "images/train_optical.png",
      "mask": "masks/train_optical.png",
      "annotated_input": false
    },
    {
      "borehole": "BH_VAL",
      "channel": "optical",
      "split": "val",
      "image": "images/val_optical.png",
      "mask": "masks/val_optical.png",
      "annotated_input": false
    },
    {
      "borehole": "BH_TEST",
      "channel": "optical",
      "split": "test",
      "image": "images/test_optical.png",
      "mask": "masks/test_optical.png",
      "annotated_input": false
    }
  ]
}
~~~

These are illustrative identifiers and paths, not supplied labelled RGL files.

~~~powershell
python -B unet_segmentation.py train --manifest labels/manifest.json --output models/rgl_optical --epochs 50 --tile-rows 256 --base 16 --device cpu
python -B unet_segmentation.py evaluate --manifest labels/manifest.json --checkpoint models/rgl_optical/best.pt --split test --output models/rgl_optical/test_metrics.json
~~~

Use --device cuda only with a suitable installed GPU build. Training defaults to batch size one and supports different native image widths. Closely related crops and optical/acoustic samples from the SAME borehole must remain in the SAME split. The loader checks borehole IDs and duplicate image hashes. It cannot infer disguised related boreholes or verify that supplied expert labels are correct.

Train/validate optical and acoustic models separately unless a jointly trained dataset demonstrably supports both. A checkpoint is rejected for a modality absent from its training provenance.

## Enable U-Net for an RGL track

Copy the RGL configuration to a new experiment config and add these keys to the clean target track:

~~~json
{
  "name": "optical",
  "x_bounds": [103, 254],
  "annotated": false,
  "unet_checkpoint": "models/rgl_optical/best.pt",
  "unet_device": "cpu"
}
~~~

The checkpoint path is relative to the experiment configuration. Do not add allow_synthetic_checkpoint for an RGL-trained model. Configure only modalities with a corresponding trained checkpoint; other tracks continue using the classical route.

~~~powershell
python -B televiewer.py --config rgl_unet_config.json --channels optical --output results_unet
python -B review_results.py --results results_unet
~~~

Choose either unet_checkpoint or evidence_npy for a track, not both. U-Net input must not contain drawn expert picks. Checkpoints do not provide missing depth/diameter/orientation calibration.

The BH01 acoustic report has expert overlays, while BH02's expert overlays are on its travel-time track. Use clean original exports for supervised training and independent evaluation.

## Reproduce the synthetic training demonstration

~~~powershell
python -B unet_segmentation.py synthetic-data --output new_unet_demo/data
python -B unet_segmentation.py train --manifest new_unet_demo/data/manifest.json --output new_unet_demo/model --epochs 30 --base 8 --tile-rows 96 --threads 2
python -B unet_segmentation.py evaluate --manifest new_unet_demo/data/manifest.json --checkpoint new_unet_demo/model_v2/best.pt --split test --allow-synthetic --output new_unet_demo/test_metrics.json
~~~

This demonstrates actual learning and evaluation; it does not replace labelled RGL training. The earlier 40/40 synthetic benchmark in README.md concerns the classical detector, not this U-Net model.

## References

- Ronneberger et al., U-Net: https://arxiv.org/abs/1505.04597 .
- PyTorch binary cross-entropy with logits: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html .
- Supplied Al-Sit paper and RGL notes: motivation for segmenting useful pixels before sinusoidal Hough detection.


## Verified demonstration results

All 25 regression tests pass. The new synthetic checkpoint is selected at epoch 26 of a 30-epoch run using validation Dice. On the four synthetic test scenes, pixel Dice is 0.9861 and IoU is 0.9726. The complete pipeline matches all four mask-derived reference traces with no extra curves. See [verification.json](unet_demo/verification.json), [demo results](unet_demo/results/index.html), and [the visual comparison](unet_demo/segmentation_to_hough.png).

These scenes were also used to diagnose the initial border-mask integration issue. Treat these as engineering demonstration results, not an unbiased generalization estimate. Independent labelled RGL test boreholes remain necessary.
