# Supervisor handover: televiewer proof of principle

The implementation and saved results are ready for technical review. Field accuracy has not been established.

## Delivered

# RGL televiewer proof of principle

This project replaces the earlier exploratory script with a reproducible classical computer-vision baseline for borehole televiewer analysis. It detects **unclassified planar sinusoidal candidates**, not confirmed fractures. The implemented sequence is:

image preparation -> candidate-pixel evidence -> sinusoidal detection -> robust fitting -> geometric parameters -> optional reference comparison

The default RGL analysis remains a classical computer-vision baseline. A trainable U-Net segmentation pipeline is also implemented and can be connected directly to the sinusoidal-fitting stage. However, no RGL-trained, field-validated U-Net model has yet been established.

This is not a complete reproduction of Al-Sit's full texture-segmentation architecture, and it is not a field-validated geological interpretation system.

## Start here

From PowerShell, in this folder:

~~~powershell
python -m pip install -r requirements.txt
python -B -m unittest discover -v
python -B benchmark.py
python -B televiewer.py --channels optical acoustic
python -B review_results.py
~~~

If the optional U-Net route will be used, install the additional ML dependencies listed in the U-Net requirements file or guide.

Open **results/review.html** for a searchable candidate table, quality meters, downloadable reviewer decisions and expandable page images. The simpler **results/index.html** links directly to all saved plots. Rebuild `review.html` after each new analysis.

A shorter run:

~~~powershell
python -B televiewer.py --boreholes BH01 --channels optical --output trial_BH01
~~~

Each per-track output directory contains:

- `candidates.csv`: fitted native-pixel parameters, provisional depth, support statistics, fit residuals and any available physical geometry.
- `evidence.npz`: detector evidence, valid-pixel mask and annotation mask.
- `page_NNN.png`: paginated visual review showing the original image, detector evidence and fitted candidate curves.

If a U-Net checkpoint is configured, additional files are saved:

- `segmentation_probability.npy`
- `segmentation_mask.png`
- `segmentation_model.json`

The run root contains:

- `all_candidates.csv`
- `summary.csv`
- `config_used.json`
- `run_manifest.json`
- `index.html`

Use a fresh `--output` directory for experiments because output files are overwritten and old pages from another run are not automatically deleted.

**Candidate counts are not fracture counts.** A low count can mean missed features; a high count can mean false detections.

## Findings in the old code

1. A single crop layout was applied to different reports. BH02's optical track begins at column 54; the old crop began at 103 and mixed optical/acoustic information. BH01's old acoustic crop also extended into the dip-plot area.
2. A fixed 100-row header removal left large blank report intervals in the analysis. Those pixels then influenced the global darkest-20-percent threshold.
3. A minimum of eight votes was too permissive. With a 20% dark-pixel mask, a random 150-column curve would already be expected to overlap many dark pixels before searching over amplitude and phase.
4. The maximum amplitude of 15% of image width was an arbitrary restriction and could miss steeper traces.
5. Depth-first suppression kept the first acceptable nearby candidate rather than selecting the globally best-supported curve.
6. The old dip proxy assumed a pixel aspect ratio that is not physically valid for the report images. Simply stretching the report vertically changed the apparent geology.
7. The BH01 acoustic track and BH02 travel-time track already contain coloured expert picks. Detecting those overlays cannot demonstrate independent automatic interpretation accuracy.

## What the new classical implementation does

- Uses explicit, size-checked layouts for each report and preserves original report-row coordinates.
- Excludes white report margins, headers and invalid image boundaries.
- Masks the known saturated annotation palette on annotated tracks.
- Uses nearest-neighbour filling only for filtering; overwritten annotation pixels never contribute evidence.
- Uses local multiscale dark/bright ridge responses instead of selecting a fixed percentage of dark pixels.
- Uses both dark and bright ridge evidence for optical images.
- Uses dark-ridge evidence for non-optical rendered tracks by default.
- Uses structure-tensor ridge direction to reject locally incompatible evidence.
- Uses generalized sinusoidal Hough voting to generate candidate curves.
- Refines candidates using robust soft-L1 least-squares fitting.
- Requires angular coverage, multiple supported sectors, bounded gaps, connected arcs, acceptable residual error and support above nearby traces.
- Processes long logs using overlapping depth tiles.
- Suppresses near-duplicate whole curves rather than simply suppressing nearby centre rows.
- Allows crossing curves at similar depths where their fitted geometry is sufficiently different.
- Flags near-horizontal candidates for manual review because acquisition or casing artefacts are possible.
- Exports an explicit **uncalibrated quality score**. It is a ranking statistic, not a probability of geological correctness.
- Supports one-to-one reference matching inside fully reviewed intervals.
- Does not fabricate missing ground-truth values.

The defaults remain exploratory. They were selected using synthetic checks and visual review of the supplied reports. They are **not held-out field-performance estimates**.

Weak or irregular optical traces, closely spaced layers, severe occlusion, broad acoustic textures, steep traces beyond the configured amplitude range and acquisition artefacts remain difficult.

## Image layout and calibration

All x bounds use zero-based pixels and half-open intervals `[x0, x1)`.

Never resize a report image without updating the crop layout and depth calibration.

| Borehole | Optical x bounds | Acoustic x bounds | Header end row |
|---|---|---|---|
| BH01 | 103:254 | 255:406 | 113 |
| BH02 | 54:212 | 213:371 | 110 |
| BH03 | 107:292 | 293:477 | 973 |

Depth calibration currently uses two visually read tick marks per report. This is provisional, especially over long raster exports.

- BH01: row 224 = 0.4 m; row 336 = 0.8 m.
- BH02: row 518 = 0.0 m; row 630 = 0.4 m.
- BH03: row 1490 = 2.4 m; row 1602 = 2.8 m.

These values should eventually be replaced with original depth samples or independently verified report calibration.

An azimuth period is assumed to equal the full native track width unless `endpoint=true` is explicitly configured for a duplicated final azimuth column. The image seam and azimuth direction should be verified against the original export.

Travel-time tracks are configured for BH02 and BH03 but are not included when the detector is run with:

~~~powershell
python -B televiewer.py --channels optical acoustic
~~~

Running without `--channels` can also process the configured travel-time tracks as exploratory rendered-image inputs.

Rendered travel-time colours are not calibrated physical travel-time measurements, so this is not a validated aperture or ovalisation workflow.

## Geometry: when a dip angle is justified

The fitted curve is represented as:

~~~text
row(theta) = center + s*sin(theta) + c*cos(theta)

amplitude_px = hypot(s, c)

phase = atan2(c, s)

depth_m = row * metres_per_pixel + intercept

amplitude_m = amplitude_px * metres_per_pixel

borehole_relative_dip = atan(amplitude_m / radius)

radius = diameter_m / 2
~~~

Amplitude is the half peak-to-trough vertical displacement of the fitted sinusoid.

Set `diameter_m` in `rgl_config.json` only using verified borehole or caliper information, in metres. The program does not infer borehole radius from report width.

Without a verified diameter:

- `borehole_relative_dip_deg` remains empty.
- amplitude and phase can still be reported in fitted image coordinates.

The borehole-relative dip is measured relative to a plane perpendicular to the borehole axis. It is not automatically equal to geographic dip.

For geographic values, provide `image_frame_to_ned`, a proper 3x3 rotation whose columns represent:

- image theta = 0 radial direction,
- image theta = 90° radial direction,
- downhole axis,

expressed in North/East/Down coordinates.

The program verifies that the supplied matrix is a proper orthonormal rotation before using it.

Do not apply tool-orientation corrections twice if the report has already been geometrically corrected.

A simple identity rotation is only valid for a verified vertical borehole with the intended North/East convention.

Magnetic North should not automatically be treated as true North.

Phase is not automatically geological azimuth. With depth increasing downward, the deepest point of the fitted sinusoid depends on the phase convention. Geographic dip direction is only reported after the image frame has been verified.

## Reference validation and the feedback loop

Request the following from RGL or the supervisor:

1. Clean optical and acoustic exports without drawn expert picks.
2. Original depth and azimuth samples where available.
3. Expert picks in CSV containing depth, dip, azimuth, feature class and corresponding borehole/channel.
4. Borehole diameter/caliper data.
5. Borehole deviation/orientation survey information.
6. Image seam convention and confirmation of whether North is magnetic or true.
7. Fully reviewed depth intervals.
8. Core photography or laboratory references where available.
9. Required error tolerances.
10. Confirmation of whether bedding, veins and different joint classes belong to the target feature set.

For direct curve validation, one reference CSV per track can use:

~~~csv
feature_id,center_row_px,sin_coefficient_px,cos_coefficient_px,feature_type
~~~

These coordinates must refer to the original report and the local image track, not resized review plots.

Optional fields may include:

- `depth_m`
- `borehole_relative_dip_deg`
- `true_dip_deg`
- `true_dip_azimuth_deg`

If physical geometry is not independently supplied, agreement in derived angles only checks consistency of curve fitting and shared calibration.

### Creating expert reference curves from manually selected points

Create:

~~~csv
feature_id,x_px,row_px,feature_type
~~~

Use at least six well-distributed points around the borehole circumference for each feature.

Then run:

~~~powershell
python -B fit_reference.py --points expert_points.csv --width 151 --output bh01_truth.csv
~~~

Inspect `reference_fit_rmse_px` before treating the fitted curve as reliable ground truth.

A clean track can then be configured with:

~~~json
{
  "truth_csv": "bh01_truth.csv",
  "reviewed_intervals": [[1500, 2500]],
  "match_tolerance_px": 5
}
~~~

Every target feature whose centre lies within a reviewed interval must be labelled for precision/recall evaluation to be meaningful.

The evaluator:

- restricts predictions and truth to fully reviewed intervals;
- performs one-to-one matching;
- reports TP, FP and FN;
- reports precision, recall and F1;
- reports trace MAE and available parameter MAEs;
- handles circular phase/azimuth errors correctly;
- reports matched-value counts for each MAE;
- leaves unavailable metrics blank rather than reporting false zero errors.

Independent validation is refused on tracks marked as annotated.

Use separate boreholes for development and final testing. Do not tune thresholds on one part of a report and then claim an adjacent crop from the same borehole is an independent test set.

The same separation should be used for any later classifier or U-Net.

The synthetic benchmark is a software verification tool and does **not** replace expert RGL or core validation.

## U-Net segmentation pipeline

A trainable U-Net pipeline is now implemented in `unet_segmentation.py`.

It includes:

- a three-level encoder/decoder U-Net;
- skip connections;
- periodic padding across the azimuth seam;
- RGB input plus an explicit validity channel;
- BCE + Dice segmentation loss;
- full-width depth tiling;
- simple augmentation;
- train/validation/test manifest handling;
- prevention of borehole leakage between splits;
- prevention of identical-image leakage between splits;
- refusal to train on report images marked as containing expert-pick overlays;
- validation using Dice, IoU, precision and recall;
- checkpoint provenance and file hashes;
- overlapping-tile prediction;
- conversion of thick segmentation probabilities into thin ridge evidence for the Hough stage;
- synthetic-data generation for software testing.

A synthetic-only checkpoint can be used for demonstrations only when explicitly allowed. It must not be presented as RGL-trained or field-validated.

### Direct U-Net integration in `televiewer.py`

A clean track can specify:

~~~json
{
  "unet_checkpoint": "path/to/best.pt",
  "unet_device": "cpu"
}
~~~

The main detector then performs:

~~~text
clean televiewer track
        ↓
U-Net probability map
        ↓
binary segmentation mask
        ↓
thin ridge evidence
        ↓
sinusoidal Hough candidate generation
        ↓
robust curve fitting and geometric checks
~~~

When a U-Net checkpoint is used, the per-track output can include:

- `segmentation_probability.npy`
- `segmentation_mask.png`
- `segmentation_model.json`

The sinusoidal geometry stage still applies validity masks, orientation checks, continuity checks, fitting and duplicate suppression.

U-Net and external `evidence_npy` input cannot be selected simultaneously for the same track.

The U-Net route is refused on tracks configured as annotated because expert overlays would leak reference information into the segmentation stage.

See `UNET_GUIDE.md` for the training and integration workflow if it is included in the handover package.

## External segmentation-evidence interface

The earlier `evidence_npy` interface is still supported for segmentation methods other than the included U-Net.

A track may specify an `evidence_npy` file containing a finite, non-negative HxW NumPy array in native report coordinates.

The detector then uses that supplied evidence while still applying:

- validity masking;
- sinusoidal fitting;
- orientation checks;
- continuity checks;
- duplicate suppression;
- optional expert validation.

This provides an interface for future Gabor segmentation, alternative neural networks or another image-processing method without rewriting the downstream geometry stage.

## Sources and scope

Primary project requirements are based on:

- supplied Y+A RGL notes;
- supplied borehole-analysis flowchart;
- supplied Waleed Al-Sit thesis;
- supplied Al-Sit et al. televiewer-image paper;
- supplied Cruz et al. deep-learning paper;
- RGL televiewer information supplied during the project.

The current implementation uses local ridge evidence and a structure tensor rather than reproducing Al-Sit's complete Gabor/clustering pipeline.

The supplied literature motivates later texture segmentation and learned candidate rejection, but the current RGL results should still be treated as exploratory.

Older dip histograms from the exploratory script should not be presented as verified physical dip measurements.

## Validation delivered with this version

The project includes deterministic regression tests and synthetic verification.

The development benchmark and a separate frozen-parameter synthetic set contain known sinusoidal traces and are intended to verify that the implementation behaves consistently on controlled inputs.

These synthetic scenes are substantially simpler than real RGL logs.

Therefore synthetic performance must **not** be presented as field accuracy.

The complete reference-CSV workflow is also exercised in the verification material.

Reproduce the separate synthetic benchmark where the relevant files are included:

~~~powershell
python -B benchmark.py --seed-start 2000 --output synthetic_holdout
~~~

The interactive review dashboard can export accept/reject/uncertain decisions for detected candidates.

These reviewer decisions do not constitute complete ground truth because missed features still need to be independently labelled.

## Saved RGL results from the current classical baseline

These counts represent accepted geometric candidates, not confirmed fractures.

No verified physical dip values are reported because borehole diameter/orientation metadata is not yet established.

| Borehole | Track | Candidates | Annotated input |
|---|---|---:|---|
| BH01 | acoustic | 8 | True |
| BH01 | optical | 8 | False |
| BH02 | acoustic | 0 | False |
| BH02 | optical | 20 | False |
| BH03 | acoustic | 692 | False |
| BH03 | optical | 1044 | False |

### BH02 acoustic limitation

The saved BH02 acoustic-amplitude analysis accepts zero candidates.

This should **not** be interpreted as evidence that the borehole contains no relevant features.

The coarse/broad acoustic texture is poorly represented by the current narrow-ridge classical baseline.

This suggests that broad-boundary or learned texture segmentation should be investigated using clean original acoustic-amplitude data.

### BH03 limitation

BH03 produces very high candidate counts in both optical and acoustic tracks.

This indicates that the classical detector is still responding strongly to repeated geological texture and/or acquisition artefacts.

These candidates require expert review and properly labelled validation intervals.

Thresholds should not simply be tuned until the candidate count appears visually reasonable.

## Engineering verification

Where the complete project package is supplied, it includes:

- deterministic regression tests;
- synthetic development benchmark;
- separate synthetic holdout benchmark;
- end-to-end CLI testing;
- reference-CSV validation;
- source and input hashes;
- saved configuration;
- run manifest;
- review dashboard.

Synthetic tests cover controlled narrow dark/bright traces, noise, partial traces and crossings.

They do not reproduce the full complexity of field televiewer data.

## Required before any accuracy claim

Before claiming field accuracy, obtain:

- clean original image/amplitude exports;
- complete expert pick CSVs;
- verified depth calibration;
- verified borehole diameter/caliper data;
- verified borehole orientation/deviation information;
- fully reviewed intervals;
- supervisor/RGL matching tolerances;
- core/lab references where available.

Development and held-out testing should remain borehole-disjoint.

Fit-quality scores are not calibrated probabilities.

## Current project status

The project currently provides:

- a reproducible classical televiewer-image baseline;
- report-specific crop/calibration configuration;
- missing-data and known-annotation masking;
- multiscale ridge evidence;
- direction-aware sinusoidal Hough voting;
- robust curve fitting;
- continuity and duplicate checks;
- provisional depth conversion;
- physical geometry gated by verified calibration;
- candidate CSV outputs;
- evidence arrays;
- paginated review plots;
- interactive review support;
- expert-point fitting;
- one-to-one reference evaluation;
- a trainable U-Net segmentation implementation;
- direct U-Net-to-Hough integration;
- an external segmentation-evidence interface.

It does **not** yet provide validated:

- six-class geological feature classification;
- aperture;
- ovalisation;
- RQD;
- fracture-network interpretation;
- rock-type classification;
- field-calibrated confidence probabilities;
- true dip without verified diameter;
- geographic dip/azimuth without verified orientation;
- demonstrated RGL field accuracy.

The current code should therefore be described as a **reviewable proof of principle and research baseline**, ready for the missing RGL calibration and independent labels.
