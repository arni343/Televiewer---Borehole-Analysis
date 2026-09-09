# RGL televiewer proof of principle

This replaces the exploratory script with a reproducible classical computer-vision baseline. It detects **unclassified planar candidates**, not confirmed fractures. It follows the project sequence: image preparation -> candidate pixels -> sinusoid detection -> robust fitting -> geometric parameters -> reference comparison.

The default RGL configuration selects the classical baseline. A trainable U-Net and its direct Hough integration are also implemented; see [UNET_GUIDE.md](UNET_GUIDE.md). The included checkpoint is synthetic-trained, not RGL-trained. This is not a complete reproduction of Al-Sit's texture-segmentation method, or a field-validated geological interpretation system. The supplied files do not establish a measurable "best" model or perfect compliance with an unseen marking rubric.

## Start here

From PowerShell, in this folder:

~~~powershell
python -m pip install -r requirements.txt
python -B -m unittest discover -v
python -B benchmark.py
python -B televiewer.py --channels optical acoustic
python -B review_results.py
~~~

Open **results/review.html** for a searchable candidate table, quality meters, downloadable reviewer decisions and expandable page images. The simpler **results/index.html** links directly to all plots. Rebuild review.html after each new analysis. The original assignment script is unchanged.

A shorter run:

~~~powershell
python -B televiewer.py --boreholes BH01 --channels optical --output trial_BH01
~~~

Each output directory contains:
- candidates.csv: native pixel parameters, provisional depth, support, residual error, and available physical geometry.
- evidence.npz: ridge evidence, valid-pixel mask, and annotation mask.
- page_NNN.png: original image, evidence, and fitted curves in manageable depth intervals.

The run root contains all_candidates.csv, summary.csv, config_used.json, run_manifest.json, and index.html. Use a fresh --output directory for experiments: output files are overwritten, and old pages from a different run are not automatically deleted.

Candidate counts are not fracture counts. A low count can mean missed features; a high count can mean false detections.

## Findings in the old code

1. A single crop layout was applied to different reports. BH02's optical track begins at column 54; the old crop began at 103 and mixed optical/acoustic information. BH01's old acoustic crop extended into the dip plot.
2. A fixed 100-row header removal left large blank intervals. Those pixels participated in the global darkest-20-percent threshold.
3. Eight votes is too permissive: at 20% dark occupancy a random 150-column curve already expects about 30 hits, even before maximizing over many hypotheses.
4. The maximum amplitude of 15% of image width is an arbitrary limit on apparent dip and misses steeper traces.
5. Depth-first suppression selected the first acceptable curve rather than the best-supported curve.
6. The dip proxy assumes a pixel aspect ratio that does not apply to report images. Scaling a plot vertically changed the reported geology.
7. The BH01 acoustic track and the BH02 travel-time track already contain coloured expert sinusoids. The BH02 acoustic-amplitude track is unpicked. Detecting those curves cannot demonstrate automatic interpretation accuracy.

## What the new implementation does

- Uses explicit, size-checked layouts for each report and preserves original row coordinates.
- Excludes white report regions and image boundaries.
- Masks the known saturated annotation palette on annotated tracks. Nearest-neighbour filling is used only for filtering; overwritten pixels never contribute evidence. It cannot restore geology hidden by a pick, and it can mask naturally similar colours. Antialiased/unknown ink can remain.
- Uses local multiscale dark/bright ridge responses instead of a fixed fraction of the image. Optical uses both polarities. Acoustic uses dark ridges from the rendered image.
- Uses structure-tensor ridge direction to reject incompatible evidence.
- Uses generalized sinusoidal Hough voting and robust soft-L1 curve refinement.
- Requires angular coverage, sectors, bounded gaps, a connected arc, fit residual limits, and support above nearby traces.
- Processes overlapping depth tiles. Suppression considers whole-curve similarity and reuse of supporting pixels, allowing crossing curves at similar depths.
- Flags near-horizontal candidates for checking acquisition/casing artifacts. Horizontal geological features remain possible.
- Exports an explicit uncalibrated quality score. It is a ranking statistic, not an 80%-correct probability.
- Supports a supplied pixel-evidence array as a future interface for a separately trained segmentation model.
- Offers one-to-one reference matching within fully reviewed intervals. No ground-truth values are fabricated.

The defaults are exploratory, chosen with synthetic checks and visual inspection of the supplied reports. They are not held-out field estimates. Weak/irregular optical traces, closely spaced layers, severe occlusions, steep features beyond the configured amplitude range, and acquisition artifacts remain difficult.

## Image layout and calibration

All bounds use zero-based pixels and half-open x intervals [x0,x1). Never resize the report without updating both the layout and calibration.

| Borehole | Optical x bounds | Acoustic x bounds | Header end row |
|---|---|---|---|
| BH01 | 103:254 | 255:406 | 113 |
| BH02 | 54:212 | 213:371 | 110 |
| BH03 | 107:292 | 293:477 | 973 |

Depth calibration uses two visually read tick marks per report. This is provisional, especially over a long raster export; replace it with original depth samples or more thoroughly verified ticks.

- BH01: row 224 = 0.4 m; row 336 = 0.8 m.
- BH02: row 518 = 0.0 m; row 630 = 0.4 m.
- BH03: row 1490 = 2.4 m; row 1602 = 2.8 m.

An azimuth period equals the full track width by default. Set endpoint=true on a track only if the last pixel duplicates the first azimuth sample. Verify seam position and direction against the original export.

Travel-time tracks are configured for BH02/BH03 but are not included in the saved optical/acoustic run. Running without --channels also processes those tracks as exploratory dark-ridge images. Rendered travel-time colours are not calibrated measurements; this is not aperture or ovalisation analysis.

## Geometry: when a dip angle is justified

The fitted curve is:

~~~text
row(theta) = center + s*sin(theta) + c*cos(theta)
amplitude_px = hypot(s, c)
phase = atan2(c, s)  # for A*sin(theta + phase)
depth_m = row * metres_per_pixel + intercept
amplitude_m = amplitude_px * metres_per_pixel
borehole_relative_dip = atan(2 * amplitude_m / diameter_m)
~~~

Amplitude here is HALF the peak-to-trough vertical difference.

Set diameter_m in rgl_config.json only using verified borehole/caliper information, in metres. No diameter is assumed from image width or guessed from a report curve. Without diameter, dip columns remain empty.

The relative dip is measured against the plane perpendicular to the borehole axis. It equals geographic dip only for a vertical borehole. For geographic values, supply image_frame_to_ned: a proper 3x3 rotation whose columns are the image theta=0 radial direction, theta=90 radial direction, and downhole axis, expressed in North/East/Down coordinates. This defines the image's actual azimuth frame, including any orientation corrections already applied. Do not apply tool corrections twice.

Identity means a VERIFIED vertical borehole with theta=0 North and theta=90 East, in the intended geographic reference. Magnetic North does not automatically equal true North. A constant matrix/diameter is only suitable where they are adequately constant; varying caliper and deviation require splitting the log into appropriate intervals or extending the geometry interface.

Phase is not automatically dip azimuth. With depth increasing downward, the deepest point is theta = 90 degrees - phase. For a verified North/East vertical frame that is the dip direction. Phase/dip direction is undefined for a flat trace.

## Reference validation and the feedback loop

Request these from RGL/supervisor:
1. Clean optical and acoustic exports with depth/azimuth samples and palette or original amplitude data.
2. Expert picks in CSV with depth, dip, azimuth, feature class, and corresponding borehole/channel.
3. Diameter/caliper data, deviation survey, seam convention, and whether North is magnetic or true.
4. Fully reviewed depth intervals, core photos/lab references and depth alignment where available.
5. Required error tolerances and whether bedding, veins, and different joint classes belong in the target set.

For direct curve validation, one reference CSV per track needs:
~~~csv
feature_id,center_row_px,sin_coefficient_px,cos_coefficient_px,feature_type
~~~

These are original report rows and LOCAL track columns, not coordinates from a plotted/resized PNG. Optional measured depth_m, borehole_relative_dip_deg, true_dip_deg, and true_dip_azimuth_deg columns may be included. If absent, geometry is derived from the same configuration; agreement in derived angles checks curve fitting, not independent calibration accuracy.

To fit human-selected trace points, create:
~~~csv
feature_id,x_px,row_px,feature_type
~~~
Use at least six well-distributed points per feature, then:
~~~powershell
python -B fit_reference.py --points expert_points.csv --width 151 --output bh01_truth.csv
~~~
Inspect reference_fit_rmse_px. Poor fits are not reliable truth.

Add these fields to the clean track's JSON configuration:
~~~json
{
  "truth_csv": "bh01_truth.csv",
  "reviewed_intervals": [[1500, 2500]],
  "match_tolerance_px": 5
}
~~~

All target features whose centers fall within those intervals must be labelled. The evaluator restricts both predictions and references to those intervals, matches each trace at most once by mean vertical distance across the complete azimuth, and reports TP/FP/FN, precision, recall, F1, and parameter MAEs. Circular phase/azimuth errors wrap correctly. Counts of matched values accompany MAEs. Blank metrics indicate unavailable or undefined quantities, not zero error. On annotated input, independent evaluation is refused.

Use separate boreholes for development and testing, not adjacent patches from the same report. Tune thresholds on development labels, freeze them, and run the held-out borehole once. Group corresponding optical/acoustic images and overlapping crops together to avoid leakage. A later trained classifier or U-Net needs similarly independent training, validation, and test boreholes. Calibrate probability estimates on a separate labelled validation set.

The synthetic benchmark is a reproducible software check; it does not replace expert RGL/core validation. Classification, aperture, rock types, ovalisation, fracture networks and RQD are not inferred from these sinusoids.

## Segmentation model integration

For the implemented U-Net training, inference and direct integration, use [UNET_GUIDE.md](UNET_GUIDE.md). Set `unet_checkpoint` on a clean track to select that route. The `evidence_npy` interface below remains available for other segmenters.

A track may specify evidence_npy pointing to a finite nonnegative HxW NumPy array in native report coordinates, containing thin candidate ridges (zero elsewhere). The pipeline still applies validity masks, geometric fitting, direction/continuity checks and validation. A thick U-Net probability mask must first be converted to appropriate ridge evidence. This interface does not train a model or certify its output.

## Sources and scope

- Supplied Y+A RGL NOTES.pdf and Borehole Analysis Flowchart.jpg: primary project requirements.
- Al-Sit et al. (2015), Visual texture for automated characterisation of geological features in borehole televiewer imagery, DOI: https://doi.org/10.1016/j.jappgeo.2015.05.015. The supplied paper supports preprocessing, sinusoidal geometry and continuity checks. This implementation uses local ridge evidence and a structure tensor; it does not implement the paper's full Gabor/clustering architecture.
- Supplied Waleed Al-Sit thesis and Cruz et al. deep-learning paper: motivation for later texture/segmentation and learned candidate rejection.
- Robertson Geo instrument description: https://www.robertson-geo.com/borehole-televiewers/ . Optical and acoustic images carry complementary information; acoustic amplitude fractures may appear as dark sinusoids.

The older dip histograms should not be presented as physical dip measurements. The new outputs are a reviewable proof of principle with explicit limits, ready for the missing RGL calibration and independent labels.


## Validation delivered with this version

All 16 regression tests pass. The development benchmark and a separate frozen-parameter synthetic set each contain 20 scenes and 40 known traces: both recovered 40/40 with zero extras. These controlled scenes are simpler than field logs and are not estimates of RGL accuracy. See synthetic_benchmark/summary.json and synthetic_holdout/summary.json. The full CLI/reference-CSV workflow is exercised in verification/results.

Reproduce the separate synthetic set:

```powershell
python -B benchmark.py --seed-start 2000 --output synthetic_holdout
```

The interactive dashboard exports candidate accept/reject/uncertain decisions. It does not create complete ground truth: missed features must also be labelled, for example using fit_reference.py.

BH02-specific limitation: the saved acoustic-amplitude analysis accepts zero candidates. Its coarse, broad texture is poorly represented by this narrow-ridge baseline. This is not evidence that the borehole lacks fractures, and it is not caused by annotation masking: the expert overlays are on the separate travel-time track. Broad-boundary/texture segmentation and original amplitude samples should be evaluated next.
