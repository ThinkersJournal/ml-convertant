# ml-convertant-calibrate

Quantizes dense weights into the AWQ 4-bit layout (`ml_convertant_awq::LogicalAwq`) and reports the error.

This release has the **round-to-nearest baseline** (`rtn_awq`): per group of `group_size` input features, asymmetric
min/max, f16 scale rounded up, nearest zero point. Reports: weight error with a proven per-group bound, and, when the
caller supplies calibration activations, the measured error of the layer's output on them.

The AWQ-style scale search is the next release. Not verified: no comparison with a real AWQ checkpoint. This crate
never runs a model; the caller collects the activations. See `docs/CALIBRATION-DESIGN.md` in the repository.
