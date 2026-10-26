# Per-image predictions

`patternnet/<run key>.csv.gz` contains, for every test (and validation) image of every run: image path, split,
true label, predicted label and the 38 class probabilities. `millionaid/` contains the corresponding linear-probe
and robustness predictions. All tables and figures in `results/` and `figures/` are computed from these files by
the reporting cells of the two notebooks, so the reported numbers can be recomputed without retraining.
Distributed with the GitHub Release / Zenodo archive (see the main README).
