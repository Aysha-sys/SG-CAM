# Trained weights

The PatternNet-trained checkpoints (`<run key>.pt`, one per configuration and seed) are larger than is practical
for a Git repository and are distributed as assets of the GitHub Release / Zenodo archive associated with this repository.
Download them into this folder. Notebook 02 also accepts them when the folder is attached as a Kaggle input
(any path matching `**/checkpoints/*.pt` is discovered).

Run keys follow the pattern `<backbone>_<fusion>_cl<clip>t<tile>_i<size>_d<dim>_lr<lr>_bs<batch>_ep<epochs>_s<seed>`,
for example `mobilenet_v2_sgcam_cl2t8_i256_d256_lr0.0001_bs16_ep15_s0`.
