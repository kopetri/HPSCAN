# HPSCAN: Human Perception-Based Scattered Data Clustering

Official code, checkpoints and evaluation scripts of

> S. Hartwig, C. v. Onzenoodt, D. Engel, P. Hermosilla and T. Ropinski.
> **HPSCAN: Human Perception-Based Scattered Data Clustering.**
> Computer Graphics Forum, 2025. [doi:10.1111/cgf.15275](https://doi.org/10.1111/cgf.15275)

HPSCAN is a PointNet++ model that clusters 2D scatterplots the way humans perceive them. It is
trained on the **Perceptual Scatterplot Clustering (PSC)** dataset, crowdsourced point-wise cluster
annotations of 1,464 scatterplots, available on [Hugging Face](https://huggingface.co/datasets/kopetri/PSC).
For every point, HPSCAN predicts a cluster id or NOISE, and an estimate of how much human
raters agree on that point.

## Contents

| File | Purpose |
| --- | --- |
| `predict.py` | cluster your own scatterplots (`.csv` / `.npy`) |
| `reproduce.py` | reproduce Tables 2–5 of the paper with the released checkpoints |
| `evaluate.py` | evaluate a checkpoint or a scikit-learn baseline on a test split |
| `clustme.py` | cluster counting on ClustMe (Table 5) |
| `train.py` | train HPSCAN or the image-based baseline |
| `search_baseline_parameters.py` | parameter search of the scikit-learn baselines (Table 3) |
| `download.py` | download checkpoints and datasets |
| `tools/` | conversion of the ClustMe data |
| `improved/` | post-publication improvements (not part of the paper), see [improved/README.md](improved/README.md) |
| `hpscan/` | model, losses, metrics, datasets and evaluation protocol; `hpscan/paper.py` holds the configuration of every table row |

## Installation

Tested with Python 3.8, PyTorch 2.0.0 and CUDA 11.7.

```bash
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## Data and checkpoints

```bash
python download.py --checkpoints   # -> checkpoints/
python download.py --psc           # -> data/PSC  (Hugging Face kopetri/PSC)
python download.py --sdr           # -> data/SDR  (SDR scatterplots with our annotations)
```

**SDR** is released with this repository (`SDR.zip`, release v1.0): 272 scatterplots of the
dimensionality reduction benchmark of [Sedlmair et al. 2013](https://doi.org/10.1109/TVCG.2013.153),
annotated with the crowdsourcing procedure of PSC (68 participants).

A dataset folder holds one `.npy` file per scatterplot with shape `(N, 2 + R)`: the x/y position
of the N points in [-1, 1], followed by the cluster id assigned by each of the R raters (`-1` =
NOISE). `train.txt`, `valid.txt` and `test.txt` list the files of each split (PSC: 1,171 / 87 /
206 stimuli, SDR: 272 test stimuli).

**ClustMe** (Table 5) is not redistributed here. Obtain the Experiment 1 data of
[Abbas et al. 2019](https://doi.org/10.1111/cgf.13684) (`1000_2gaussians_scatterplots_ClustMe_EXP1.RData`
and `1000_34answers_scatterplots_ClustMe_EXP1.RData`) from the ClustMe authors, then convert it
(requires R):

```bash
Rscript tools/clustme_export.R <folder with the .RData files> data/ClustMe
python tools/convert_clustme.py data/ClustMe
```

### Released checkpoints

| Checkpoint | Training run | Used for |
| --- | --- | --- |
| `hpscan_psc.ckpt` | e5kceusg, epoch 202 | **final HPSCAN model**: Table 2c (D=0.1), Table 3 (PSC) |
| `hpscan_sdr.ckpt` | 75gkt0qz, step 5000 | Table 3 (SDR), Table 4 |
| `image_baseline.ckpt` | 08pfg7yf, step 8000 | image-based U-Net baseline, Table 3 |
| `hpscan_clustme.ckpt` | t5bv36p7, step 16000 | Table 5 |
| `table2a_D{0.1,1,10,50,100}.ckpt` | gygf32f8, lpi6njag, 4x24ihpn, shyfd9e4, 144c5mqo (epoch 999) | Table 2a; D=50 is also Table 2b (UNF) |
| `table2b_T{50,60,70,80,90}.ckpt` | 3penawtl, z9qcp50k, ncymmvgg, am1wqwmu, ada78pvv (epoch 999) | Table 2b |
| `table2c_D0.01.ckpt`, `table2c_D1.ckpt` | 777ck0hq (step 8000), 75gkt0qz (step 1000) | Table 2c |

Each checkpoint contains the network weights (`state_dict`), the training hyper-parameters
(`hyper_parameters`) and the training run it comes from (`info`).

## Inference

```bash
python predict.py --model checkpoints/hpscan_psc.ckpt --input my_scatterplot.csv --plot
```

The first two columns of the input are used as x and y, min-max scaled to [-1, 1] like the
training data. `my_scatterplot.hpscan.csv` contains the cluster id of every point (`0` = NOISE)
and the predicted agreement score in [0, 1]. `--plot` also saves an image of the clustering.

From Python:

```python
import numpy as np
from hpscan.module import load_model
from predict import normalize, predict

model = load_model("checkpoints/hpscan_psc.ckpt")
points = normalize(np.random.randn(512, 2))          # (N, 2) in [-1, 1]
labels, agreement = predict(model, points, device="cuda")
```

HPSCAN was trained on scatterplots with 512 points and predicts up to 20 clusters.

## Reproducing the paper

```bash
python reproduce.py --table all        # or e.g. --table 3
```

`reproduce.py` evaluates every row of Tables 2–5 with the configuration in `hpscan/paper.py`
(checkpoint, test set, inference precision) and prints it next to the published number. Metrics
that differ after rounding are listed after `!=`. Small differences of ±0.01 can come from GPU
numerics. Single rows can be evaluated with `evaluate.py`:

```bash
# Table 3, PSC
python evaluate.py --data data/PSC --model checkpoints/hpscan_psc.ckpt checkpoints/image_baseline.ckpt \
    dbscan_0.2_10 optics_0.1_0.05_0.1 ward meanshift_0.4 affinity-propagation_0.9 spectral \
    average-linkage_2 birch_2 kmeans gmm
# Table 3, SDR (the HPSCAN and image baseline rows were computed with mixed precision)
python evaluate.py --data data/SDR --max_cluster 3 --amp --model checkpoints/hpscan_sdr.ckpt checkpoints/image_baseline.ckpt
python evaluate.py --data data/SDR --max_cluster 3 --model dbscan_0.2_10 optics_0.1_0.05_0.1 ward_1 \
    meanshift_0.4 affinity-propagation_0.9 spectral_1 average-linkage_2 birch_2 kmeans_1 gmm_1
# Table 5
python clustme.py --data data/ClustMe --model checkpoints/hpscan_clustme.ckpt --dummy random single multi
```

A baseline is written as `<method>_<parameters>` (see `hpscan/baselines.py`). A method name
without parameters, e.g. `ward`, uses the ground-truth number of clusters of each stimulus.

### Evaluation protocol

* The test stimuli are grouped by their perceived number of clusters, from 0 (NO CLUSTER) up to 6
  for PSC and 3 for SDR; stimuli with more clusters are not evaluated. The count of a stimulus is
  the count its three lowest rater counts agree on, otherwise the rounded mean count.
* Each metric is averaged over the stimuli of a group. The group means are combined with weights
  proportional to 1 / (number of stimuli in the group), so rare groups such as NO CLUSTER weigh as
  much as frequent ones (`hpscan/evaluation.py`).
* κα is the change of the group agreement when one rater is replaced by the prediction (×100,
  higher is better, 0 = as good as a human). κv is the Vanbelle kappa index of the prediction vs.
  the group of raters; if the prediction has a single label, the positive agreement is used
  instead. κn is the Jaccard index of the NOISE points, averaged over the raters.
* The MSE / MAE of the agreement estimation (Table 4) are the unweighted mean over the groups.

### What is reproduced

| Table | Status |
| --- | --- |
| 2a | κα and κv reproduced. κn not reproduced (see below). |
| 2b | UNF and T=90% reproduced (κα, κv). For T=50–80%, the evaluated checkpoints could not be identified. The final checkpoints of these runs are released; their scores differ from the paper by up to 0.32 (κα) and 0.02 (κv). |
| 2c | κα and κv reproduced. |
| 3 | Reproduced (HPSCAN on PSC: κn 59.69% vs. 59.67%), except for the randomised baselines (Affinity Propagation; Spectral Clustering, K-means and GMM on PSC). Their seed was not recorded, so their scores vary slightly between runs. |
| 4 | PSC MSE reproduced; see below. |
| 5 | Reproduced. |

**Notes on the published numbers** (also listed in `hpscan/paper.py`):

* *Precision.* The SDR rows of HPSCAN and the image baseline in Table 3 were computed with float16
  mixed precision (`--amp`). All other rows use float32. With float32 inference, `hpscan_sdr.ckpt`
  reaches κα −0.74, κv 0.71, κn 57.91% on SDR.
* *SDR model.* The paper states that the SDR rows of Table 3 use the best model of the clustering
  comparison. They were computed with `hpscan_sdr.ckpt`, which was fine-tuned with D=1.0 on all
  stimuli with T_agree ≥ 70%. The final PSC model `hpscan_psc.ckpt` reaches κα −3.87, κv 0.54,
  κn 40.88% on SDR.
* *Starred baselines on SDR.* Table 3 marks Ward, Spectral Clustering, K-means and GMM as using the
  ground-truth number of clusters. This holds for PSC. On SDR, these four rows were computed with a
  single cluster (`ward_1`, …), which is why they are identical. With the ground-truth count, they
  reach κα 0.20 / κv 0.75 (Ward), −1.35 / 0.66 (Spectral Clustering), −0.04 / 0.73 (K-means) and
  0.01 / 0.74 (GMM) on SDR (`evaluate.py --data data/SDR --max_cluster 3 --model ward spectral kmeans gmm`).
* *κn in Table 2* was computed with a variant of the noise index that was not recorded. The κn
  values of the same checkpoint differ between Table 2c (60.91%) and Table 3 (59.67%). Table 3's
  κn is reproduced by `hpscan/metrics.py`.
* *Table 2* reports test-set numbers; the caption says validation set.
* *Table 2c.* D=0.01 and D=0.1 were fine-tuned on single-cluster stimuli only (`--n_cluster 1`).
  The D=1.0 row comes from a run fine-tuned on all stimuli (the run of `hpscan_sdr.ckpt`, step 1000).
* *Table 4.* The PSC MSE (0.0305) is reproduced with `hpscan_sdr.ckpt`. Its MAE is 0.1493, while
  the paper states 0.1393. The SDR row was computed in a separate run that we could not identify;
  `hpscan_sdr.ckpt` reaches MSE 0.0308 and MAE 0.1541 on SDR.
* *Table 5.* The ClustMe points are min-max scaled to [−0.95, 0.95] before inference, and a
  cluster is counted if it contains ≥ 5% of the points. 999 of the 1,000 stimuli were evaluated;
  `tools/convert_clustme.py --all` adds the last one. Half of the ClustMe scatterplots have 100
  points; as in the original export, their points are repeated to 1,000.

## Training

`train.py` implements the training protocol of the paper:
* Optimiser: Adam with learning rate 1e-5, weight decay 0.1 and batch size 32.
* LR schedule: ReduceLROnPlateau on the validation loss (factor 0.1, patience 100 epochs).
* Loss: L = L_noise + 0.1 L_mcl + L_agree.
* Augmentation: random rotation and crop.
* Seed 1337, float32.

Checkpoints are written every 1,000 steps and for the best `valid_loss` / `valid_cross`. Pass
`--wandb_project` to log to Weights & Biases.

The released models were trained as follows (`--data data/PSC` omitted):

```bash
# Table 2a: contrastive weight D, from scratch
python train.py --dissimilar_weight 50 --max_epochs 1000                  # D in {0.1, 1, 10, 50, 100}
# Table 2b: only training stimuli with group agreement >= T_agree
python train.py --dissimilar_weight 50 --filter_agreement 0.9 --max_epochs 1000
# final model (Table 2c, Table 3 PSC): T_agree = 70% for 1,000 epochs, continued for 147k steps,
# then fine-tuned on single-cluster stimuli with D = 0.1 (D = 0.01 for table2c_D0.01.ckpt)
python train.py --dissimilar_weight 50 --filter_agreement 0.7 --max_epochs 1000 --name t70
python train.py --dissimilar_weight 50 --filter_agreement 0.7 --max_epochs 10000 --max_steps 147000 \
    --finetune runs/t70/checkpoints/step=28000.ckpt --name t70_continued
python train.py --dissimilar_weight 0.1 --filter_agreement 0.7 --n_cluster 1 --learning_rate 1e-7 \
    --max_epochs 10000 --finetune runs/t70_continued/checkpoints/step=147000.ckpt
# SDR model (hpscan_sdr.ckpt = step 5000, table2c_D1.ckpt = step 1000): as above, D = 1.0, all cluster counts
python train.py --dissimilar_weight 1 --filter_agreement 0.7 --learning_rate 1e-7 --max_epochs 10000 \
    --finetune runs/t70_continued/checkpoints/step=147000.ckpt
# Table 5 model: per stimulus, the largest group of raters with agreement >= 90%, D = 80
python train.py --dissimilar_weight 80 --best_agreement 0.9 --max_epochs 1000
# image-based baseline: U-Net with ResNet-18 encoder, then fine-tuned with lr 1e-7 and D = 0.1
python train.py --model image --dissimilar_weight 50 --filter_agreement 0.7 --max_epochs 1000 --name unet
python train.py --model image --dissimilar_weight 0.1 --filter_agreement 0.7 --learning_rate 1e-7 \
    --finetune runs/unet/checkpoints/step=20000.ckpt
```

Training is not bit-exact across GPUs, so retrained models differ slightly from the released
checkpoints. The original training differs from the paper's description in three ways, and the
code reproduces the original:

* The ReduceLROnPlateau patience was 100 epochs; the paper states 50.
* Only random rotations and crops were applied. The horizontal and vertical flips described in the
  paper had no effect in the original code. `hpscan/data.py` keeps their random draws so the random
  sequence is unchanged.
* The image baseline was trained with a crop that leaves a gap between the two halves (margin 1.2)
  and min-max rescales afterwards, plus random flips of the input image. The flips were applied to
  the image only, not to the pixel coordinates of the points. `--model image` reproduces this.

## Improved model (post-publication)

The folder [`improved/`](improved/README.md) contains improvements found after publication; they are not
part of the paper and do not change any of the code above. Decoding the clusters from HPSCAN's pairwise
similarities with test-time augmentation, together with a new training recipe, gives a single model that
outperforms all baselines on PSC (κα 0.70, κv 0.76) and ClustMe (κv 0.65, 88 % accuracy). SDR remains
difficult; see [improved/README.md](improved/README.md) for the results, the selection protocol and the
limitations.

## Citation

```bibtex
@article{hartwig2025hpscan,
  title   = {{HPSCAN}: Human Perception-Based Scattered Data Clustering},
  author  = {Hartwig, Sebastian and van Onzenoodt, Christian and Engel, Dominik and Hermosilla, Pedro and Ropinski, Timo},
  journal = {Computer Graphics Forum},
  year    = {2025},
  doi     = {10.1111/cgf.15275}
}
```

## License

The code, the checkpoints and the SDR annotations are released under the [MIT License](LICENSE).
The SDR scatterplots are derived from the data of Sedlmair et al. 2013. The PSC dataset is
released under Apache-2.0 on [Hugging Face](https://huggingface.co/datasets/kopetri/PSC).
Third-party code and its licenses: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
