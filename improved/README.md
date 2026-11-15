# Improved HPSCAN (post-publication)

This folder is **not part of the paper**. It contains improvements found after publication; the code and
checkpoints that reproduce the paper are unchanged (see the main [README](../README.md) and `reproduce.py`).

Two changes:

1. **Similarity decoding with test-time augmentation** (`decoding.py`, works with every HPSCAN checkpoint).
   HPSCAN is trained with a pairwise loss: for two points the network predicts whether they belong to the same
   cluster, P = pᵀp over the 20 cluster channels. The paper reads clusters out per point (argmax over the
   channels), which can spread one cluster over several channels. Here P, the NOISE probability and the
   agreement are averaged over the 8 rotations / mirror images of the [-1, 1] box; points with a NOISE
   probability above 0.7 are NOISE and the others are grouped by average-linkage clustering of 1 − P, cut at
   τ = 0.9.
2. **A new training recipe** (`train.py`, `training.py`): same network and loss terms as the paper, trained in
   one stage (AdamW, learning rate 1e-3 with cosine decay, weight decay 1e-4, 300 epochs, ≈45 min on one GPU)
   with more augmentation: the paper's crop and rotation plus mirror flips, min-max renormalisation, Gaussian
   jitter (σ ≤ 0.05) and random point subsets (256–512 points). Released as `hpscan_improved.ckpt`.

## Usage

```bash
python download.py --improved   # -> checkpoints/hpscan_improved.ckpt

# cluster your own scatterplots (same input/output as predict.py)
python improved/predict.py --model checkpoints/hpscan_improved.ckpt --input my_scatterplot.csv --plot

# evaluate (metrics and protocol of the paper, Tables 3 and 5)
python improved/evaluate.py --model checkpoints/hpscan_improved.ckpt --data data/PSC --max_cluster 6
python improved/evaluate.py --model checkpoints/hpscan_improved.ckpt --data data/SDR --max_cluster 3
python improved/evaluate.py --model checkpoints/hpscan_improved.ckpt --clustme data/ClustMe

# same prior as the starred baselines of Table 3: ground-truth number of clusters, no NOISE
python improved/evaluate.py --model checkpoints/hpscan_improved.ckpt --data data/SDR --max_cluster 3 --gt_count --noise_threshold 1

# the paper model with similarity decoding (parameters selected for it on PSC valid)
python improved/evaluate.py --model checkpoints/hpscan_psc.ckpt --data data/PSC --max_cluster 6 --tau 0.8 --min_fraction 0.05

# train (the released checkpoint: all of PSC train, seed 0)
python improved/train.py --data data/PSC --output_dir runs/improved
```

## Results

Metrics and evaluation protocol of the paper (`hpscan/evaluation.py`); κα / κv / κn for PSC test and SDR,
κv / accuracy / F1 for cluster counting on ClustMe (999 stimuli, Table 5 protocol).

| | PSC test | SDR | ClustMe |
| --- | --- | --- | --- |
| paper, Table 3 / 5 (three different checkpoints) | −0.52 / 0.69 / 59.67 | −1.34 / 0.64 / 46.28 | 0.50 / 84 % / 55 % |
| `hpscan_psc.ckpt` + similarity decoding (τ 0.8, min_fraction 0.05) | −0.13 / 0.70 / 74.65 | −1.91 / 0.59 / 50.24 | 0.61 / 84 % / 61 % |
| **`hpscan_improved.ckpt`** | **0.70 / 0.76 / 64.08** | −3.16 / 0.56 / 44.35 | **0.65 / 88 % / 59 %** |
| best baseline without the ground-truth count | −2.40 / 0.51 / 36.13 (image baseline), −4.13 / 0.62 / 52.12 (DBSCAN) | −3.15 / 0.60 / 36.11 (OPTICS) | 0.34 / 63 % / 44 % (Mean Shift), 0.33 / 75 % / 38 % (DBSCAN) |
| best baseline with the ground-truth count | −0.90 / 0.67 (GMM\*), −1.04 / 0.66 (Ward\*) | 0.20 / 0.75 (Ward\*) | – |

With the same prior as the starred baselines (`--gt_count --noise_threshold 1`), `hpscan_improved.ckpt` reaches
−0.22 / 0.70 on PSC test (Ward\* −1.04 / 0.66) and 0.19 / 0.75 on SDR (Ward\* 0.20 / 0.75). The agreement
regression (Table 4) improves on PSC (MSE 0.0290, MAE 0.1302) and is similar on SDR (0.0357 / 0.1573).

The ClustMe baselines are not in the paper; they use the parameters of Table 3 and the same counting rule
(clusters with ≥ 5 % of the points). Baselines with the ground-truth count are not defined on ClustMe (the count is
the target). The SDR baselines with the ground-truth count are the corrected values of the main README.

**Summary.** One model now outperforms all baselines on PSC (also those given the ground-truth number of clusters)
and on ClustMe, where it also improves on the paper's dedicated ClustMe checkpoint. **SDR is not solved:** without
the ground-truth count, `hpscan_improved.ckpt` is on par with OPTICS (κα −3.16 vs −3.15, κv 0.56 vs 0.60) and
below the paper's SDR checkpoint; given the count, it is on par with Ward\*. SDR consists of diffuse,
Gaussian-like blobs, while PSC mostly contains thin structures from t-SNE / PCA; PSC-trained models label parts of
the blobs as NOISE. Neither noise calibration (even when tuned on SDR) nor training with SDR stimuli (5-fold
cross-validation) closed this gap in our experiments.

## How the recipe and the decoding parameters were chosen

No choice was made on a test set. During development, models were trained on PSC train without 200 held-out
stimuli (`--holdout 200`, fixed seed); the training variants, the checkpoint epoch and the decoding parameters
(NOISE threshold ∈ {0.5, 0.7, 0.9}, τ ∈ {0.5, …, 0.98}, min_fraction ∈ {0, 0.03, 0.05}) were selected on the
held-out stimuli + PSC valid (287 stimuli) with κα + 10 κv. The selected recipe was then trained once on all of
PSC train for the released checkpoint. The decoding parameters of the paper model were selected on PSC valid,
which it was not trained on.

Development runs (held-out protocol, three seeds of the selected recipe, not released):

| seed | PSC test | SDR | ClustMe | SDR with ground-truth count |
| --- | --- | --- | --- | --- |
| 0 | 0.79 / 0.76 / 68.9 | −2.75 / 0.58 / 44.5 | 0.57 / 85 % / 53 % | 0.59 / 0.78 |
| 1 | 0.36 / 0.75 / 63.4 | −3.53 / 0.54 / 43.4 | 0.66 / 87 % / 63 % | 0.23 / 0.75 |
| 2 | 0.79 / 0.77 / 66.9 | −3.74 / 0.55 / 43.3 | 0.51 / 83 % / 52 % | 0.16 / 0.75 |

(computed with a faster re-implementation of the metrics; PSC numbers differ by up to ±0.03 from
`improved/evaluate.py` because of batched inference.)

Retraining the released configuration with `improved/train.py` (seed 0; GPU training is not bit-wise
deterministic) gave 0.75 / 0.76 / 61.20 on PSC test, −3.31 / 0.56 / 40.96 on SDR and 0.62 / 87 % / 53 % on
ClustMe, i.e. results within the seed-to-seed variation above.

What did not help: a consensus target over all raters instead of one random rater, balanced NOISE loss weights,
post-processing of the argmax output, and fine-tuning with SDR stimuli.

Decoding matters for the new models: with the paper's argmax readout they are worse than the paper model
(PSC −1.74 / 0.62); similarity decoding without test-time augmentation gives 0.43 / 0.74, with 8 views 0.79 / 0.76
(seed 0).
