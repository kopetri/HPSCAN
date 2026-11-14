"""Released checkpoints and the configuration of every row of Tables 2-5 of the paper.

Each row lists the model (a released checkpoint or a scikit-learn baseline), the test set,
the inference precision that reproduces the published number and the published values.
`reproduce.py` evaluates the rows and prints them next to the paper's numbers.
"""

RELEASE_URL = "https://github.com/kopetri/HPSCAN/releases/download/v1.0"

# released checkpoint -> training run (W&B id of the original experiment) and description
CHECKPOINTS = {
    "hpscan_psc.ckpt": dict(run="e5kceusg", step="epoch 202",
                            desc="final HPSCAN model: fine-tuned with D=0.1 (Table 2c, Table 3 PSC)"),
    "hpscan_sdr.ckpt": dict(run="75gkt0qz", step="step 5000",
                            desc="HPSCAN fine-tuned with D=1.0 on all stimuli (Table 3 SDR, Table 4)"),
    "image_baseline.ckpt": dict(run="08pfg7yf", step="step 8000", desc="image-based U-Net baseline (Table 3)"),
    "hpscan_clustme.ckpt": dict(run="t5bv36p7", step="step 16000",
                                desc="HPSCAN trained on the best-agreeing raters (>= 90%), D=80 (Table 5)"),
    "table2a_D0.1.ckpt": dict(run="gygf32f8", step="epoch 999", desc="Table 2a, D=0.1"),
    "table2a_D1.ckpt": dict(run="lpi6njag", step="epoch 999", desc="Table 2a, D=1"),
    "table2a_D10.ckpt": dict(run="4x24ihpn", step="epoch 999", desc="Table 2a, D=10"),
    "table2a_D50.ckpt": dict(run="shyfd9e4", step="epoch 999", desc="Table 2a, D=50 (= Table 2b UNF)"),
    "table2a_D100.ckpt": dict(run="144c5mqo", step="epoch 999", desc="Table 2a, D=100"),
    "table2b_T50.ckpt": dict(run="3penawtl", step="epoch 999", desc="Table 2b, T_agree=50% (final checkpoint)"),
    "table2b_T60.ckpt": dict(run="z9qcp50k", step="epoch 999", desc="Table 2b, T_agree=60% (final checkpoint)"),
    "table2b_T70.ckpt": dict(run="ncymmvgg", step="epoch 999", desc="Table 2b, T_agree=70% (final checkpoint)"),
    "table2b_T80.ckpt": dict(run="am1wqwmu", step="epoch 999", desc="Table 2b, T_agree=80% (final checkpoint)"),
    "table2b_T90.ckpt": dict(run="ada78pvv", step="epoch 999", desc="Table 2b, T_agree=90%"),
    "table2c_D0.01.ckpt": dict(run="777ck0hq", step="step 8000", desc="Table 2c, D=0.01"),
    "table2c_D1.ckpt": dict(run="75gkt0qz", step="step 1000", desc="Table 2c, D=1.0 (same run as hpscan_sdr.ckpt)"),
}

DATASETS = {"PSC": dict(max_cluster=6), "SDR": dict(max_cluster=3)}


def row(label, model, data, paper, amp=False, note=None):
    return dict(label=label, model=model, data=data, max_cluster=DATASETS[data]["max_cluster"], amp=amp,
                paper=paper, note=note)


def ka(a, v, n=None):
    return {"kappa_alpha": a, "kappa_v": v, "kappa_n": n}


TABLE_2_NOTE = "kappa_n of Table 2 used an unrecorded variant of the noise index, see README"
UNKNOWN_CKPT = ("the evaluated checkpoint of this run could not be identified; "
                "the final checkpoint (epoch 999) is released")
RANDOMISED = "randomised baseline, the seed of the paper run was not recorded"
SINGLE_CLUSTER = "computed with 1 cluster, not with the ground-truth count"

TABLES = {
    "2a": [
        row("D=0.1", "table2a_D0.1.ckpt", "PSC", ka(-5.64, 0.53, 49.97)),
        row("D=1", "table2a_D1.ckpt", "PSC", ka(-6.33, 0.43, 48.09)),
        row("D=10", "table2a_D10.ckpt", "PSC", ka(-2.01, 0.61, 46.26)),
        row("D=50", "table2a_D50.ckpt", "PSC", ka(-1.66, 0.62, 45.91)),
        row("D=100", "table2a_D100.ckpt", "PSC", ka(-1.78, 0.61, 50.76)),
    ],
    "2b": [
        row("UNF", "table2a_D50.ckpt", "PSC", ka(-1.66, 0.62, 45.91)),
        row("T=50%", "table2b_T50.ckpt", "PSC", ka(-1.52, 0.62, 47.51), note=UNKNOWN_CKPT),
        row("T=60%", "table2b_T60.ckpt", "PSC", ka(-1.34, 0.63, 50.24), note=UNKNOWN_CKPT),
        row("T=70%", "table2b_T70.ckpt", "PSC", ka(-0.55, 0.68, 58.48), note=UNKNOWN_CKPT),
        row("T=80%", "table2b_T80.ckpt", "PSC", ka(-1.27, 0.57, 54.08), note=UNKNOWN_CKPT),
        row("T=90%", "table2b_T90.ckpt", "PSC", ka(-1.66, 0.58, 53.92)),
    ],
    "2c": [
        row("D=0.01", "table2c_D0.01.ckpt", "PSC", ka(-0.85, 0.66, 57.07)),
        row("D=0.1", "hpscan_psc.ckpt", "PSC", ka(-0.52, 0.69, 60.91)),
        row("D=1.0", "table2c_D1.ckpt", "PSC", ka(-0.68, 0.67, 52.36)),
    ],
    "3": [
        # --- PSC test set
        row("PSC HPSCAN", "hpscan_psc.ckpt", "PSC", ka(-0.52, 0.69, 59.67)),
        row("PSC Baseline (image)", "image_baseline.ckpt", "PSC", ka(-2.40, 0.51, 36.13)),
        row("PSC DBSCAN", "dbscan_0.2_10", "PSC", ka(-4.13, 0.62, 52.12)),
        row("PSC OPTICS", "optics_0.1_0.05_0.1", "PSC", ka(-5.56, 0.56, 21.29)),
        row("PSC Ward*", "ward", "PSC", ka(-1.04, 0.66)),
        row("PSC Mean Shift", "meanshift_0.4", "PSC", ka(-6.90, 0.53)),
        row("PSC Affinity Propagation", "affinity-propagation_0.9", "PSC", ka(-10.34, 0.39), note=RANDOMISED),
        row("PSC Spectral Clustering*", "spectral", "PSC", ka(-2.65, 0.53), note=RANDOMISED),
        row("PSC Agglomerative Clustering", "average-linkage_2", "PSC", ka(-7.02, 0.38)),
        row("PSC BIRCH", "birch_2", "PSC", ka(-7.68, 0.38)),
        row("PSC K-means*", "kmeans", "PSC", ka(-1.49, 0.63), note=RANDOMISED),
        row("PSC Gaussian Mixture Model*", "gmm", "PSC", ka(-0.90, 0.67), note=RANDOMISED),
        # --- SDR test set
        row("SDR HPSCAN", "hpscan_sdr.ckpt", "SDR", ka(-1.34, 0.64, 46.28), amp=True),
        row("SDR Baseline (image)", "image_baseline.ckpt", "SDR", ka(-5.35, 0.43, 33.86), amp=True),
        row("SDR DBSCAN", "dbscan_0.2_10", "SDR", ka(-3.19, 0.50, 28.36)),
        row("SDR OPTICS", "optics_0.1_0.05_0.1", "SDR", ka(-3.15, 0.60, 36.11)),
        row("SDR Ward*", "ward_1", "SDR", ka(-7.21, 0.62), note=SINGLE_CLUSTER),
        row("SDR Mean Shift", "meanshift_0.4", "SDR", ka(-5.28, 0.60)),
        row("SDR Affinity Propagation", "affinity-propagation_0.9", "SDR", ka(-13.04, 0.31), note=RANDOMISED),
        row("SDR Spectral Clustering*", "spectral_1", "SDR", ka(-7.21, 0.62), note=SINGLE_CLUSTER),
        row("SDR Agglomerative Clustering", "average-linkage_2", "SDR", ka(-5.85, 0.43)),
        row("SDR BIRCH", "birch_2", "SDR", ka(-6.47, 0.46)),
        row("SDR K-means*", "kmeans_1", "SDR", ka(-7.21, 0.62), note=SINGLE_CLUSTER),
        row("SDR Gaussian Mixture Model*", "gmm_1", "SDR", ka(-7.21, 0.62), note=SINGLE_CLUSTER),
    ],
    "4": [
        row("PSC", "hpscan_sdr.ckpt", "PSC", {"mse": 0.0305, "mae": 0.1393},
            note="MSE reproduced; the MAE of this checkpoint is 0.1493"),
        row("SDR", "hpscan_sdr.ckpt", "SDR", {"mse": 0.0317, "mae": 0.1564}, amp=True,
            note="not reproduced: Table 4 (SDR) was computed in a separate, unrecorded run"),
    ],
}

# Table 5 (ClustMe cluster counting): evaluated by clustme.py
TABLE_5 = {
    "model": "hpscan_clustme.ckpt",
    "amp": False,
    "paper": {"HPSCAN": (0.50, 0.84, 0.55), "random": (0.00, 0.50, 0.28), "single": (0.00, 0.80, 0.00),
              "multi": (0.00, 0.20, 0.33)},
}

for _table in ("2a", "2b", "2c"):
    for _row in TABLES[_table]:
        _row["note"] = _row["note"] or TABLE_2_NOTE
