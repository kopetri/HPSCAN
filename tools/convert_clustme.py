"""Convert the CSV export of tools/clustme_export.R to the format read by `ClustMeDataset`.

Usage: python tools/convert_clustme.py <export folder>

Writes NNNNN.npy (standardised and min-max scaled to [-1, 1]) next to every NNNNN.csv and a
`test.txt` split. Two details reproduce the data of Table 5:
* Half of the ClustMe scatterplots have 100 instead of 1000 points. The original export
  (an R data.frame of all scatterplots) recycled them to 1000 rows, i.e. every point is
  repeated 10 times. --repeat_to 1000 (default) does the same.
* Table 5 was computed on scatterplots 00001-00999 (the split file of the original
  experiments did not include 01000); pass --all to also include 01000.
"""
import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def read_csv2(path):
    """Read an R write.csv2 file (';' separated, ',' decimal mark, first column = row name)."""
    rows = []
    for line in Path(path).read_text().splitlines()[1:]:
        values = line.strip().split(";")[1:]
        rows.append([float(v.replace(",", ".")) for v in values])
    return np.array(rows, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--all", action="store_true", help="include scatterplot 01000 in test.txt")
    parser.add_argument("--repeat_to", type=int, default=1000,
                        help="repeat the points of smaller scatterplots up to this number (0: off)")
    args = parser.parse_args()
    folder = Path(args.folder)

    files = sorted(f for f in folder.glob("*.csv") if not f.name.endswith(".answer.csv"))
    for csv in files:
        xy = read_csv2(csv).astype(np.float32)  # as in the original conversion
        if args.repeat_to and len(xy) < args.repeat_to:
            xy = np.tile(xy, (int(np.ceil(args.repeat_to / len(xy))), 1))[:args.repeat_to]
        xy = MinMaxScaler((-1, 1)).fit_transform(StandardScaler().fit_transform(xy))
        np.save(csv.with_suffix(".npy"), xy.astype(np.float32))
    names = [f.with_suffix(".npy").name for f in files]
    if not args.all:
        names = [n for n in names if n != "01000.npy"]
    (folder / "test.txt").write_text("\n".join(names) + "\n")
    print(f"converted {len(files)} scatterplots, {len(names)} in test.txt")


if __name__ == "__main__":
    main()
