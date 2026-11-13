"""Recall@10 vs QPS: BenchYAML (graph) and BenchIvfYAML (IVF-ASH) experiments.csv on one chart.

usage: python3 plot_recall_qps.py GRAPH.csv IVF.csv [OUT.png] [--ivf-qps-column COL] [--fvec RESULTS.csv]...

--fvec adds fastervector's fvec-bench results.csv (throughput cases), one curve per build.
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # validated categorical order
MARKERS = ["o", "s", "^", "D"]                          # secondary encoding, so identity isn't color-only
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("graph_csv")
    p.add_argument("ivf_csv")
    p.add_argument("out", nargs="?", default="recall_vs_qps.png")
    p.add_argument("--ivf-qps-column", default="search.throughput.avg_qps",
                   help="IVF throughput column: avg_qps (batched) or coalesced.avg_qps")
    p.add_argument("--recall", default="search.accuracy.recall_at_10")
    p.add_argument("--fvec", action="append", default=[],
                   help="an fvec-bench results.csv; repeatable")
    a = p.parse_args()

    g = pd.read_csv(a.graph_csv)
    ivf = pd.read_csv(a.ivf_csv)

    # (label, recall, qps, knob values, knob name, color key, dashed): one color per graph build or
    # IVF nlist, so the same IVF build through jvector (solid) and fvec-bench (dashed) shares a hue
    curves = []
    for (m, ef, comp), d in g.groupby(["M", "efConstruction", "build_compressor"]):
        d = d.sort_values("overquery")
        curves.append((f"graph M={m} ef={ef}, fused PQ + NVQ rerank",
                       d[a.recall], d["search.throughput.avg_qps"], d["overquery"], "oq",
                       ("graph", m, ef, comp), False))
    for (nlist, sf), d in ivf.groupby(["nlist", "storageFactor"]):
        d = d.sort_values("nprobe")
        curves.append((f"IVF-ASH nlist={nlist} sf={sf}",
                       d[a.recall], d[a.ivf_qps_column], d["nprobe"], "np",
                       ("ivf", nlist), False))
    for path in a.fvec:
        f = pd.read_csv(path)
        f = f[f["mode"] == "throughput"]
        for (nc, qd, bits, storage), d in f.groupby(["ncentroids", "quantized_dim", "ashbits", "storage"]):
            d = d.sort_values("nprobe")
            curves.append((f"fvec-bench nlist={nc} q{qd} b{bits} {storage.split('-')[0]}",
                           d["recall"], d["qps_mean"], d["nprobe"], "np",
                           ("ivf", nc), True))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
    ax.set_facecolor("#fcfcfb")
    fig.patch.set_facecolor("#fcfcfb")

    keys = list(dict.fromkeys(c[5] for c in curves))
    for label, r, q, knob, kname, key, dashed in curves:
        i = keys.index(key)
        ax.plot(r, q, color=SERIES[i % len(SERIES)], lw=2, ls="--" if dashed else "-",
                marker=MARKERS[i % len(MARKERS)], ms=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label, zorder=3)
        # label the knob at the ends and every other point, not every point
        n = len(r)
        for j, (x, y, k) in enumerate(zip(r, q, knob)):
            if j in (0, n - 1) or j % 2 == 0:
                txt = f"{kname} {k:g}"
                ax.annotate(txt, (x, y), textcoords="offset points", xytext=(5, 5 if kname == "oq" else -11),
                            fontsize=7, color=MUTED)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if str(int(v))[0] in "25" else ""))
    ax.grid(True, which="both", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS (log scale)")

    ds = g["dataset_name"].iloc[0]
    ivf_mode = "batched" if a.ivf_qps_column == "search.throughput.avg_qps" else a.ivf_qps_column.split(".")[-2]
    ax.set_title(f"{ds}: Recall@10 vs QPS (IVF throughput: {ivf_mode}, {int(ivf['searchThreads'].iloc[0])} threads)",
                 loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
