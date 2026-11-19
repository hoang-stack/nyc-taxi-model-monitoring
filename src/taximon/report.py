"""Figures and results/results.md from the study outputs (weekly.csv, detectors.csv, ...)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from taximon.drift import Threshold  # noqa: E402
from taximon.evaluate import COVID_SHOCK, add_aggregates, degraded, score  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
ACCENT, ORANGE, GREEN, YELLOW, PINK, VIOLET = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#6250d6",
)
ALERT_SIGNALS = {  # signal -> label in the alert map (a readable subset of all signals)
    "dle_mae": "DLE estimated MAE",
    "domain_auc": "domain classifier AUC",
    "psi:prediction": "PSI of predictions",
    "max_feature_psi": "max PSI over features",
    "psi:trip_distance": "PSI trip distance",
    "psi:hour": "PSI pickup hour",
    "psi:PULocationID": "PSI pickup zone",
    "psi:passenger_count": "PSI passenger count",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def _shock_line(ax, label=True) -> None:
    ax.axvline(COVID_SHOCK, color=BASELINE, lw=0.9, zorder=0)
    if label:
        ax.annotate(
            "NY emergency, 7 Mar 2020",
            (COVID_SHOCK, 1),
            xycoords=("data", "axes fraction"),
            xytext=(4, -2),
            textcoords="offset points",
            fontsize=8,
            color=MUTED,
            va="top",
        )


def plot_shock(weekly: pd.DataFrame, quality: pd.DataFrame, out: Path) -> Path:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 5.2), sharex=True)
    q = quality.loc[quality.index >= weekly.index.min()]
    a1.plot(q.index, q["trips_estimated"] / 1e6, color=INK)
    a1.set_ylabel("trips per week (M)")
    a1.set_ylim(bottom=0)
    a1.set_title("The shock: yellow-taxi traffic collapsed, and the trips that remained got faster")
    a2.plot(weekly.index, weekly["mean_speed_mph"], color=ACCENT)
    a2.set_ylabel("mean trip speed (mph)")
    for ax in (a1, a2):
        _shock_line(ax, label=ax is a1)
    _dates(a2)
    return _save(fig, out / "shock.png")


def plot_error(weekly: pd.DataFrame, out: Path) -> Path:
    truth, t = degraded(weekly)
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.axhspan(t.mean - t.k * t.std, t.upper, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.annotate(
        "band of ordinary weeks (2019 H2, mean ± 3 sd)",
        (weekly.index[2], t.upper),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
    )
    ax.plot(weekly.index, weekly["mae"], color=INK, label="realised MAE (needs labels)")
    ax.plot(
        weekly.index,
        weekly["dle_mae"],
        color=ACCENT,
        label="DLE estimate (no labels)",
    )
    ax.set_ylabel("MAE (minutes)")
    ax.set_ylim(bottom=0)
    _shock_line(ax)
    _dates(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    ax.set_title("The frozen 2019 model's error, week by week")
    return _save(fig, out / "error.png")


def plot_alert_map(weekly: pd.DataFrame, out: Path) -> Path:
    w = add_aggregates(weekly)
    truth, _ = degraded(w)
    ref = w["period"] == "reference"
    rows = {"MAE outside its band (ground truth)": truth.to_numpy()}
    for s, label in ALERT_SIGNALS.items():
        t = Threshold().fit(w.loc[ref, s].dropna())
        rows[label] = t.alerts(w[s].fillna(-np.inf))
    M = np.array([v.astype(float) for v in rows.values()])
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    x = mdates.date2num(w.index.to_pydatetime())
    step = np.median(np.diff(x))
    for i, v in enumerate(M):
        color = INK if i == 0 else ACCENT
        on = np.flatnonzero(v > 0)
        ax.barh(i, np.full(len(on), step), left=x[on] - step / 2, height=0.7, color=color, lw=0)
    ax.set_yticks(range(len(rows)), list(rows))
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.axvspan(x[0] - step / 2, x[ref.to_numpy()][-1] + step / 2, color=GRID, alpha=0.5, lw=0)
    ax.annotate(
        "reference weeks\n(thresholds set here)",
        (x[0], len(rows) - 0.4),
        fontsize=8,
        color=MUTED,
        va="bottom",
    )
    _shock_line(ax, label=False)
    ax.xaxis_date()
    _dates(ax)
    ax.set_xlim(x[0] - step, x[-1] + step)
    ax.set_title("Weeks in alert: every drift signal fires for three years; the error does not")
    return _save(fig, out / "alert_map.png")


def plot_psi_heatmap(weekly: pd.DataFrame, out: Path) -> Path:
    cols = [c for c in weekly if c.startswith("psi:")]
    M = weekly[cols].T.clip(lower=1e-3)
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    x = mdates.date2num(weekly.index.to_pydatetime())
    im = ax.pcolormesh(
        np.r_[x, x[-1] + 7] - 3.5,
        np.arange(len(cols) + 1),
        M.to_numpy(),
        norm=LogNorm(1e-3, max(1.0, float(M.max().max()))),
        cmap="Blues",
        shading="flat",
    )
    ax.set_yticks(np.arange(len(cols)) + 0.5, [c.split(":")[1] for c in cols])
    ax.invert_yaxis()
    ax.grid(False)
    ax.xaxis_date()
    _dates(ax)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("PSI vs training data (log scale)")
    cb.outline.set_visible(False)
    ax.set_title("Feature drift (PSI) per week against the 2019 H1 training data")
    return _save(fig, out / "psi_heatmap.png")


POLICY_STYLE = {
    "never retrain": INK,
    "every week (8-week window)": GREEN,
    "every 4 weeks (8-week window)": ACCENT,
    "every 13 weeks (8-week window)": VIOLET,
    "every 13 weeks (52-week window)": PINK,
    "error-triggered: MAE > 1.15x at deploy": ORANGE,
    "drift-triggered: domain AUC > 0.60": YELLOW,
}


def policy_table(pw: pd.DataFrame) -> pd.DataFrame:
    shock_end = pd.Timestamp("2020-06-29")
    rows = []
    for name, g in pw.groupby("policy", sort=False):
        rows.append(
            {
                "policy": name,
                "retrains": int(g["retrained_after"].sum()),
                "mae_all": g["mae"].mean(),
                "mae_shock": g.loc[(g.index >= COVID_SHOCK) & (g.index <= shock_end), "mae"].mean(),
                "mae_2022": g.loc[g.index >= pd.Timestamp("2022-01-01"), "mae"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("policy").sort_values("mae_all")


def plot_policies(pw: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    for name, color in POLICY_STYLE.items():
        g = pw[pw["policy"] == name]
        if len(g):
            smooth = g["mae"].rolling(4, min_periods=1).mean()
            ax.plot(g.index, smooth, color=color, label=name, lw=1.4)
    ax.set_ylabel("MAE (minutes, 4-week mean)")
    ax.set_ylim(bottom=0)
    _shock_line(ax)
    _dates(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5)
    ax.set_title("Retraining policies replayed on the real stream")
    return _save(fig, out / "policies.png")


def build_report(results_dir="results") -> Path:
    r = Path(results_dir)
    fig_dir = r / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _style()
    weekly = pd.read_csv(r / "weekly.csv", index_col=0, parse_dates=True)
    quality = pd.read_csv(r / "quality_weekly.csv", index_col=0, parse_dates=True)
    study = json.loads((r / "study.json").read_text())
    det = score(weekly)
    det.to_csv(r / "detectors.csv")
    plot_shock(weekly, quality, fig_dir)
    plot_error(weekly, fig_dir)
    plot_alert_map(weekly, fig_dir)
    plot_psi_heatmap(weekly, fig_dir)
    truth, t = degraded(weekly)
    mon = weekly["period"] == "monitor"
    lines = [
        "# Results",
        "",
        f"Model trained on {study['train_rows']:,} trips (2019 H1); "
        f"{int((~mon).sum())} reference weeks, {int(mon.sum())} monitored weeks.",
        "",
        f"MAE band from the reference weeks: {t.mean:.3f} ± 3 × {t.std:.3f} "
        f"(upper {t.upper:.3f} min). Monitored weeks above it: {int(truth[mon].sum())}.",
        "",
        "## Signals",
        "",
        "| Signal | Weeks in alert | Precision | Recall | False alarms before the shock "
        "| First alert after the shock | Spearman with MAE |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for s, row in det.iterrows():
        first = row["first_alert_after_shock"]
        lines.append(
            f"| {s} | {row['alert_weeks']} | {row['precision']:.2f} | {row['recall']:.2f} | "
            f"{row['false_alarms_before_shock']} | "
            f"{pd.Timestamp(first).date() if pd.notna(first) else '–'} | "
            f"{row['spearman_with_error']:.2f} |"
        )
    pol_path = r / "policies_weekly.csv"
    if pol_path.exists():
        pw = pd.read_csv(pol_path, index_col=0, parse_dates=True)
        plot_policies(pw, fig_dir)
        tab = policy_table(pw)
        tab.to_csv(r / "policies.csv")
        lines += [
            "",
            "## Retraining policies (monitored weeks)",
            "",
            "| Policy | Retrains | Mean weekly MAE | During the shock (Mar-Jun 2020) | 2022 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        lines += [
            f"| {name} | {row['retrains']} | {row['mae_all']:.3f} | {row['mae_shock']:.3f} | "
            f"{row['mae_2022']:.3f} |"
            for name, row in tab.iterrows()
        ]
    md = r / "results.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md
