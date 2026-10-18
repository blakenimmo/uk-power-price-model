"""What sets the GB wholesale electricity price? A fundamentals model for 2025.

Reads data/gb_power_2025.csv (made by get_data.py), builds a merit-order model of
price against residual demand (demand minus wind), tests it on held-out weeks,
and saves the figures in figures/ and the headline numbers in results.json.
"""
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# Chart styling: recessive axes and grid, one blue + one orange series colour
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "lines.linewidth": 2,
})

# ---------------------------------------------------------------- data
df = pd.read_csv(HERE / "data" / "gb_power_2025.csv")
df["time"] = pd.to_datetime(df["date"]) + pd.to_timedelta((df["period"] - 1) * 30, unit="min")
df["month"] = df["time"].dt.month
df["week"] = df["time"].dt.isocalendar().week.astype(int)
# Residual demand: what is left for gas, imports, storage etc. once wind has run.
# Solar is almost all embedded in the distribution network, so it already shows
# up as lower transmission demand and is not subtracted again.
df["residual_gw"] = (df["demand"] - df["wind"]) / 1000

# ---------------------------------------------------------------- model
# Merit order as a piecewise-linear curve in residual demand (knots every few GW,
# denser where the curve bends), plus a monthly level to absorb gas and carbon
# price changes, which Elexon does not publish.
KNOTS = [10, 12.5, 15, 20, 25, 30, 35, 39]


def design(d):
    x = d["residual_gw"].to_numpy()
    cols = [x] + [np.maximum(x - k, 0) for k in KNOTS]
    months = [(d["month"] == m).astype(float).to_numpy() for m in range(1, 13)]
    return np.column_stack(cols + months)


def fit(d, iters=30):
    """Least-absolute-deviation fit (iteratively reweighted least squares).

    Prices have rare spikes above GBP 1,000/MWh; an ordinary least-squares fit
    lets a handful of those bend the whole curve, so the model fits the median
    price instead.
    """
    X, y = design(d), d["price"].to_numpy()
    w = np.ones(len(y))
    for _ in range(iters):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        w = 1 / np.maximum(np.abs(y - X @ beta), 1.0)
    return beta


def predict(beta, d):
    return design(d) @ beta


# Out-of-sample test: hold out every 5th week in turn (blocked cross-validation,
# so neighbouring half-hours never sit on both sides of the split).
df["fold"] = df["week"] % 5
df["pred_cv"] = np.nan
df["pred_baseline"] = np.nan
for f in range(5):
    train, test = df[df.fold != f], df[df.fold == f]
    df.loc[test.index, "pred_cv"] = predict(fit(train), test)
    df.loc[test.index, "pred_baseline"] = test["month"].map(train.groupby("month")["price"].mean())


def scores(pred):
    err = df["price"] - df[pred]
    return {"r2": 1 - (err ** 2).sum() / ((df.price - df.price.mean()) ** 2).sum(),
            "mae": err.abs().mean()}


model, base = scores("pred_cv"), scores("pred_baseline")
daily = df.groupby("date")[["price", "pred_cv"]].mean()
daily_r2 = 1 - ((daily.price - daily.pred_cv) ** 2).sum() / ((daily.price - daily.price.mean()) ** 2).sum()

beta = fit(df)  # final model on the full year


def slope(res_gw, h=0.25):
    """d(price)/d(residual demand) in GBP/MWh per GW, from the fitted curve."""
    probe = pd.DataFrame({"residual_gw": [res_gw - h, res_gw + h], "month": [1, 1]})
    p = predict(beta, probe)
    return (p[1] - p[0]) / (2 * h)


# One extra GW of wind lowers residual demand by one GW, so the wind effect is
# minus the slope of the curve, averaged over the year's actual conditions.
avg_wind_effect = -np.mean([slope(r) for r in df["residual_gw"].sample(2000, random_state=0)])
neg_share_low = (df.loc[df.residual_gw < 10, "price"] < 0).mean()

results = {
    "half_hours": int(len(df)),
    "mean_price_gbp_mwh": round(df.price.mean(), 1),
    "half_hours_negative_price": int((df.price < 0).sum()),
    "share_negative_when_residual_below_10gw": round(float(neg_share_low), 2),
    "cv_r2_half_hourly": round(model["r2"], 2), "cv_mae_half_hourly": round(model["mae"], 1),
    "baseline_r2_month_mean_only": round(base["r2"], 2), "baseline_mae": round(base["mae"], 1),
    "cv_r2_daily_average": round(float(daily_r2), 2),
    "price_drop_per_extra_gw_wind_gbp_mwh": round(float(-avg_wind_effect), 1),
    "slope_at_12gw": round(slope(12), 1), "slope_at_22gw": round(slope(22), 1),
    "slope_at_38gw": round(slope(38), 1),
    "corr_price_residual": round(float(np.corrcoef(df.price, df.residual_gw)[0, 1]), 2),
}
(HERE / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))

# ---------------------------------------------------------------- figure 1: merit order
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df.residual_gw, df.price, s=3, alpha=0.12, color=BLUE, linewidths=0)
bins = np.arange(5, 45, 1)
mid = df.groupby(pd.cut(df.residual_gw, bins), observed=True)["price"].median()
ax.plot([b.mid for b in mid.index], mid.values, color=INK, lw=2, label="median price")
grid = pd.DataFrame({"residual_gw": np.linspace(6, 43, 200), "month": 1})
level = beta[len(KNOTS) + 1:].mean()  # average monthly level
curve = design(grid)[:, :len(KNOTS) + 1] @ beta[:len(KNOTS) + 1] + level
ax.plot(grid.residual_gw, curve, color=ORANGE, lw=2, ls="--", label="fitted merit-order curve")
ax.set_ylim(-60, 260)
ax.set_xlim(5, 44)
ax.set_xlabel("Residual demand: transmission demand minus wind (GW)")
ax.set_ylabel("Price (£/MWh)")
ax.set_title("GB power price follows the merit order (2025, half-hourly)")
for x, y, txt in [(9, 165, "Wind + nuclear\nset the price"),
                  (22, 165, "Gas plants set the price"),
                  (39.5, 75, "Scarcity:\npeaking plant")]:
    ax.text(x, y, txt, ha="center", va="top", color=INK2, fontsize=9)
ax.axhline(0, color=INK2, lw=0.8)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "merit_order.png", dpi=200)

# ---------------------------------------------------------------- figure 2: a windy week
# Pick the week with the biggest swing in wind, to show price moving against it.
wk = df.groupby("week")["wind"].agg(lambda w: w.max() - w.min()).idxmax()
w = df[df.week == wk]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
a1.plot(w.time, w.wind / 1000, color=BLUE)
a1.set_ylabel("Wind (GW)")
a1.set_title(f"Low wind, high prices (week of {w.time.min():%d %b %Y})")
a2.plot(w.time, w.price, color=ORANGE)
a2.set_ylabel("Price (£/MWh)")
a2.axhline(0, color=INK2, lw=0.8)
a2.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
fig.tight_layout()
fig.savefig(FIG / "windy_week.png", dpi=200)

# ---------------------------------------------------------------- figure 3: out-of-sample fit
fig, ax = plt.subplots(figsize=(8, 4.2))
d = daily.copy()
d.index = pd.to_datetime(d.index)
ax.plot(d.index, d.price, color=INK2, lw=1.5, label="actual")
ax.plot(d.index, d.pred_cv, color=ORANGE, lw=1.5, label="model (predicted on unseen weeks)")
ax.set_ylabel("Daily average price (£/MWh)")
ax.set_title(f"Out-of-sample fit: R² = {daily_r2:.2f} on daily averages")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "model_fit.png", dpi=200)
print("figures saved")
