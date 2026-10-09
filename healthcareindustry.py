"""
Credit Risk Analysis of Portfolio 2 (Healthcare, 60 obligors)
=============================================================
Final Assignment - Introduction to Credit Risk and Application in Python

Model: one-factor Gaussian copula (Vasicek / Merton-type threshold model)
        X_i = sqrt(rho_i) * Z + sqrt(1 - rho_i) * eps_i,   Z, eps_i ~ iid N(0,1)
        obligor i defaults  <=>  X_i < Phi^{-1}(PD_i)
        Loss = sum_i EAD_i * LGD * 1{default_i}

Pipeline
  1. Data processing      : load, clean, merge companies / ratings / exposures
  2. Descriptive analysis : rating distribution, exposure concentration (HHI, Lorenz)
  3. Model parameters     : PD from the transition matrix, asset correlation
                            estimated three ways (index-factor, pairwise, Basel IRB)
  4. Monte Carlo          : simulate portfolio losses
  5. Risk metrics         : EL, VaR, ES, economic capital
  6. Extensions           : analytical Vasicek VaR, Basel IRB capital,
                            correlation sensitivity, stress tests

Run:  python healthcareindustry.py
All plots and tables are written to ./output/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # writing figures to file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# ----------------------------------------------------------------------------
# Configuration (all assumptions in one place)
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "output"

FILES = {
    "companies": "companies.csv",
    "ratings": "ratings.csv",
    "portfolio": "portfolio2_composition.csv",
    "transitions": "rating_transitions.csv",
    "stock_returns": "healthcare_stock_returns.csv",
    "index": "healthcare_index.csv",
}

RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C"]
LGD = 0.45            # Basel IRB foundation LGD for senior unsecured corporate debt
PD_FLOOR = 0.0003     # Basel floor of 3 bp (AAA/AA have PD = 0 in the matrix)
HORIZON_YEARS = 1     # risk horizon (1-year transition matrix)
N_SIM = 200_000       # Monte Carlo scenarios
CONF_LEVELS = [0.95, 0.99, 0.999]
SEED = 42
MATURITY = 2.5        # effective maturity for the Basel IRB maturity adjustment


# ----------------------------------------------------------------------------
# 1. Data processing
# ----------------------------------------------------------------------------
def load_data(data_dir=DATA_DIR):
    """Read all CSV files into a dict of DataFrames."""
    d = {}
    d["companies"] = pd.read_csv(data_dir / FILES["companies"])
    d["ratings"] = pd.read_csv(data_dir / FILES["ratings"])
    d["portfolio"] = pd.read_csv(data_dir / FILES["portfolio"])
    d["transitions"] = pd.read_csv(data_dir / FILES["transitions"], index_col=0)
    d["returns"] = pd.read_csv(data_dir / FILES["stock_returns"],
                               parse_dates=["Date"]).set_index("Date")
    d["index"] = pd.read_csv(data_dir / FILES["index"],
                             parse_dates=["Date"]).set_index("Date")
    return d


def clean_and_merge(d):
    """Merge portfolio with company info and ratings; report data issues."""
    for name in ["companies", "ratings", "portfolio"]:
        df = d[name]
        n_dup = df.duplicated("CompanyID").sum()
        if n_dup:
            print(f"  [{name}] dropping {n_dup} duplicate CompanyIDs")
            d[name] = df.drop_duplicates("CompanyID")

    # Strip stray whitespace / case in ratings
    d["ratings"]["Rating"] = d["ratings"]["Rating"].astype(str).str.strip().str.upper()

    pf = (d["portfolio"]
          .merge(d["companies"], on="CompanyID", how="left", validate="1:1")
          .merge(d["ratings"], on="CompanyID", how="left", validate="1:1"))

    missing = pf[["CompanyName", "Industry", "Rating"]].isna().sum()
    print("  Missing values after merge:\n" + missing.to_string())
    # Policy for missing ratings: assign the worst observed rating in the
    # portfolio (conservative). Not triggered for portfolio 2, but kept so the
    # code is robust.
    if pf["Rating"].isna().any():
        worst = max(pf["Rating"].dropna(), key=RATING_ORDER.index)
        print(f"  -> {pf['Rating'].isna().sum()} missing ratings set to {worst}")
        pf["Rating"] = pf["Rating"].fillna(worst)

    bad = pf[(pf["Exposure"] <= 0) | pf["Exposure"].isna()]
    if len(bad):
        print(f"  -> dropping {len(bad)} rows with invalid exposure")
        pf = pf.drop(bad.index)

    # Missing daily returns: count only; correlations use pairwise-complete obs
    na_share = d["returns"].isna().mean().mean()
    print(f"  Stock returns: {d['returns'].shape[0]} days x {d['returns'].shape[1]} "
          f"firms, {na_share:.1%} missing (handled by pairwise deletion)")
    return pf.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Descriptive analysis
# ----------------------------------------------------------------------------
def describe_portfolio(pf):
    total = pf["Exposure"].sum()
    w = pf["Exposure"] / total
    hhi = (w ** 2).sum()
    by_rating = (pf.groupby("Rating")["Exposure"]
                 .agg(["count", "sum"])
                 .reindex([r for r in RATING_ORDER if r in pf["Rating"].unique()]))
    by_rating["share"] = by_rating["sum"] / total
    stats = {
        "n_obligors": len(pf),
        "total_exposure": total,
        "mean_exposure": pf["Exposure"].mean(),
        "max_exposure": pf["Exposure"].max(),
        "top10_share": w.sort_values(ascending=False).head(10).sum(),
        "HHI": hhi,
        "effective_n": 1 / hhi,      # number of equal loans with same HHI
    }
    return stats, by_rating


def plot_descriptives(pf, by_rating, out):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].bar(by_rating.index, by_rating["count"], color="#3a6ea5")
    ax[0].set_title("Number of obligors by rating")
    ax[0].set_ylabel("Count")

    ax[1].bar(by_rating.index, by_rating["sum"] / 1e3, color="#d98c3a")
    ax[1].set_title("Exposure by rating")
    ax[1].set_ylabel("EAD (thousands)")

    # Lorenz curve = visual exposure concentration
    e = np.sort(pf["Exposure"].values)
    cum = np.insert(np.cumsum(e) / e.sum(), 0, 0)
    x = np.linspace(0, 1, len(cum))
    gini = 1 - 2 * np.trapezoid(cum, x)
    ax[2].plot(x, cum, label=f"Portfolio (Gini = {gini:.2f})")
    ax[2].plot([0, 1], [0, 1], "k--", lw=1, label="Equal exposures")
    ax[2].set_title("Lorenz curve of exposures")
    ax[2].set_xlabel("Share of obligors")
    ax[2].set_ylabel("Share of exposure")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(out / "fig1_portfolio_descriptives.png", dpi=150)
    plt.close(fig)
    return gini


# ----------------------------------------------------------------------------
# 3. Model parameters: PD and correlation
# ----------------------------------------------------------------------------
def pd_from_transitions(trans, horizon=HORIZON_YEARS, floor=PD_FLOOR):
    """PD per rating = 'D' column of P^horizon (Markov chain assumption)."""
    P = trans.loc[trans.index, trans.index].values      # square matrix incl. D
    Ph = np.linalg.matrix_power(P, horizon)
    pd_raw = pd.Series(Ph[:, list(trans.index).index("D")], index=trans.index)
    pd_raw = pd_raw.drop("D")
    return pd_raw.clip(lower=floor), pd_raw


def cumulative_pd_table(trans, years=(1, 3, 5)):
    """Multi-year cumulative PDs, useful to show the term structure of risk."""
    P = trans.values
    j = list(trans.columns).index("D")
    out = {f"{t}y": np.linalg.matrix_power(P, t)[:, j] for t in years}
    return pd.DataFrame(out, index=trans.index).drop("D")


def corr_index_factor(returns, index_ret, firm_ids):
    """
    Approach A: the healthcare index proxies the systematic factor Z.
    Factor loading = corr(r_i, r_index), asset correlation rho_i = loading^2.
    Also returns a t-test of H0: corr = 0.
    """
    cols = [f"Company_{i}" for i in firm_ids]
    r = returns[cols]
    beta = r.corrwith(index_ret)
    n = r.notna().sum()
    tstat = beta * np.sqrt((n - 2) / (1 - beta ** 2))
    df = pd.DataFrame({"loading": beta.values, "rho": beta.values ** 2,
                       "t_stat": tstat.values}, index=firm_ids)
    return df


def corr_pairwise(returns, firm_ids):
    """
    Approach B: average pairwise correlation of firm returns.
    In a one-factor model corr(r_i, r_j) = sqrt(rho_i rho_j), so the mean
    off-diagonal correlation estimates a homogeneous rho directly.
    Also returns the largest eigenvalue vs. the Marchenko-Pastur noise bound.
    """
    cols = [f"Company_{i}" for i in firm_ids]
    C = returns[cols].corr()        # pairwise-complete observations
    off = C.values[np.triu_indices(len(cols), 1)]
    T, N = returns.shape[0], len(cols)
    mp_bound = (1 + np.sqrt(N / T)) ** 2
    lam_max = np.linalg.eigvalsh(C.values).max()
    return off.mean(), C, lam_max, mp_bound


def corr_basel(pd_):
    """Approach C: Basel II IRB supervisory correlation for corporates (12%-24%)."""
    w = (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


# ----------------------------------------------------------------------------
# 4. Monte Carlo simulation
# ----------------------------------------------------------------------------
def simulate_losses(ead, pd_, rho, lgd=LGD, n_sim=N_SIM, seed=SEED, chunk=50_000):
    """
    Simulate portfolio losses under the one-factor Gaussian copula.
    ead, pd_, rho: arrays of length N (rho may be scalar). Returns losses
    (n_sim,) and number of defaults (n_sim,). Done in chunks to save memory.
    """
    rng = np.random.default_rng(seed)
    ead, pd_ = np.asarray(ead, float), np.asarray(pd_, float)
    rho = np.broadcast_to(np.asarray(rho, float), ead.shape)
    thresh = norm.ppf(pd_)
    a, b = np.sqrt(rho), np.sqrt(1 - rho)
    loss_given = ead * lgd

    losses = np.empty(n_sim)
    n_def = np.empty(n_sim, dtype=int)
    for s in range(0, n_sim, chunk):
        m = min(chunk, n_sim - s)
        Z = rng.standard_normal((m, 1))
        eps = rng.standard_normal((m, len(ead)))
        X = a * Z + b * eps
        D = X < thresh
        losses[s:s + m] = D @ loss_given
        n_def[s:s + m] = D.sum(1)
    return losses, n_def


# ----------------------------------------------------------------------------
# 5. Risk metrics
# ----------------------------------------------------------------------------
def risk_metrics(losses, levels=CONF_LEVELS):
    el = losses.mean()
    res = {"EL": el, "UL (std)": losses.std()}
    for q in levels:
        var = np.quantile(losses, q)
        es = losses[losses >= var].mean()
        res[f"VaR {q:.1%}"] = var
        res[f"ES {q:.1%}"] = es
        res[f"EC {q:.1%} (VaR-EL)"] = var - el
    return res


def vasicek_var(ead, pd_, rho, q, lgd=LGD):
    """
    Analytical ASRF / Vasicek VaR: conditional EL at the q-quantile of Z.
    Exact only for an infinitely granular portfolio.
    """
    ead, pd_ = np.asarray(ead), np.asarray(pd_)
    rho = np.broadcast_to(np.asarray(rho, float), ead.shape)
    cond_pd = norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(q)) / np.sqrt(1 - rho))
    return (ead * lgd * cond_pd).sum()


def basel_irb_capital(ead, pd_, lgd=LGD, m=MATURITY):
    """Basel II/III IRB capital requirement K (corporates) and RWA."""
    pd_ = np.asarray(pd_)
    rho = corr_basel(pd_)
    b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
    cond = norm.cdf((norm.ppf(pd_) + np.sqrt(rho) * norm.ppf(0.999)) / np.sqrt(1 - rho))
    K = lgd * (cond - pd_) * (1 + (m - 2.5) * b) / (1 - 1.5 * b)
    capital = (K * ead).sum()
    return capital, capital * 12.5


# ----------------------------------------------------------------------------
# 6. Plots for simulation and extensions
# ----------------------------------------------------------------------------
def plot_correlation_estimates(fac, C, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(fac["loading"], bins=20, color="#3a6ea5", edgecolor="white")
    se = 1 / np.sqrt(1274)
    for s in (-1.96 * se, 1.96 * se):
        ax[0].axvline(s, color="red", ls="--", lw=1)
    ax[0].set_title("Firm-index correlations (dashed: 95% band under H0: 0)")
    ax[0].set_xlabel("corr(r_i, r_index)")
    im = ax[1].imshow(C.values, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
    ax[1].set_title("Pairwise return correlations (60 firms)")
    ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "fig2_correlation_estimates.png", dpi=150)
    plt.close(fig)


def plot_loss_distribution(results, out):
    """results: dict label -> (losses, metrics)"""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    colors = ["#999999", "#3a6ea5", "#c0392b"]
    for (lab, (L, met)), c in zip(results.items(), colors):
        ax[0].hist(L / 1e3, bins=60, density=True, alpha=0.45, color=c, label=lab)
    main_lab = list(results)[-1]
    L, met = results[main_lab]
    for k, ls in [("VaR 99.0%", "--"), ("VaR 99.9%", "-"), ("ES 99.9%", ":")]:
        ax[0].axvline(met[k] / 1e3, color="black", ls=ls, lw=1.2,
                      label=f"{k} ({main_lab.split(' (')[0]}) = {met[k]/1e3:.0f}k")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Portfolio loss (thousands)")
    ax[0].set_ylabel("Density (log scale)")
    ax[0].set_title("Simulated 1-year loss distribution")
    ax[0].legend(fontsize=8)

    for (lab, (L, _)), c in zip(results.items(), colors):
        xs = np.sort(L)
        ax[1].plot(xs / 1e3, 1 - np.arange(1, len(xs) + 1) / len(xs), color=c, label=lab)
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-4, 1)
    ax[1].set_xlabel("Loss threshold (thousands)")
    ax[1].set_ylabel("P(Loss > x)")
    ax[1].set_title("Tail probability: dependence fattens the tail")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_loss_distribution.png", dpi=150)
    plt.close(fig)


def plot_sensitivity(sens, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sens.index, sens["MC VaR 99.9%"] / 1e3, "o-", label="MC VaR 99.9%")
    ax.plot(sens.index, sens["MC ES 99.9%"] / 1e3, "s-", label="MC ES 99.9%")
    ax.plot(sens.index, sens["Analytical VaR 99.9%"] / 1e3, "^--", label="Vasicek VaR 99.9%")
    ax.plot(sens.index, sens["EL"] / 1e3, "k:", label="Expected loss")
    ax.set_xlabel("Asset correlation rho")
    ax.set_ylabel("Thousands")
    ax.set_title("Sensitivity of tail risk to the correlation assumption")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig4_correlation_sensitivity.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

    print("=" * 70, "\n1. DATA PROCESSING\n" + "=" * 70)
    d = load_data()
    pf = clean_and_merge(d)
    print(f"  Portfolio: {len(pf)} obligors, industries: {', '.join(pf['Industry'].unique())}")

    print("=" * 70, "\n2. DESCRIPTIVE ANALYSIS\n" + "=" * 70)
    stats, by_rating = describe_portfolio(pf)
    gini = plot_descriptives(pf, by_rating, OUT_DIR)
    stats["Gini"] = gini
    for k, v in stats.items():
        print(f"  {k:<16}{v:>14,.3f}")
    print(by_rating.to_string())

    print("=" * 70, "\n3. MODEL PARAMETERS\n" + "=" * 70)
    pd_map, pd_raw = pd_from_transitions(d["transitions"])
    print("  1-year PD by rating (raw -> floored):")
    print(pd.DataFrame({"raw": pd_raw, "used": pd_map}).to_string())
    print("  Cumulative PDs (Markov chain, P^t):")
    print(cumulative_pd_table(d["transitions"]).to_string())
    pf["PD"] = pf["Rating"].map(pd_map)
    pf["EL"] = pf["Exposure"] * LGD * pf["PD"]

    # --- correlation, three approaches
    fac = corr_index_factor(d["returns"], d["index"]["Return"], pf["CompanyID"])
    n_sig = (fac["t_stat"].abs() > 1.96).sum()
    rho_pair, C, lam_max, mp = corr_pairwise(d["returns"], pf["CompanyID"])
    pf["rho_basel"] = corr_basel(pf["PD"].values)
    print(f"  A) Index factor : mean loading = {fac['loading'].mean():.4f}, "
          f"mean rho = {fac['rho'].mean():.5f}, significant at 5%: {n_sig}/60")
    print(f"  B) Pairwise     : mean pairwise corr = {rho_pair:.4f}; "
          f"largest eigenvalue {lam_max:.3f} vs noise bound {mp:.3f}")
    print(f"  C) Basel IRB    : rho range {pf['rho_basel'].min():.3f}-"
          f"{pf['rho_basel'].max():.3f}, exposure-weighted "
          f"{np.average(pf['rho_basel'], weights=pf['Exposure']):.3f}")
    plot_correlation_estimates(fac, C, OUT_DIR)
    pf["rho_index"] = pf["CompanyID"].map(fac["rho"])

    print("=" * 70, "\n4-5. SIMULATION AND RISK METRICS\n" + "=" * 70)
    ead, p = pf["Exposure"].values, pf["PD"].values
    scenarios = {
        "Independent (rho=0)": 0.0,
        "Empirical (index factor)": pf["rho_index"].values,
        "Basel IRB rho (main)": pf["rho_basel"].values,
    }
    sim_results, table = {}, {}
    for lab, rho in scenarios.items():
        L, nd = simulate_losses(ead, p, rho)
        met = risk_metrics(L)
        met["P(no default)"] = (nd == 0).mean()
        met["Max defaults"] = nd.max()
        met["Vasicek VaR 99.9%"] = vasicek_var(ead, p, rho, 0.999)
        sim_results[lab] = (L, met)
        table[lab] = met
    table = pd.DataFrame(table)
    print(table.to_string(float_format=lambda x: f"{x:,.0f}" if abs(x) > 1 else f"{x:.4f}"))
    table.to_csv(OUT_DIR / "risk_metrics.csv")
    plot_loss_distribution(sim_results, OUT_DIR)

    # MC accuracy: VaR estimates across different seeds
    reps = [np.quantile(simulate_losses(ead, p, pf["rho_basel"].values,
                                        n_sim=50_000, seed=s)[0], 0.999)
            for s in range(10)]
    print(f"  MC stability (10 x 50k sims): VaR99.9 mean {np.mean(reps):,.0f}, "
          f"std {np.std(reps):,.0f}")

    print("=" * 70, "\n6. EXTENSIONS\n" + "=" * 70)
    capital, rwa = basel_irb_capital(ead, p)
    main_met = sim_results["Basel IRB rho (main)"][1]
    print(f"  Basel IRB capital (M={MATURITY}): {capital:,.0f}  "
          f"(RWA {rwa:,.0f}, {capital/ead.sum():.2%} of EAD)")
    print(f"  Economic capital 99.9% (MC)    : {main_met['EC 99.9% (VaR-EL)']:,.0f}")
    print(f"  Granularity gap (MC - Vasicek) : "
          f"{main_met['VaR 99.9%'] - main_met['Vasicek VaR 99.9%']:,.0f}")

    # Correlation sensitivity (homogeneous rho)
    sens = {}
    for r in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        L, _ = simulate_losses(ead, p, r, n_sim=100_000)
        m = risk_metrics(L, [0.999])
        sens[r] = {"EL": m["EL"], "MC VaR 99.9%": m["VaR 99.9%"],
                   "MC ES 99.9%": m["ES 99.9%"],
                   "Analytical VaR 99.9%": vasicek_var(ead, p, r, 0.999)}
    sens = pd.DataFrame(sens).T
    print("  Correlation sensitivity:\n" + sens.to_string(float_format="{:,.0f}".format))
    sens.to_csv(OUT_DIR / "correlation_sensitivity.csv")
    plot_sensitivity(sens, OUT_DIR)

    # Stress tests (all with Basel rho recomputed from stressed PD)
    downgrade = {r: RATING_ORDER[min(i + 1, len(RATING_ORDER) - 1)]
                 for i, r in enumerate(RATING_ORDER)}
    stresses = {
        "Baseline": (p, LGD),
        "PD x 2": (np.minimum(p * 2, 0.999), LGD),
        "1-notch downgrade": (pf["Rating"].map(downgrade).map(pd_map).values, LGD),
        "LGD 60% (downturn)": (p, 0.60),
        "PD x 2 & LGD 60%": (np.minimum(p * 2, 0.999), 0.60),
    }
    st = {}
    for lab, (ps, lgd) in stresses.items():
        L, _ = simulate_losses(ead, ps, corr_basel(ps), lgd=lgd, n_sim=100_000)
        m = risk_metrics(L, [0.99, 0.999])
        st[lab] = {k: m[k] for k in ["EL", "VaR 99.0%", "VaR 99.9%", "ES 99.9%"]}
    st = pd.DataFrame(st).T
    print("  Stress tests:\n" + st.to_string(float_format="{:,.0f}".format))
    st.to_csv(OUT_DIR / "stress_tests.csv")

    # Save the cleaned portfolio and list the largest expected-loss contributors
    pf[["CompanyID", "CompanyName", "Rating", "Exposure", "PD",
        "rho_basel", "rho_index", "EL"]].to_csv(OUT_DIR / "portfolio_clean.csv",
                                                index=False)
    top = pf.sort_values("EL", ascending=False).head(5)
    print("  Largest EL contributors:\n" +
          top[["CompanyID", "Rating", "Exposure", "PD", "EL"]].to_string(index=False))
    print(f"\nDone. Figures and tables written to {OUT_DIR}")


if __name__ == "__main__":
    main()