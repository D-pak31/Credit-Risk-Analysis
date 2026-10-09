# Credit Risk Analysis of Portfolio 2 (Healthcare)

Final Assignment, *Introduction to Credit Risk and Application in Python*
Author: Dipak Raj Paneru

This project estimates the one-year credit risk of a loan portfolio of 60 healthcare companies (total exposure 1,454,724). It models dependence between borrowers with a one-factor Gaussian copula, simulates portfolio losses with Monte Carlo, and reports Expected Loss, Value-at-Risk and Expected Shortfall. It also compares the simulated results with the analytical Vasicek VaR, the Basel IRB capital requirement, a correlation sensitivity analysis and stress tests.

## How to run

```bash
pip install numpy pandas scipy matplotlib
python credit_risk_portfolio2.py
```

- Put the script in the same folder as the original data files (filenames unchanged).
- Runtime is about 10 seconds. Results are printed to the console.
- All figures and tables are written to a new `output/` folder.
- A fixed random seed (42) makes every run reproducible.

## Input files

| File | Content |
|---|---|
| `portfolio2_composition.csv` | CompanyID and exposure (EAD) of the 60 loans |
| `companies.csv` | Company name and industry |
| `ratings.csv` | Credit rating per company |
| `rating_transitions.csv` | 1-year rating transition matrix, including default (D) |
| `healthcare_stock_returns.csv` | Daily returns of the 60 firms, 2018–2022 |
| `healthcare_index.csv` | Daily healthcare index level and return |

## What the script does

The code is organised into functions, one per step. `main()` calls them in order.

### 1. Data processing: `load_data()`, `clean_and_merge()`
- Loads all six CSV files.
- Checks for duplicate CompanyIDs and normalises the rating strings.
- Merges portfolio → companies → ratings, with each merge validated as one-to-one.
- Missing ratings would be assigned the worst observed rating, a conservative choice. None occur in Portfolio 2.
- Rows with invalid or missing exposure are dropped.
- Missing stock returns (3.0%) are handled with pairwise deletion when computing correlations. Listwise deletion would keep only 205 of 1,274 days.

### 2. Descriptive analysis: `describe_portfolio()`, `plot_descriptives()`
- Number of obligors and exposure by rating.
- Exposure concentration: top-10 share, Herfindahl–Hirschman Index (HHI), effective number of obligors (1/HHI) and Gini coefficient.
- Figure 1 shows a rating histogram, exposure by rating and a Lorenz curve.

### 3. Model parameters
**Probability of default:** `pd_from_transitions()`, `cumulative_pd_table()`
- The 1-year PD is the default column of the transition matrix.
- AAA and AA have PD = 0 in the matrix. They are floored at 0.03% (Basel floor).
- 3- and 5-year cumulative PDs are computed as matrix powers Pᵗ (Markov chain assumption).

**Asset correlation, estimated three ways:**
| Function | Approach | Result |
|---|---|---|
| `corr_index_factor()` | Correlation of each stock with the healthcare index; ρᵢ = corr², with t-test | mean ρ ≈ 0.0009, only 4/60 significant |
| `corr_pairwise()` | Mean pairwise stock correlation; largest eigenvalue vs. Marchenko–Pastur noise bound | ≈ −0.001; 1.457 < 1.481 (pure noise) |
| `corr_basel()` | Basel IRB supervisory formula | ρ = 0.130–0.238 |

The data show no detectable common factor, but zero correlation within one industry is implausible. The Basel ρ is therefore the main assumption. Figure 2 shows the correlation estimates.

**Other assumptions:** LGD is 45% (Basel foundation IRB), EAD equals the given exposure, and the horizon is 1 year.

### 4. Monte Carlo simulation: `simulate_losses()`
One-factor Gaussian copula:

```
X_i = sqrt(rho_i) * Z + sqrt(1 - rho_i) * eps_i      Z, eps_i ~ N(0,1)
default_i  if  X_i < Phi^-1(PD_i)
Loss = sum_i EAD_i * LGD * default_i
```

- 200,000 scenarios, simulated in chunks to limit memory use.
- Three dependence scenarios: independent (ρ = 0), empirical ρ, and Basel ρ (main).
- Accuracy check: 10 repeated runs of 50,000 scenarios give a 99.9% VaR standard deviation of about 3,300 (≈ 2.7%).

### 5. Risk metrics: `risk_metrics()`
Expected Loss, standard deviation (unexpected loss), VaR and Expected Shortfall at 95%, 99% and 99.9%, and economic capital (VaR − EL). Figure 3 shows the loss distribution and tail probabilities.

| Metric | Independent | Basel ρ (main) |
|---|---|---|
| Expected loss | 9,625 | 9,635 |
| VaR 99.9% | 61,118 | 123,170 |
| ES 99.9% | 68,365 | 147,745 |
| Economic capital 99.9% | 51,493 | 113,535 |

Dependence leaves the expected loss unchanged but roughly doubles the tail risk.

### 6. Extensions
- **Analytical VaR:** `vasicek_var()` gives the Vasicek/ASRF VaR at 99.9%, 98,850. The gap of about 24k to the Monte Carlo result reflects granularity: 60 loans versus the formula's infinitely many.
- **Capital requirement:** `basel_irb_capital()` gives Basel IRB capital K = 109,924 (7.56% of EAD) with maturity 2.5, and RWA = 1,374,049.
- **Correlation sensitivity:** VaR, ES and analytical VaR are computed for ρ from 0 to 0.5. VaR 99.9% rises from 60k to 312k (Figure 4).
- **Stress tests:** PD × 2, a one-notch downgrade, LGD 60%, and PD × 2 combined with LGD 60%. The combined scenario gives VaR 99.9% ≈ 222k.
- **Largest expected-loss contributors:** the script lists them. Firm 90 (CCC) is the largest.

## Output files (`output/`)

| File | Content |
|---|---|
| `fig1_portfolio_descriptives.png` | Rating histogram, exposure by rating, Lorenz curve |
| `fig2_correlation_estimates.png` | Firm–index correlations and pairwise correlation heatmap |
| `fig3_loss_distribution.png` | Simulated loss distribution and exceedance probabilities |
| `fig4_correlation_sensitivity.png` | Tail risk as a function of ρ |
| `risk_metrics.csv` | EL, VaR, ES and economic capital for the three dependence scenarios |
| `correlation_sensitivity.csv` | VaR/ES (MC and analytical) for ρ = 0 … 0.5 |
| `stress_tests.csv` | Stress test results |
| `portfolio_clean.csv` | Merged portfolio with PD, ρ and EL per obligor |

## Configuration

All assumptions are constants at the top of the script and can be changed in one place: `LGD`, `PD_FLOOR`, `HORIZON_YEARS`, `N_SIM`, `CONF_LEVELS`, `SEED`, `MATURITY`.

## Limitations

- The Gaussian copula has no tail dependence.
- LGD is fixed and not linked to the economic cycle.
- There is only one systematic factor.
- PDs come from a generic, time-homogeneous transition matrix.
- The return data are uninformative about dependence, so results depend on the assumed correlation.

See Section 6 of the report for details.
