# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 2: Load input data from CSV files
# Each file provides part of the necessary data for credit risk modeling

companies = pd.read_csv('data/companies.csv')                    # Contains company metadata
portfolio = pd.read_csv('data/portfolio2_composition.csv')       # Portfolio exposure details
ratings = pd.read_csv('data/ratings.csv')                        # Credit ratings for companies
transitions = pd.read_csv('data/rating_transitions.csv', index_col=0) # Transition matrix with PDs
stock_returns = pd.read_csv('data/healthcare_stock_returns.csv', parse_dates=['Date']) # Individual stock returns
index_returns = pd.read_csv('data/healthcare_index.csv', parse_dates=['Date'])  # Market index returns

# Step 3: Merge portfolio, ratings, and company data into one master table
# Creates a unified view of all portfolio exposures with company names and credit ratings

portfolio_full = (portfolio.merge(ratings, on='CompanyID').merge(companies, on='CompanyID'))

# Step 4: Count number of companies by credit rating in the portfolio
rating_counts = portfolio_full['Rating'].value_counts().sort_index()
print("Rating distribution:\n", rating_counts)

# Step 5: Visualize the credit rating distribution of portfolio companies
rating_counts.plot(kind='bar', title='Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Number of Companies')
plt.grid(True)
plt.show()

# Step 6: Map each credit rating to its probability of default (PD) and corresponding threshold
# # Thresholds are inverse-normal of PDs used in Vasicek one-factor credit model
min_pd = 1e-6 # Avoid zero probabilities
transitions['PD'] = transitions['D'].clip(lower=min_pd)
transitions['Threshold'] = transitions['PD'].apply(norm.ppf)

# Merge default probabilities and thresholds into portfolio
portfolio_full = portfolio_full.merge(transitions[['PD', 'Threshold']],left_on='Rating',right_index=True,how='left')
portfolio_full.rename(columns={'Threshold': 'threshold'}, inplace=True)

# Step 6.5: Add constant LGD (Loss Given Default) value of 45%
portfolio_full['LGD'] = 0.45  # This means only 45% of exposure is lost on default

# Step 7: Calculate asset correlation (rho) with market index for each company
# Required for Vasicek model; rho is correlation between firm's returns and market index
# First, rename 'Index' column to 'IndexReturn' for clarity
if 'Index' in index_returns.columns:
    index_returns.rename(columns={'Index': 'IndexReturn'}, inplace=True)
# Merge company returns and index returns on date
merged_returns = stock_returns.merge(index_returns[['Date', 'IndexReturn']], on='Date', how='inner') #Inner join = intersection of keys.
#matching rows based on a common key, keeping only the rows where the key exists in both tables.


# Compute correlation of each company's returns with market index
correlations = {}
for col in stock_returns.columns:
    if col != 'Date':
        correlations[col] = merged_returns[col].corr(merged_returns['IndexReturn'])  # excluding date from field and calculating correlation in each column

# Convert correlation dictionary into a DataFrame and extract CompanyID
correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['rho']) # using row indices and converting dictionaries to df
correlation_df.index.name = 'CompanyColumn' # indexing with name CompanyColumn
correlation_df.reset_index(inplace=True) # ( normal indexing like 0,1,2 added and CompanyColumn is not indexing column anymore)
correlation_df['CompanyID'] = correlation_df['CompanyColumn'].str.extract(r'(\d+)').astype(int)

# Merge asset correlation (rho) values into the master portfolio table
portfolio_full = portfolio_full.merge(correlation_df[['CompanyID', 'rho']], on='CompanyID', how='left')

# Fill any missing correlations with a default value (e.g. 0.2) and clip between 0.01 and 0.9
portfolio_full['rho'] = portfolio_full['rho'].fillna(0.2).clip(lower=0.01, upper=0.9)

# Step 8: Run Monte Carlo simulation to estimate credit portfolio loss distribution
# Uses Vasicek model with idiosyncratic and systematic shocks

np.random.seed(42)
num_simulations = 10000
losses = []

for _ in range(num_simulations):
    F = np.random.normal()  # Common market factor
    total_loss = 0
    for _, row in portfolio_full.iterrows():
        epsilon = np.random.normal(0,1)  # Idiosyncratic shock
        rho = row['rho']
        asset_return = rho * F + np.sqrt(1 - rho ** 2) * epsilon
        if asset_return < row['threshold']:
            total_loss += row['Exposure'] * row['LGD']  # <--- LGD applied here
    losses.append(total_loss)

# Step 9: Calculate expected loss, VaR and CVaR from simulations
losses = np.array(losses)
expected_loss = losses.mean()
VaR_99 = np.percentile(losses, 99)
expected_shortfall_99 = losses[losses > VaR_99].mean()

print("\nMonte Carlo Simulation Results (with LGD = 45%)")

print(f"Expected Loss: ${expected_loss:,.2f}")
print(f"99% Value-at-Risk (VaR): ${VaR_99:,.2f}")
print(f"99% Expected Shortfall (VaR): ${expected_shortfall_99:,.2f}")

# Step 10: Plot histogram of simulated losses
plt.figure(figsize=(10, 6))
plt.hist(losses, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(expected_loss, color='blue', linestyle='--', linewidth=2, label=f'Expected Loss = ${expected_loss:,.0f}')
plt.axvline(VaR_99, color='red', linestyle='--', linewidth=2, label=f'99% VaR = ${VaR_99:,.0f}')
plt.axvline(expected_shortfall_99, color='purple', linestyle='--', linewidth=2,
            label=f'Expected Shortfall= ${expected_shortfall_99:,.0f}')
plt.title('Simulated Loss Distribution of Credit Portfolio')
plt.xlabel('Portfolio Loss ($)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Analytical approximation
def Analytical_var(portfolio_df, confidence_level=0.99):
    ### Calculates portfolio VaR using the Vasicek one-factor model (with individual rhos) at the given confidence level.
    # Reverse the threshold back to PDs
    PDs = norm.cdf(portfolio_df['threshold'])  # Back out PDs from threshold
    exposures = portfolio_df['Exposure'].values
    LGDs = portfolio_df['LGD'].values
    rhos = portfolio_df['rho'].values

    z = norm.ppf(1 - confidence_level)  # Confidence level threshold

    # Vasicek formula # Conditional PD under market stress
    conditional_PDs = norm.cdf((norm.ppf(PDs) - rhos * z) / np.sqrt(1 - rhos ** 2))

    # Individual loss at confidence level
    losses = exposures * LGDs * conditional_PDs

    # Calculate expected and stressed losses
    expected_loss = np.sum(PDs * exposures * LGDs)
    VaR = np.sum(losses)

    return expected_loss, VaR

# Run analytical approximation at 99.9% confidence
expected_loss_analytical, VaR_Analytical = Analytical_var(portfolio_full, confidence_level=0.99)

print("\nAnalytical VaR Using Vasicek Model (99%)")
print(f"Analytical Expected Loss: ${expected_loss_analytical:,.2f}")
print(f"Analytical 99% VaR: ${VaR_Analytical:,.2f}")

# Step 12: Additional Risk Measures
# Estimate Regulatory Capital Requirement as( VaR - Expected Loss )
capital_requirement = VaR_Analytical - expected_loss_analytical
print(f"Estimated Capital Requirement (99% VaR - Expected Loss): ${capital_requirement:,.2f}")

# Sensitivity Analysis: Impact of rho (asset correlation) on Analytical VaR
print("\nSensitivity of Analytical VaR to Different Correlation (rho) Assumptions:")
for test_rho in [0.1, 0.3, 0.5, 0.7]:
    portfolio_test = portfolio_full.copy()
    portfolio_test['rho'] = test_rho  # Set fixed rho for testing
    el_test, var_test = Analytical_var(portfolio_test, confidence_level=0.99)
    capital_test = var_test - el_test
    print(f"rho = {test_rho:.1f} | Expected Loss = ${el_test:,.0f} | VaR(99%) = ${var_test:,.0f} | Capital = ${capital_test:,.0f}")
