# Step 1: Import required libraries
# These libraries are used for data handling, numerical computation, plotting, and statistical operations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 2: Load input data from CSV files
# These files contain:
# - companies: metadata of companies (like names, sectors)
# - portfolio: list of companies and exposures in the portfolio
# - ratings: credit ratings for each company
# - transitions: default probabilities from each rating
# - stock_returns: historical stock returns for each company
# - index_returns: returns of the healthcare index
companies = pd.read_csv('companies.csv')
portfolio = pd.read_csv('portfolio2_composition.csv')
ratings = pd.read_csv('ratings.csv')
transitions = pd.read_csv('rating_transitions.csv', index_col=0)
stock_returns = pd.read_csv('healthcare_stock_returns.csv', parse_dates=['Date'])
index_returns = pd.read_csv('healthcare_index.csv', parse_dates=['Date'])

# Step 3: Merge portfolio, ratings, and company data into one master table
# This will allow us to access all needed info (exposure, rating, etc.) for each company in one place
portfolio_full = (portfolio.merge(ratings, on='CompanyID').merge(companies, on='CompanyID'))

 # Step 4: Counting how many companies are assigned to each rating grade
# Sorting the result alphabetically by rating label for clean plotting
rating_counts = portfolio_full['Rating'].value_counts().sort_index()
print("Rating distribution:\n", rating_counts)

# Step 5: Plotting a bar chart showing how many companies fall into each credit rating category
# X-axis → Credit Ratings (e.g., AAA, AA, A, etc.)
# Y-axis → Number of companies with each rating
rating_counts.plot(kind='bar', title='Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Number of Companies')
plt.grid(True)
plt.show()

# Step 6: Map credit ratings to default probabilities (PDs) and thresholds
# Threshold is the z-score (norm.ppf) corresponding to each PD, used in one-factor model
min_pd = 1e-6  # Prevents issues with norm.ppf(0)
transitions['PD'] = transitions['D'].clip(lower=min_pd)
transitions['Threshold'] = transitions['PD'].apply(norm.ppf)

# Add PD and threshold to the master portfolio table
portfolio_full = portfolio_full.merge(
    transitions[['PD', 'Threshold']],
    left_on='Rating',
    right_index=True,
    how='left'
)
portfolio_full.rename(columns={'Threshold': 'threshold'}, inplace=True)

# Step 7: Calculate asset correlation (rho) between each stock and the healthcare index
# First, rename 'Index' column to 'IndexReturn' if needed for clarity
if 'Index' in index_returns.columns:
    index_returns.rename(columns={'Index': 'IndexReturn'}, inplace=True)

# Merge company returns and index returns on date
merged_returns = stock_returns.merge(index_returns[['Date', 'IndexReturn']], on='Date', how='inner') # new df created by merging data with common date field

# Compute correlation of each company’s returns with the index
# This gives asset correlation (rho) for the one-factor model
correlations = {}
for col in stock_returns.columns:
    if col != 'Date':
        correlations[col] = merged_returns[col].corr(merged_returns['IndexReturn']) # excluding date from field and calculating correlation in each column

# Convert correlation dictionary into a DataFrame and extract CompanyID
correlation_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['rho'])  # converting dictionary to dataframe
correlation_df.index.name = 'CompanyColumn'  # indexing with name companycolumn
correlation_df.reset_index(inplace=True)  # ( normal indexing like 0,1,2 added and companycolumn is not indexing column anymore)
correlation_df['CompanyID'] = correlation_df['CompanyColumn'].str.extract(r'(\d+)').astype(int)  # extracting id from companycolumn data

# Merge asset correlation (rho) values into the master portfolio table
portfolio_full = portfolio_full.merge(correlation_df[['CompanyID', 'rho']], on='CompanyID', how='left')
# Fill any missing correlations with a default value (e.g. 0.2) and clip between 0.01 and 0.9
portfolio_full['rho'] = portfolio_full['rho'].fillna(0.2).clip(lower=0.01, upper=0.9) # but no na field

# Step 8: Monte Carlo Simulation to estimate losses
# We simulate 10,000 market scenarios to estimate portfolio loss due to defaults
np.random.seed(42)
num_simulations = 10000
losses = []

for _ in range(num_simulations):
    F = np.random.normal()  # Common market shock
    total_loss = 0
    for _, row in portfolio_full.iterrows():
        epsilon = np.random.normal()  # Idiosyncratic shock
        rho = row['rho']
        asset_return = rho * F + np.sqrt(1 - rho**2) * epsilon
        # If the asset return falls below threshold, the company defaults
        if asset_return < row['threshold']:
            total_loss += row['Exposure']
    losses.append(total_loss)

# Step 9: Calculate expected loss, VaR and CVaR from simulations
losses = np.array(losses)
expected_loss = losses.mean()
VaR_99 = np.percentile(losses, 99)  # 99% Value at Risk
expected_shortfall_99 = losses[losses > VaR_99].mean()  # Conditional VaR

print(" Monte Carlo Simulation Results")
print(f"Expected Loss: ${expected_loss:,.2f}")
print(f"99% Value-at-Risk (VaR): ${VaR_99:,.2f}")
print(f"99% Expected Shortfall (CVaR): ${expected_shortfall_99:,.2f}")

# Step 10: Plot histogram of the simulated loss distribution
plt.figure(figsize=(10, 6))
plt.hist(losses, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(expected_loss, color='blue', linestyle='--', linewidth=2, label=f'Expected Loss = ${expected_loss:,.0f}')
plt.axvline(VaR_99, color='red', linestyle='--', linewidth=2, label=f'99% VaR = ${VaR_99:,.0f}')
plt.axvline(expected_shortfall_99, color='purple', linestyle='--', linewidth=2, label=f'99% CVaR = ${expected_shortfall_99:,.0f}')
plt.title('Simulated Loss Distribution of Credit Portfolio')
plt.xlabel('Portfolio Loss ($)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 9: Analytical approximation using Vasicek-style model
# This gives a closed-form estimate for Expected Loss and VaR assuming independent exposures
def analytical_var(portfolio_df, confidence_level=0.99):
    PDs = norm.cdf(portfolio_df['threshold'])  # Get PDs from thresholds
    exposures = portfolio_df['Exposure'].values
    expected_loss = np.sum(PDs * exposures)  # Weighted average of losses
    variance = np.sum((exposures ** 2) * PDs * (1 - PDs))  # Variance under binomial assumption
    std_dev = np.sqrt(variance)
    z_score = norm.ppf(confidence_level)
    VaR = expected_loss + z_score * std_dev  # Normal approx for VaR
    return expected_loss, VaR

expected_loss_analytic, VaR_analytic = analytical_var(portfolio_full)

print("\n Analytical Approximation")
print(f"Expected Loss: ${expected_loss_analytic:,.2f}")
print(f"Analytical 99% VaR: ${VaR_analytic:,.2f}")
