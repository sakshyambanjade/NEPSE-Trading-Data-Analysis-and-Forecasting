import pandas as pd
import numpy as np

def calculate_portfolio_metrics(df: pd.DataFrame, top_stocks: pd.Series):
    """
    Calculate portfolio risk-return metrics with NaN filtering.
    """
    # Filter for top stocks
    portfolio_df = df[df['Symbol'].isin(top_stocks)].copy()
    
    if 'Return' not in portfolio_df.columns:
        raise ValueError("'Return' column missing.")

    # Drop rows with NaN returns
    portfolio_df = portfolio_df.dropna(subset=['Return'])

    # Ensure enough data points exist per stock
    counts = portfolio_df.groupby('Symbol').size()
    symbols_to_use = counts[counts > 1].index.intersection(top_stocks)
    portfolio_df = portfolio_df[portfolio_df['Symbol'].isin(symbols_to_use)]

    # Recompute mean returns and covariance matrix
    mean_returns = portfolio_df.groupby('Symbol')['Return'].mean()
    returns_matrix = portfolio_df.pivot_table(index='Contract No', columns='Symbol', values='Return')
    cov_matrix = returns_matrix.cov()

    if cov_matrix.isnull().values.any():
        cov_matrix = cov_matrix.fillna(0)

    weights = np.array([1/len(symbols_to_use)] * len(symbols_to_use))
    expected_return = np.dot(weights, mean_returns.loc[symbols_to_use])
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.loc[symbols_to_use, symbols_to_use], weights))
    sharpe_ratio = expected_return / np.sqrt(portfolio_variance) if portfolio_variance > 0 else np.nan

    return {
        'Expected Return': expected_return,
        'Variance': portfolio_variance,
        'Sharpe Ratio': sharpe_ratio
    }
