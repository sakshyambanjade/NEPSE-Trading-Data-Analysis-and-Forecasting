import pandas as pd
import numpy as np

def stock_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics per stock symbol, including:
      - Total traded volume
      - Total traded value
      - Average trade size
      - Trade count
      - Max and Min Rate
      - VWAP (volume weighted average price), safely handling zero quantities
      - Volatility (std of trade-to-trade pct change in Rate)
    """
    grouped = df.groupby('Symbol')

    # Suppress FutureWarning by excluding grouping columns explicitly (requires pandas 2.3+)
    stats = pd.DataFrame({
        'TotalVolume': grouped['Quantity'].sum(),
        'TotalValue': grouped['Amount'].sum(),
        'AverageTradeSize': grouped['Quantity'].mean(),
        'TradeCount': grouped.size(),
        'MaxRate': grouped['Rate'].max(),
        'MinRate': grouped['Rate'].min(),
        'VWAP': grouped.apply(
            lambda x: np.average(x['Rate'], weights=x['Quantity'])
                      if x['Quantity'].sum() > 0 else float('nan'),
            include_groups=False
        ),
        'Volatility': grouped['Rate'].apply(lambda x: x.pct_change().std(), include_groups=False)
    })

    stats.sort_values(by='TotalVolume', ascending=False, inplace=True)
    return stats

def top_n_stocks_by_metric(stats: pd.DataFrame, metric: str, n=10) -> pd.DataFrame:
    """
    Return top N stocks sorted by specified metric.
    """
    return stats.sort_values(by=metric, ascending=False).head(n)

def buyer_seller_analysis(buy_df: pd.DataFrame, sell_df: pd.DataFrame):
    """
    Perform analysis of buyer and seller participation:
      - Aggregate total quantities, amounts traded
      - Count number of trades
      - Calculate average trade size per participant

    Columns are stripped of whitespace and verified to exist.

    Returns two DataFrames: buyer stats and seller stats.
    """
    buy_df.columns = buy_df.columns.str.strip()
    sell_df.columns = sell_df.columns.str.strip()

    if 'Buyer' not in buy_df.columns:
        raise ValueError(f"'Buyer' column not found. Available columns: {buy_df.columns.tolist()}")
    if 'Seller' not in sell_df.columns:
        raise ValueError(f"'Seller' column not found. Available columns: {sell_df.columns.tolist()}")

    buy_stats = buy_df.groupby('Buyer').agg(
        TotalQuantity=('Quantity', 'sum'),
        TotalAmount=('Amount', 'sum'),
        TradeCount=('Quantity', 'size'),
        AverageTradeSize=('Quantity', 'mean')
    ).sort_values(by='TotalQuantity', ascending=False)

    sell_stats = sell_df.groupby('Seller').agg(
        TotalQuantity=('Quantity', 'sum'),
        TotalAmount=('Amount', 'sum'),
        TradeCount=('Quantity', 'size'),
        AverageTradeSize=('Quantity', 'mean')
    ).sort_values(by='TotalQuantity', ascending=False)

    return buy_stats, sell_stats

if __name__ == '__main__':
    cleaned_path = "../data/processed/cleaned_dataset.csv"
    enhanced_path = "../data/processed/enhanced_dataset.csv"
    buy_pressure_path = "../data/processed/buy_pressure.csv"
    sell_pressure_path = "../data/processed/sell_pressure.csv"

    # Load datasets
    df_clean = pd.read_csv(cleaned_path)
    df_enhanced = pd.read_csv(enhanced_path)
    df_buy = pd.read_csv(buy_pressure_path)
    df_sell = pd.read_csv(sell_pressure_path)

    # Compute stock-level summary stats
    stats = stock_summary_stats(df_enhanced)
    print("Stock Summary Stats:")
    print(stats.head())

    # Display top 10 stocks by volume
    top_volume = top_n_stocks_by_metric(stats, 'TotalVolume')
    print("\nTop 10 Stocks by Volume:")
    print(top_volume)

    # Buyer and Seller participation analysis with column checks and stripping
    buy_stats, sell_stats = buyer_seller_analysis(df_buy, df_sell)
    print("\nTop Buyers:")
    print(buy_stats.head())
    print("\nTop Sellers:")
    print(sell_stats.head())
