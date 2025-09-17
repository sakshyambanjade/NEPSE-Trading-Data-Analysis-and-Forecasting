import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate algorithmic signals based on VWAP deviation and outlier detection.
    
    Adds columns:
    - 'Above_VWAP' (bool)
    - 'Below_VWAP' (bool)
    - 'Qty_Outlier' (bool)
    - 'Amount_Outlier' (bool)
    - 'Signal_Type' (categorical: 'Buy', 'Sell', 'Outlier', 'Neutral')
    """
    df = df.copy()
    df['Above_VWAP'] = df['Rate'] > df['VWAP']
    df['Below_VWAP'] = df['Rate'] < df['VWAP']
    
    qty_mean, qty_std = df['Quantity'].mean(), df['Quantity'].std()
    amt_mean, amt_std = df['Amount'].mean(), df['Amount'].std()
    
    df['Qty_Outlier'] = (df['Quantity'] > qty_mean + 3 * qty_std) | (df['Quantity'] < qty_mean - 3 * qty_std)
    df['Amount_Outlier'] = (df['Amount'] > amt_mean + 3 * amt_std) | (df['Amount'] < amt_mean - 3 * amt_std)
    
    def signal_type(row):
        if row['Qty_Outlier'] or row['Amount_Outlier']:
            return 'Outlier'
        elif row['Above_VWAP']:
            return 'Buy'
        elif row['Below_VWAP']:
            return 'Sell'
        else:
            return 'Neutral'
    
    df['Signal_Type'] = df.apply(signal_type, axis=1)
    
    return df

def top_participants(df: pd.DataFrame):
    """
    Get top buyers and sellers by total quantity and value.
    """
    top_buyers = df.groupby('Buyer')['Quantity'].sum().sort_values(ascending=False).head(10)
    top_sellers = df.groupby('Seller')['Quantity'].sum().sort_values(ascending=False).head(10)
    
    return top_buyers, top_sellers

if __name__ == '__main__':
    enhanced_path = "../data/processed/enhanced_dataset.csv"
    df = pd.read_csv(enhanced_path)
    
    df_signals = generate_signals(df)
    print("Sample signals:")
    print(df_signals[['Symbol', 'Contract No', 'Buyer', 'Seller', 'Rate', 'VWAP', 'Signal_Type']].head())
    
    top_buyers, top_sellers = top_participants(df)
    print("\nTop Buyers:")
    print(top_buyers)
    print("\nTop Sellers:")
    print(top_sellers)
