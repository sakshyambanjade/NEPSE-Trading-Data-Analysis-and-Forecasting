import pandas as pd

def load_cleaned_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the cleaned dataset CSV and perform final checks and type conversions.
    """
    df = pd.read_csv(filepath)
    
    df.columns = df.columns.str.strip()
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    df.dropna(subset=['Quantity', 'Rate', 'Amount'], inplace=True)
    df.drop_duplicates(inplace=True)
    
    return df

def load_enhanced_dataset(filepath: str) -> pd.DataFrame:
    """
    Load enhanced dataset with additional features.
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'Return' column as percentage change in Rate for each stock symbol.
    """
    df = df.sort_values(by=['Symbol', 'Contract No'])
    df['Return'] = df.groupby('Symbol')['Rate'].pct_change()
    return df

if __name__ == "__main__":
    cleaned_path = "../data/processed/cleaned_dataset.csv"
    df_clean = load_cleaned_dataset(cleaned_path)
    df_clean = add_returns(df_clean)
    print(df_clean.head())
