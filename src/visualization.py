import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

def plot_top_stocks_by_volume(stats, output_dir, top_n=10):
    """
    Plot the top N stocks by trading volume from the stats DataFrame.

    Args:
        stats (pd.DataFrame): DataFrame indexed by stock Symbol with 'TotalVolume' column.
        output_dir (str): Directory where to save the plot.
        top_n (int): Number of top stocks to display.
    """
    top_volume = stats['TotalVolume'].nlargest(top_n)

    plt.figure(figsize=(10, 6))
    top_volume.plot(kind='bar', color='skyblue')
    plt.title(f"Top {top_n} Stocks by Trading Volume")
    plt.xlabel("Stock Symbol")
    plt.ylabel("Total Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"top_{top_n}_stocks_by_volume.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Saved plot: {file_path}")

def plot_trade_size_distribution(df, output_dir):
    """
    Plot the distribution of trade sizes from the 'Quantity' column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the trade data.
        output_dir (str): Path where to save the figure.
    """
    plt.figure(figsize=(10,6))
    sns.histplot(df['Quantity'].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.title("Trade Size Distribution")
    plt.xlabel("Quantity")
    plt.ylabel("Frequency")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "trade_size_distribution.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

def plot_price_distribution(df, output_dir):
    """
    Plot histogram of trade prices grouped by Symbol.
    """
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df, x='Rate', hue='Symbol', multiple='stack', bins=30)
    plt.title("Price Distribution per Stock")
    plt.xlabel("Trade Rate")
    plt.ylabel("Count")
    plt.tight_layout()
    filepath = os.path.join(output_dir, "price_distribution_per_stock.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

def plot_top_participants_bar(participants_series, title, output_dir):
    """
    Plot horizontal bar chart of top participants (buyers/sellers).
    """
    plt.figure(figsize=(14, 6))
    sns.barplot(x=participants_series.values, y=participants_series.index, color='blue')
    plt.title(title)
    plt.xlabel("Total Quantity")
    plt.tight_layout()
    filename = title.lower().replace(" ", "_") + ".png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")

def plot_clusters(df, output_dir, feature_cols=['Quantity', 'Rate', 'Amount']):
    """
    Scatter plot of clustered data points.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=feature_cols[0], y=feature_cols[1],
        hue='Cluster', palette='Set2',
        data=df, legend='full', alpha=0.7
    )
    plt.title("Trade Clusters")
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.tight_layout()
    filepath = os.path.join(output_dir, "trade_clusters.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved cluster plot: {filepath}")

def plot_trade_signals(df, output_dir):
    """
    Scatter plot showing buy/sell/outlier trading signals.
    """
    plt.figure(figsize=(12, 8))
    colors = {'Buy': 'green', 'Sell': 'red', 'Outlier': 'orange', 'Neutral': 'blue'}
    for signal_type, color in colors.items():
        subset = df[df['Signal_Type'] == signal_type]
        plt.scatter(subset['Rate'], subset['Quantity'], c=color, label=signal_type, alpha=0.6)
    plt.xlabel('Rate')
    plt.ylabel('Quantity')
    plt.title('Trading Signals: Buy / Sell / Outlier')
    plt.legend()
    plt.tight_layout()
    filepath = os.path.join(output_dir, "trading_signals_scatter.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved trading signals plot: {filepath}")
