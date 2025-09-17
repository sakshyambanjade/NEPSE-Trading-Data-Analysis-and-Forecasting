from src import data_processing, analysis, portfolio, signals, ml_analysis, visualization
import pandas as pd
from pathlib import Path

def main():
    data_dir = Path('./data/processed')
    outputs_fig_dir = Path('./outputs/figures')
    outputs_tbl_dir = Path('./outputs/tables')
    
    outputs_fig_dir.mkdir(parents=True, exist_ok=True)
    outputs_tbl_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_path = data_dir / 'cleaned_dataset.csv'
    enhanced_path = data_dir / 'enhanced_dataset.csv'
    top_volume_path = data_dir / 'top_volume.csv'
    
    print("Loading datasets...")
    df_clean = data_processing.load_cleaned_dataset(str(cleaned_path))
    df_enhanced = data_processing.load_enhanced_dataset(str(enhanced_path))
    top_volume = pd.read_csv(top_volume_path)
    
    # Add returns to enhanced dataset
    df_enhanced = data_processing.add_returns(df_enhanced)
    
    print("Running stock-level analysis...")
    stats = analysis.stock_summary_stats(df_enhanced)
    stats.to_csv(outputs_tbl_dir / 'stock_summary_stats.csv')
    
    # Use df_clean for buyer/seller participation (contains Buyer/Seller columns)
    print("Running buyer/seller analysis...")
    buy_stats, sell_stats = analysis.buyer_seller_analysis(df_clean, df_clean)
    buy_stats.to_csv(outputs_tbl_dir / 'buy_stats.csv')
    sell_stats.to_csv(outputs_tbl_dir / 'sell_stats.csv')
    
    print("Calculating portfolio metrics...")
    top_stocks = top_volume['Symbol'].head(5)
    portfolio_metrics = portfolio.calculate_portfolio_metrics(df_enhanced, top_stocks)
    print("Portfolio Metrics:", portfolio_metrics)
    
    print("Generating algorithmic signals...")
    df_signals = signals.generate_signals(df_enhanced)
    df_signals.to_csv(outputs_tbl_dir / 'trade_signals.csv')
    
    print("Running ML analysis...")
    df_clustered, _ = ml_analysis.cluster_trades(df_signals)
    df_classified, _ = ml_analysis.classify_trade_size(df_clustered)
    df_classified.to_csv(outputs_tbl_dir / 'ml_analysis_results.csv')
    
    print("Generating visualizations...")
    visualization.plot_top_stocks_by_volume(stats, str(outputs_fig_dir))
    visualization.plot_trade_size_distribution(df_enhanced, str(outputs_fig_dir))
    visualization.plot_price_distribution(df_enhanced, str(outputs_fig_dir))
    
    top_buyers = df_classified.groupby('Buyer')['Quantity'].sum().sort_values(ascending=False).head(10)
    top_sellers = df_classified.groupby('Seller')['Quantity'].sum().sort_values(ascending=False).head(10)
    visualization.plot_top_participants_bar(top_buyers, "Top Buyers by Quantity", str(outputs_fig_dir))
    visualization.plot_top_participants_bar(top_sellers, "Top Sellers by Quantity", str(outputs_fig_dir))
    
    visualization.plot_clusters(df_classified, str(outputs_fig_dir))
    visualization.plot_trade_signals(df_signals, str(outputs_fig_dir))
    
    print("All tasks completed and outputs saved.")

if __name__ == '__main__':
    main()
