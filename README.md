# NEPSE Trading Data Analysis and Forecasting

#Project Overview

This project is an end-to-end data analysis and forecasting pipeline built for the Nepal Stock Exchange (NEPSE) trading floorsheet data. It leverages Python and several libraries to clean, analyze, and visualize stock market data and apply machine learning for trading signal generation.

---

#Features

- Data Cleaning and Processing: Handling raw trade-level NEPSE data to remove duplicates, fix datatypes, and calculate returns.
- Stock and Participant Analytics: Aggregate trade metrics per stock, and analyze buyer and seller participation.
- Portfolio Risk Metrics:Compute expected returns, portfolio variance, and Sharpe ratios.
- Algorithmic Signals: Generate buy/sell/outlier signals based on trading patterns.
- Machine Learning: Clustering and classification of trades with evaluation reports.
- Visualizations: Comprehensive set of charts including top stocks by volume, trade size distribution, price distribution, ML clusters, and trading signals.
- Modular design: Organized into dedicated modules for processing, analysis, visualization, and ML.

---

#Getting Started

#Prerequisites

- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, fastapi (for backend API plans), and others listed in `requirements.txt`

#Installation

```
git clone https://github.com/yourusername/project.git
cd nepse-trading-data-analysis
pip install -r requirements.txt
```

#Running the Pipeline

Run the main driver script:

```
python main.py
```

This loads data, runs analysis, creates trading signals, executes ML models, and outputs visualizations and reports to `/outputs`.

---

#Project Structure

```
NEPSE_Trading_Data_Analysis_and_Forecasting/
│
├── data/                     # Raw and processed input datasets
├── outputs/                  # Generated plots and CSV summaries
├── src/                      # Python source code modules
│   ├── data_processing.py    # Data loading, cleaning, returns calculation
│   ├── analysis.py           # Stock and participant analytics
│   ├── portfolio.py          # Portfolio risk-return calculations
│   ├── signals.py            # Signal generation algorithms
│   ├── ml_analysis.py        # Clustering and classification models
│   ├── visualization.py      # Visualization functions
│
├── main.py                   # Main pipeline execution script
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and instructions
```

---

#Results

- Successfully cleaned and enhanced NEPSE floorsheet data.
- Computed comprehensive stock-level statistics and buyer/seller participation summaries.
- Calculated portfolio metrics with risk-adjusted returns.
- Developed ML models yielding clustering insights and classification results (~63% accuracy).
- Generated multiple insightful visualizations saved in `outputs/figures`.
- Modular, reusable code for each pipeline stage.

---

#Future Work

- Expose pipeline as backend REST API using FastAPI with MongoDB storage.
- Build interactive frontend dashboard with React for dynamic data visualization.
- Incorporate real-time data streaming and live portfolio updates.
- Extend ML with sequential models (RNN, LSTM) for improved predictive power.
- Automate report generation and backtesting framework integration.

---

## Acknowledgments

The data has been only used for Education only.

---

```
