# Real-Time Financial Dashboard

A comprehensive financial dashboard built with Python and Streamlit that provides real-time market data, technical analysis, and portfolio tracking.

## Features

- Real-time stock market data visualization
- Cryptocurrency price tracking and analysis
- Technical indicators and chart patterns
- Portfolio management and performance tracking
- Market sentiment analysis
- Customizable watchlists
- Historical data analysis
- Export functionality for reports

## Tech Stack

- Python 3.11+
- Streamlit for web interface
- yfinance for stock market data
- CCXT for cryptocurrency data
- Plotly for interactive charts
- pandas-ta for technical analysis
- scikit-learn for predictive analytics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-dashboard.git
cd financial-dashboard
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your API keys:
```bash
ALPHA_VANTAGE_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

## Usage

1. Start the dashboard:
```bash
streamlit run src/app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
financial-dashboard/
├── data/               # Data storage and cache
├── src/               # Source code
│   ├── components/    # UI components
│   ├── services/     # Data services
│   └── utils/        # Utility functions
├── static/           # Static files (images, css)
├── templates/        # HTML templates
├── tests/           # Test files
├── .env             # Environment variables
├── requirements.txt # Project dependencies
└── README.md       # Project documentation
```

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Yahoo Finance](https://finance.yahoo.com/) for financial data
- [CCXT](https://github.com/ccxt/ccxt) for cryptocurrency data
- [Streamlit](https://streamlit.io/) for the web framework

Copyright (c) 2025 Abhishek Banerjee
