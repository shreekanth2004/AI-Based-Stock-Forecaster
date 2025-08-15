import os
from dotenv import load_dotenv
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
from flask import Flask, request, jsonify, send_from_directory
import google.generativeai as genai
import math
import re
import requests
import pandas as pd
from flask_cors import CORS

# This line loads the .env file
load_dotenv()

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Global Cache for Ticker List ---
ticker_list_cache = None

# --- Load Models ---
try:
    prediction_model = load_model('DM me for original model')
except Exception as e:
    print(f"Error loading prediction model: {e}")
    prediction_model = None

# --- Configure API Keys ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
fmp_api_key = os.getenv("FMP_API_KEY")

# --- Configure Gemini API ---
try:
    if not gemini_api_key:
        print("Warning: GEMINI_API_KEY not found in environment. 'Ask AI' will not work.")
        gemini_model = None
    else:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    gemini_model = None

# --- Helper Functions ---
def load_tickers_from_api():
    """ Fetches and caches the list of all stocks from the FMP API. """
    global ticker_list_cache
    if ticker_list_cache is not None:
        return ticker_list_cache

    if not fmp_api_key:
        print("Warning: FMP_API_KEY not found. Ticker search will not work.")
        return []

    try:
        url = f"https://financialmodelingprep.com/api/v3/stock/list?apikey={fmp_api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"FMP API returned {len(data)} total results.")
        
        filtered_data = [
            {'symbol': item['symbol'], 'name': item.get('name', '')}
            for item in data
            if item.get('symbol') and item.get('name')
        ]
        
        ticker_list_cache = filtered_data
        print(f"Successfully loaded and cached {len(ticker_list_cache)} tickers from FMP API.")
        return ticker_list_cache
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tickers from FMP API: {e}")
        return []

def clean_nan(obj):
    if isinstance(obj, dict): return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean_nan(i) for i in obj]
    if isinstance(obj, float) and math.isnan(obj): return None
    return obj

# --- Static File Serving (FOR FRONTEND) ---

@app.route('/')
def serve_index():
    """Serves the index.html file."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serves other static files like script.js, style.css, etc."""
    return send_from_directory('.', path)

# --- API Endpoints ---
@app.route('/search-tickers', methods=['GET'])
def search_tickers():
    query = request.args.get('query', '').upper()
    tickers = load_tickers_from_api()
    if not query or not tickers:
        return jsonify([])
        
    results = [
        item for item in tickers
        if item.get('name') and (query in item['symbol'].upper() or query in item['name'].upper())
    ]
    
    results.sort(key=lambda x: (not x['symbol'].upper().startswith(query), not x['name'].upper().startswith(query), x['symbol']))
    return jsonify(results[:7])

@app.route('/market-overview', methods=['GET'])
def get_market_overview():
    indices = {'S&P 500': '^GSPC', 'Dow Jones': '^DJI', 'NASDAQ': '^IXIC'}
    try:
        index_data = yf.download(list(indices.values()), period="2d", progress=False)
        index_results = []
        for name, symbol in indices.items():
            hist = index_data['Close'][symbol]
            if len(hist) > 1:
                prev_close, latest_close = hist.iloc[-2], hist.iloc[-1]
                change = latest_close - prev_close
                percent_change = (change / prev_close) * 100
                index_results.append({'name': name, 'price': round(latest_close, 2), 'change': round(change, 2), 'percent_change': round(percent_change, 2)})
        news_ticker = yf.Ticker('SPY')
        market_news = news_ticker.news[:7] if news_ticker.news else []
        return jsonify({'indices': index_results, 'news': market_news})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask-gemini', methods=['POST'])
def ask_gemini():
    if not gemini_model:
        return jsonify({'error': 'Gemini AI is not configured.'}), 503
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt is missing.'}), 400
    prompt, history = data['prompt'], data.get('history', [])
    try:
        chat = gemini_model.start_chat(history=history)
        response = chat.send_message(prompt)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': f'An error occurred with the Gemini API: {str(e)}'}), 500

@app.route('/market-movers', methods=['GET'])
def get_market_movers():
    popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ']
    try:
        data = yf.download(popular_stocks, period="2d", progress=False)
        if data.empty or len(data['Close']) < 2: return jsonify({'error': 'Not enough data.'}), 500
        close_prices = data['Close']
        prev_day_close, latest_close = close_prices.iloc[-2], close_prices.iloc[-1]
        percent_change = ((latest_close - prev_day_close) / prev_day_close) * 100
        movers = []
        for symbol in popular_stocks:
            change, price = percent_change.get(symbol), latest_close.get(symbol)
            if pd.notna(change) and pd.notna(price):
                movers.append({'symbol': symbol, 'price': round(price, 2), 'change': round(change, 2)})
        movers.sort(key=lambda x: abs(x['change']), reverse=True)
        return jsonify(movers[:6])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['GET'])
def predict():
    if not prediction_model: return jsonify({'error': 'Prediction model not available.'}), 503
    stock_symbol = request.args.get('stock', 'GOOG').upper()
    try:
        data = yf.download(stock_symbol, start=datetime.date.today() - datetime.timedelta(days=365), end=datetime.date.today(), progress=False)
        if data.empty or len(data) < 100: return jsonify({'error': f"Not enough data for '{stock_symbol}'."}), 404
        close_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        current_sequence = scaled_data[-100:].reshape(1, 100, 1)
        future_predictions_scaled = []
        for _ in range(10):
            next_prediction_scaled = prediction_model.predict(current_sequence, verbose=0)
            future_predictions_scaled.append(next_prediction_scaled[0, 0])
            new_sequence = np.append(current_sequence[0, 1:, :], next_prediction_scaled, axis=0)
            current_sequence = new_sequence.reshape(1, 100, 1)
        future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
        response = {'stock_symbol': stock_symbol, 'last_known_price': {'date': data.index[-1].strftime('%Y-%m-%d'), 'price': float(data['Close'].iloc[-1])}, 'predictions': [{'date': (datetime.date.today() + datetime.timedelta(days=i)).strftime('%Y-%m-%d'), 'price': float(price)} for i, price in enumerate(future_predictions.flatten(), 1)]}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/details', methods=['GET'])
def get_details():
    stock_symbol = request.args.get('stock', 'GOOG').upper()
    try:
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info
        hist_data = ticker.history(period="2y")
        if hist_data.empty: return jsonify({'error': f"No historical data for '{stock_symbol}'."}), 404
        
        # UPDATED: Smarter name retrieval
        company_name = info.get('longName') or info.get('shortName') or stock_symbol
        
        hist_data['SMA50'] = hist_data['Close'].rolling(window=50).mean()
        hist_data['SMA100'] = hist_data['Close'].rolling(window=100).mean()
        hist_data['SMA200'] = hist_data['Close'].rolling(window=200).mean()
        def series_to_list(series): return series.where(pd.notnull(series), None).tolist()

        response = {
            'symbol': stock_symbol,
            'longName': company_name,
            'marketCap': info.get('marketCap'), 
            'longBusinessSummary': info.get('longBusinessSummary'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'), 
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
            'trailingPE': info.get('trailingPE'), 
            'dividendYield': info.get('dividendYield'),
            'news': ticker.news[:5] if ticker.news else [],
            'historicalData': {
                'dates': hist_data.index.strftime('%Y-%m-%d').tolist(),
                'prices': series_to_list(hist_data['Close']),
                'sma50': series_to_list(hist_data['SMA50']),
                'sma100': series_to_list(hist_data['SMA100']),
                'sma200': series_to_list(hist_data['SMA200'])
            }
        }
        return jsonify(clean_nan(response))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/financials', methods=['GET'])
def get_financials():
    stock_symbol = request.args.get('stock', 'GOOG').upper()
    try:
        ticker = yf.Ticker(stock_symbol)
        financials = ticker.quarterly_financials
        if financials.empty: return jsonify({'error': f"No financial data for '{stock_symbol}'."}), 404
        financials = financials.T.iloc[:8].sort_index()
        def series_to_list_financials(series): return series.where(pd.notnull(series), None).tolist()
        response = {'symbol': stock_symbol, 'dates': financials.index.strftime('%Y-%m-%d').tolist(), 'revenue': series_to_list_financials(financials.get('Total Revenue', pd.Series(0))), 'netIncome': series_to_list_financials(financials.get('Net Income', pd.Series(0)))}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-prices', methods=['GET'])
def get_batch_prices():
    symbols = request.args.get('symbols', '').upper().split(',')
    symbols = [s for s in symbols if s]
    if not symbols: return jsonify({}), 200
    try:
        data = yf.download(symbols, period="1d", progress=False)
        prices = {}
        if data.empty:
            for symbol in symbols: prices[symbol] = None
            return jsonify(prices)
        if len(symbols) > 1:
            latest_closes = data['Close'].iloc[-1]
            for symbol in symbols:
                price = latest_closes.get(symbol)
                prices[symbol] = float(price) if pd.notna(price) else None
        else:
            symbol = symbols[0]
            price = data['Close'].iloc[-1]
            prices[symbol] = float(price) if pd.notna(price) else None
        return jsonify(prices)
    except Exception as e:
        prices = {symbol: None for symbol in symbols}
        print(f"Error in /batch-prices: {e}")
        return jsonify(prices), 500

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    if not gemini_model: return jsonify({'error': 'Gemini AI is not configured.'}), 503
    data = request.get_json()
    if not data or 'news' not in data: return jsonify({'error': 'News data is missing.'}), 400
    news_items = data['news']
    titles_only = [item.get('title', '') for item in news_items]
    prompt = ("Analyze the sentiment of the following financial news headlines. " "Classify each one strictly as 'Positive', 'Negative', or 'Neutral'. " "Do not add any other commentary. Return the results as a simple comma-separated list. " "For example: Positive,Negative,Neutral\n\n" + "\n".join(f"{i+1}. {title}" for i, title in enumerate(titles_only)))
    try:
        response = gemini_model.generate_content(prompt)
        sentiments = re.sub(r'[`\*\s]', '', response.text).split(',')
        if len(sentiments) == len(news_items):
            return jsonify({'sentiments': sentiments})
        else:
            return jsonify({'sentiments': ['Neutral'] * len(news_items)})
    except Exception as e:
        return jsonify({'error': f'An error occurred with the Gemini API: {str(e)}'}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)