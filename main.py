import os
import time
import logging
import smtplib
from typing import Dict, List, Optional, Any
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from dotenv import load_dotenv

# Load environment variables (local dev only)
load_dotenv()

# Configure Logging
# We use INFO level to keep track of the pipeline's heartbeat in GitHub Actions logs.
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class PSXDataEngine:
    """
    Handles robust data extraction from Yahoo Finance with retry logic and 
    multi-asset merging.
    """
    def __init__(self, tickers: List[str], macro_tickers: List[str]):
        self.tickers = tickers
        self.macro_tickers = macro_tickers

    def fetch_data(self, period: str = "2y") -> Dict[str, pd.DataFrame]:
        """
        Orchestrates the data fetching pipeline:
        1. Fetch Macro-Economic Data (Oil, USD/PKR).
        2. Fetch Equities Data.
        3. Merge datasets using time-series logic (Forward Fill).
        """
        # 1. Macro Data Ingestion
        # We fetch macro data first because it acts as the "global state" context 
        # for our local equity predictions.
        macro_data = self._fetch_macro_data(period)

        # 2. Equity Data Ingestion & Merging
        final_data = {}
        for ticker in self.tickers:
            try:
                df = self._fetch_ticker_with_retry(ticker, period)
                if df.empty:
                    logger.warning(f"Skipping {ticker}: No data returned.")
                    continue
                
                # Flatten MultiIndex columns if YF returns them
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # 3. Time-Series Merging
               
                # Reason: Global markets (Oil) and Local markets (PSX) have different holidays.
                # We care about the state of Oil ON the day the PSX is trading.
                for name, series in macro_data.items():
                    df = df.join(series, how='left')
                    
                    # Forward Fill is essential here. If Oil didn't trade today (Sunday in US),
                    # but PSX is open (Monday in Pakistan), we assume Oil price stays same as Friday close.
                    df[name] = df[name].ffill().bfill()
                
                final_data[ticker] = df
                
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                
        return final_data

    def _fetch_macro_data(self, period: str) -> Dict[str, pd.Series]:
        data = {}
        for ticker in self.macro_tickers:
            df = self._fetch_ticker_with_retry(ticker, period)
            if not df.empty:
               
                clean_name = ticker.replace("=F", "").replace("=X", "")
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                data[clean_name] = df['Close'].rename(clean_name)
        return data

    def _fetch_ticker_with_retry(self, ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
        """
        Wraps yfinance download in a retry loop to handle transient network flakiness
        common in CI/CD environments.
        """
        for i in range(retries):
            try:
                logger.info(f"Fetching {ticker} (Attempt {i+1}/{retries})...")
                df = yf.download(ticker, period=period, progress=False)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Download failed for {ticker}: {e}")
                time.sleep(2) # Backoff
        
        logger.error(f"Given up on {ticker} after {retries} attempts.")
        return pd.DataFrame()


class FeatureEngineer:
    """
    Transforms raw OHLCV and Macro data into quantitative features.
    Implements V2.0 logic: OBV, ATR, and Macro Correlators.
    """
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        
      
        df = df.copy()

        # FIXME: Currently assuming 'Volume' exists. Some indices (like ^KSE) might not 
        # report volume reliably via YF. Should add a check or fallback.
        vol_col = 'Volume' if 'Volume' in df.columns else None

        # --- Alpha Factors (Technical) ---
        
        # 1. On-Balance Volume (Smart Money Flow)
        # Detects if volume is flowing into (Accumulation) or out of (Distribution) the stock.
        if vol_col:
            df['OBV'] = ta.obv(df['Close'], df[vol_col])
        else:
            df['OBV'] = 0 # Neutral filler

        # 2. Average True Range (Volatility Regime)
        # Used to normalize returns or detect "Fear" in the market.
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        # 3. Momentum Oscillators
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # MACD (Trend Following)
        macd = ta.macd(df['Close'])
        if macd is not None:
            df = pd.concat([df, macd], axis=1)

        # Bollinger Bands (Mean Reversion)
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None:
             df = pd.concat([df, bb], axis=1)

        # --- Macro Factors (Inter-market Analysis) ---
        # Did the USD strengthen against PKR? Did Oil crash?
        if 'CL' in df.columns:
            df['Oil_Ret_1D'] = df['CL'].pct_change(1)
        if 'PKR' in df.columns:
            df['USD_Ret_1D'] = df['PKR'].pct_change(1)

        # --- Seasonality ---
        # Markets often have "Monday Effects" or "Turn of Month" flows.
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month 

        # --- Target Generation ---
        # Prediction Target: Did the price close HIGHER tomorrow?
        # Shift(-1) aligns Tomorrow's close with Today's features.
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        return df


class OracleModel:
    """
    Encapsulates the XGBoost training pipeline, validation, and inference logic.
    """
    def __init__(self):
        # TODO: Move hyperparameters to a config file (yaml/json) for better MLOps.
        self.model = XGBClassifier(
            n_estimators=150, 
            learning_rate=0.03, 
            max_depth=4, 
            random_state=42, 
            eval_metric='logloss',
            n_jobs=-1 
        )

    def train_predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs Walk-Forward Validation (TimeSeriesSplit) to estimate model confidence,
        then retrains on full history for the final prediction.
        """
        # 1. Feature Selection
        # Drop raw pricing columns, keeping only stationary/normalized features (Returns, RSI, etc.)
        exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'CL', 'PKR']
        features = [col for col in df.columns if col not in exclude_cols]
        # Ensure we only use numeric columns
        features = df[features].select_dtypes(include=[np.number]).columns.tolist()

        # 2. Train / Inference Split
        # The last row is "Today" (Live). It has NO Target (since tomorrow hasn't happened).
        # We must isolate it to prevent it from confusing the model during training.
        live_row = df.iloc[[-1]][features].copy()
        
        # Train on everything BEFORE today.
        # We drop NaNs here which removes the early rows where indicators (RSI) were calculating.
        train_df = df.iloc[:-1].dropna()
        
        if len(train_df) < 50:
            raise ValueError(f"Insufficient training data: {len(train_df)} rows")

        X = train_df[features]
        y = train_df['Target']

        # 3. Walk-Forward Validation
        # We strictly verify that the model works on UNSEEN future data using TimeSeriesSplit.
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            self.model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = self.model.predict(X.iloc[test_idx])
            scores.append(accuracy_score(y.iloc[test_idx], preds))

        avg_confidence = np.mean(scores) * 100
        
        # 4. Production Training
        self.model.fit(X, y)
        
        # 5. Inference
        prediction = self.model.predict(live_row)[0]
        prob = self.model.predict_proba(live_row)[0][prediction]

        # 6. Explainability (Feature Importance)
        # What drove this decision?
        importances = dict(zip(features, self.model.feature_importances_))
        top_factor = max(importances.items(), key=lambda x: x[1])

        return {
            "prediction": "GREEN" if prediction == 1 else "RED",
            "confidence": avg_confidence,
            "prob": prob * 100,
            "latest_features": live_row.iloc[0].to_dict(),
            "top_factor": f"{top_factor[0]} ({top_factor[1]:.0%})"
        }


class EmailReporter:
    """
    Generates professional HTML equity research reports and dispatchs via SMTP.
    """
    def __init__(self, user: str, password: str, recipients: List[str]):
        self.user = user
        self.password = password
        self.recipients = recipients

    def generate_html(self, results: Dict[str, Any], date: str) -> str:
        # Standard FinTech CSS Palette
        colors = {
            "green": "#27ae60",
            "red": "#c0392b",
            "bg_green": "rgba(39, 174, 96, 0.1)",
            "bg_red": "rgba(192, 57, 43, 0.1)",
            "text": "#2c3e50"
        }
        
        html_cards = ""
        for ticker, res in results.items():
            is_bullish = res['prediction'] == "GREEN"
            color = colors["green"] if is_bullish else colors["red"]
            bg = colors["bg_green"] if is_bullish else colors["bg_red"]
            
            html_cards += f"""
                <div style="border-left: 5px solid {color}; background-color: {bg}; padding: 15px; margin-bottom: 20px; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(0,0,0,0.05); padding-bottom: 10px; margin-bottom: 10px;">
                        <h3 style="margin: 0; color: {colors['text']};">{ticker}</h3>
                        <span style="font-weight: bold; color: {color}; font-size: 1.2em;">{res['prediction']}</span>
                    </div>
                    
                    <div style="font-size: 14px; color: #34495e; line-height: 1.6;">
                        <strong>Signal:</strong> {res['prob']:.1f}% Probability<br>
                        <strong>Confidence:</strong> {res['confidence']:.1f}% (Backtest Accuracy)<br>
                        <strong>Analyst Note:</strong> Signal driven by {res['top_factor']}
                    </div>

                    <div style="margin-top: 12px; font-size: 11px; color: #7f8c8d; display: flex; gap: 15px;">
                        <span>🛢 Oil: {res.get('latest_features', {}).get('Oil_Ret_1D', 0)*100:+.2f}%</span>
                        <span>💵 USD: {res.get('latest_features', {}).get('USD_Ret_1D', 0)*100:+.2f}%</span>
                        <span>📈 RSI: {res.get('latest_features', {}).get('RSI', 0):.1f}</span>
                    </div>
                </div>
            """
            
        return f"""
        <html>
            <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #f8f9fa; padding: 20px;">
                <div style="max-width: 600px; margin: auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 15px rgba(0,0,0,0.05);">
                    <h2 style="color: {colors['text']}; text-align: center; margin-bottom: 10px;">PSX-Oracle Reports</h2>
                    <p style="text-align: center; color: #95a5a6; font-size: 12px; margin-top: 0;">Quant Data for {date}</p>
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 30px 0;">
                    
                    {html_cards}
                    
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 30px 0;">
                    <p style="font-size: 10px; text-align: center; color: #bdc3c7;">
                        Generated by PSX-Oracle V2.1 | Not Financial Advice
                    </p>
                </div>
            </body>
        </html>
        """

    def send_email(self, subject: str, body: str):
        if not self.user or not self.password:
            logger.warning("SMTP Config missing. Printing report to stdout.")
            
            return

        msg = MIMEMultipart()
        msg['From'] = self.user
        msg['To'] = ", ".join(self.recipients)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(self.user, self.password)
            server.sendmail(self.user, self.recipients, msg.as_string())
            logger.info("Report dispatched successfully.")


def main():
    # --- Configuration ---
    # TODO: Externalize to separate config file
    STOCKS = ["OGDC.KA", "LUCK.KA", "ENGRO.KA"] 
    MACROS = ["CL=F", "PKR=X"] # Crude Oil, USD/PKR
    
    # Secrets
    EMAIL_USER = os.environ.get("EMAIL_USER")
    EMAIL_PASS = os.environ.get("EMAIL_PASS")
    RECIPIENTS = [EMAIL_USER] if EMAIL_USER else []

    # --- Pipeline Execution ---
    logger.info("Starting PSX-Oracle V2.1 Pipeline...")
    
    # 1. Data Layer
    engine = PSXDataEngine(STOCKS, MACROS)
    data_map = engine.fetch_data()
    
    # 2. Features & Modeling
    feature_eng = FeatureEngineer()
    oracle = OracleModel()
    results = {}
    
    for ticker, df in data_map.items():
        try:
            logger.info(f"Analyzing {ticker}...")
            
            # Feature Extraction
            processed_df = feature_eng.add_features(df)
            if processed_df is None or processed_df.empty:
                logger.warning(f"Insufficient data for {ticker}")
                continue
                
            # Inference
            prediction = oracle.train_predict(processed_df)
            results[ticker] = prediction
            
        except Exception as e:
            logger.error(f"Pipeline failed for {ticker}: {e}", exc_info=True)

    # 3. Reporting Layer
    if results:
        reporter = EmailReporter(EMAIL_USER, EMAIL_PASS, RECIPIENTS)
        today = datetime.now().strftime("%Y-%m-%d")
        report_html = reporter.generate_html(results, today)
        
        reporter.send_email(f"PSX-Oracle Market Brief: {today}", report_html)
        logger.info("Pipeline completed successfully.")
    else:
        logger.error("Pipeline finished with NO results.")

if __name__ == "__main__":
    main()
