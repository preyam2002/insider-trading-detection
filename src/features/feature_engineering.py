import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, feature_windows: List[int] = [7, 14, 30, 60]):
        self.feature_windows = feature_windows
        
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['ticker', 'Date'])
        
        for window in self.feature_windows:
            df[f'return_{window}d'] = df.groupby('ticker')['Close'].pct_change(window)
            df[f'volatility_{window}d'] = df.groupby('ticker')['Close'].pct_change().rolling(window).std()
            
        return df
    
    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.feature_windows:
            df[f'sma_{window}d'] = df.groupby('ticker')['Close'].transform(
                lambda x: x.rolling(window).mean()
            )
            df[f'ema_{window}d'] = df.groupby('ticker')['Close'].transform(
                lambda x: x.ewm(span=window).mean()
            )
            
        df['sma_50_200_ratio'] = df['sma_50d'] / df['sma_200d']
        
        return df
    
    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        for window in self.feature_windows:
            df[f'volume_sma_{window}d'] = df.groupby('ticker')['Volume'].transform(
                lambda x: x.rolling(window).mean()
            )
            df[f'volume_ratio_{window}d'] = df['Volume'] / df[f'volume_sma_{window}d']
            
        return df
    
    def calculate_price_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        df['rsi'] = df.groupby('ticker').apply(
            lambda x: self._calculate_rsi(x['Close'])
        ).reset_index(level=0, drop=True)
        
        df['macd'], df['macd_signal'] = df.groupby('ticker').apply(
            lambda x: self._calculate_macd(x['Close'])
        ).reset_index(level=0, drop=True)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        return macd, signal
    
    def calculate_insider_trading_features(self, df: pd.DataFrame, insider_transactions: pd.DataFrame) -> pd.DataFrame:
        if insider_transactions.empty:
            df['insider_buys'] = 0
            df['insider_sells'] = 0
            df['insider_net_activity'] = 0
            return df
        
        insider_transactions['transactionDate'] = pd.to_datetime(insider_transactions['transactionDate'])
        
        for window in self.feature_windows:
            df[f'insider_buys_{window}d'] = 0
            df[f'insider_sells_{window}d'] = 0
            
        for _, tx in insider_transactions.iterrows():
            tx_date = tx['transactionDate']
            tx_type = tx.get('transactionShares', 0)
            
            for window in self.feature_windows:
                mask = (df['Date'] >= tx_date - pd.Timedelta(days=window)) & (df['Date'] <= tx_date)
                if tx_type > 0:
                    df.loc[mask, f'insider_buys_{window}d'] += 1
                else:
                    df.loc[mask, f'insider_sells_{window}d'] += 1
                    
        df['insider_net_activity'] = df['insider_buys_30d'] - df['insider_sells_30d']
        
        return df
    
    def calculate_ownership_features(self, df: pd.DataFrame, fundamental_data: dict) -> pd.DataFrame:
        df['market_cap'] = fundamental_data.get('market_cap', 0)
        df['pe_ratio'] = fundamental_data.get('pe_ratio', 0)
        df['insider_percent'] = fundamental_data.get('insider_percent', 0)
        df['institutional_percent'] = fundamental_data.get('institutional_percent', 0)
        df['beta'] = fundamental_data.get('beta', 1.0)
        
        return df
    
    def calculate_abnormal_returns(self, df: pd.DataFrame, market_returns: pd.Series) -> pd.DataFrame:
        df['market_return'] = market_returns
        
        for window in self.feature_windows:
            df[f'abnormal_return_{window}d'] = (
                df[f'return_{window}d'] - df['market_return']
            )
            
        return df
    
    def create_labels(self, df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
        df = df.sort_values(['ticker', 'Date'])
        
        df['future_return'] = df.groupby('ticker')['Close'].shift(-30) / df['Close'] - 1
        
        df['label'] = (df['future_return'] > threshold).astype(int)
        
        return df
    
    def engineer_all_features(
        self,
        stock_data: pd.DataFrame,
        insider_transactions: Optional[pd.DataFrame] = None,
        fundamental_data: Optional[dict] = None,
        market_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        logger.info("Engineering features...")
        
        df = stock_data.copy()
        
        df = self.calculate_returns(df)
        logger.info("Calculated returns")
        
        df = self.calculate_moving_averages(df)
        logger.info("Calculated moving averages")
        
        df = self.calculate_volume_features(df)
        logger.info("Calculated volume features")
        
        df = self.calculate_price_momentum(df)
        logger.info("Calculated momentum indicators")
        
        if insider_transactions is not None:
            df = self.calculate_insider_trading_features(df, insider_transactions)
            logger.info("Calculated insider trading features")
            
        if fundamental_data is not None:
            df = self.calculate_ownership_features(df, fundamental_data)
            logger.info("Calculated ownership features")
            
        if market_returns is not None:
            df = self.calculate_abnormal_returns(df, market_returns)
            logger.info("Calculated abnormal returns")
            
        df = self.create_labels(df)
        logger.info("Created labels")
        
        df = df.dropna()
        
        logger.info(f"Final dataset shape: {df.shape}")
        
        return df


def get_feature_columns() -> List[str]:
    feature_cols = []
    windows = [7, 14, 30, 60]
    
    for window in windows:
        feature_cols.extend([
            f'return_{window}d',
            f'volatility_{window}d',
            f'sma_{window}d',
            f'ema_{window}d',
            f'volume_ratio_{window}d',
            f'abnormal_return_{window}d'
        ])
    
    feature_cols.extend([
        'sma_50_200_ratio',
        'rsi',
        'macd',
        'macd_signal',
        'insider_buys_30d',
        'insider_sells_30d',
        'insider_net_activity',
        'market_cap',
        'pe_ratio',
        'insider_percent',
        'institutional_percent',
        'beta'
    ])
    
    return feature_cols


if __name__ == "__main__":
    pass
