import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy.orm import Session
from app.crud.crud_transaction import transaction as transaction_crud
from app.models.transaction import Transaction

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from transaction data."""
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def calculate_spending_patterns(transactions: List[Transaction]) -> Dict:
    """Calculate spending patterns from transaction history."""
    if not transactions:
        return {}
    
    df = pd.DataFrame([{
        'amount': t.amount,
        'type': t.type,
        'category_id': t.category_id,
        'date': t.date
    } for t in transactions])
    
    # Only analyze expenses
    expenses_df = df[df['type'] == 'expense']
    if expenses_df.empty:
        return {}
    
    # Extract time features
    expenses_df = extract_time_features(expenses_df)
    
    patterns = {
        'daily_avg': expenses_df.groupby('day_of_week')['amount'].mean().to_dict(),
        'monthly_avg': expenses_df.groupby('month')['amount'].mean().to_dict(),
        'category_distribution': expenses_df.groupby('category_id')['amount'].sum().to_dict(),
        'weekend_vs_weekday': {
            'weekend_avg': expenses_df[expenses_df['is_weekend'] == 1]['amount'].mean(),
            'weekday_avg': expenses_df[expenses_df['is_weekend'] == 0]['amount'].mean()
        }
    }
    
    return patterns

def predict_monthly_expenses(
    db: Session,
    user_id: int,
    recent_transactions: List[Transaction]
) -> Dict:
    """Predict monthly expenses based on historical data."""
    if not recent_transactions:
        return {}
    
    df = pd.DataFrame([{
        'amount': t.amount,
        'type': t.type,
        'category_id': t.category_id,
        'date': t.date
    } for t in recent_transactions])
    
    # Only analyze expenses
    expenses_df = df[df['type'] == 'expense']
    if expenses_df.empty:
        return {}
    
    # Group by month and calculate statistics
    monthly_expenses = expenses_df.groupby(expenses_df['date'].dt.to_period('M'))['amount'].agg(['sum', 'mean', 'std'])
    
    # Calculate trend using simple moving average
    trend = monthly_expenses['sum'].rolling(window=3, min_periods=1).mean()
    
    # Predict next month's expenses
    next_month_prediction = trend.iloc[-1] if not trend.empty else monthly_expenses['mean'].mean()
    
    # Calculate confidence interval
    confidence_interval = 1.96 * monthly_expenses['std'].mean() if len(monthly_expenses) > 1 else next_month_prediction * 0.2
    
    return {
        'predicted_amount': float(next_month_prediction),
        'lower_bound': float(max(0, next_month_prediction - confidence_interval)),
        'upper_bound': float(next_month_prediction + confidence_interval),
        'confidence_level': 0.95
    }

def detect_unusual_transactions(
    transactions: List[Transaction],
    threshold_zscore: float = 2.0
) -> List[Dict]:
    """Detect unusual transactions using Z-score method."""
    if not transactions:
        return []
    
    df = pd.DataFrame([{
        'id': t.id,
        'amount': t.amount,
        'type': t.type,
        'category_id': t.category_id,
        'date': t.date
    } for t in transactions])
    
    # Calculate Z-scores for amounts within each category
    unusual_transactions = []
    for category_id in df['category_id'].unique():
        category_df = df[df['category_id'] == category_id]
        if len(category_df) >= 3:  # Need at least 3 transactions for meaningful Z-score
            z_scores = np.abs((category_df['amount'] - category_df['amount'].mean()) / category_df['amount'].std())
            unusual_idx = z_scores > threshold_zscore
            unusual_transactions.extend(category_df[unusual_idx].to_dict('records'))
    
    return unusual_transactions
