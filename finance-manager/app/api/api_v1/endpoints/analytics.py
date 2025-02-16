from typing import Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app import crud, models
from app.api import deps
from app.ml import features

router = APIRouter()

@router.get("/spending-patterns", response_model=Dict)
def get_spending_patterns(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Get user's spending patterns analysis."""
    transactions = crud.transaction.get_multi_by_owner(
        db=db, owner_id=current_user.id, limit=1000
    )
    patterns = features.calculate_spending_patterns(transactions)
    
    # Enhance with category names
    if 'category_distribution' in patterns:
        category_distribution = {}
        for cat_id, amount in patterns['category_distribution'].items():
            category = crud.category.get(db=db, id=cat_id)
            if category:
                category_distribution[category.name] = amount
        patterns['category_distribution'] = category_distribution
    
    return patterns

@router.get("/monthly-prediction", response_model=Dict)
def get_monthly_prediction(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Get prediction for next month's expenses."""
    # Get last 6 months of transactions
    six_months_ago = datetime.utcnow() - timedelta(days=180)
    transactions = crud.transaction.get_by_date_range(
        db=db,
        owner_id=current_user.id,
        start_date=six_months_ago,
        end_date=datetime.utcnow(),
    )
    
    prediction = features.predict_monthly_expenses(
        db=db,
        user_id=current_user.id,
        recent_transactions=transactions
    )
    return prediction

@router.get("/unusual-transactions", response_model=List[Dict])
def get_unusual_transactions(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Detect unusual transactions in user's history."""
    # Get last 3 months of transactions
    three_months_ago = datetime.utcnow() - timedelta(days=90)
    transactions = crud.transaction.get_by_date_range(
        db=db,
        owner_id=current_user.id,
        start_date=three_months_ago,
        end_date=datetime.utcnow(),
    )
    
    unusual_transactions = features.detect_unusual_transactions(transactions)
    
    # Enhance with category names
    for transaction in unusual_transactions:
        category = crud.category.get(db=db, id=transaction['category_id'])
        transaction['category_name'] = category.name if category else "Unknown"
    
    return unusual_transactions
