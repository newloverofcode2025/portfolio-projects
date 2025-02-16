from typing import Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime, date

from app import crud, models, schemas
from app.api import deps

router = APIRouter()

@router.post("/", response_model=schemas.Transaction)
def create_transaction(
    *,
    db: Session = Depends(deps.get_db),
    transaction_in: schemas.TransactionCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Create new transaction."""
    # Verify category belongs to user
    category = crud.category.get(db=db, id=transaction_in.category_id)
    if not category or category.owner_id != current_user.id:
        raise HTTPException(
            status_code=400,
            detail="Category not found or doesn't belong to user",
        )
    transaction = crud.transaction.create_with_owner(
        db=db, obj_in=transaction_in, owner_id=current_user.id
    )
    return transaction

@router.get("/", response_model=List[schemas.TransactionWithCategory])
def read_transactions(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    category_id: Optional[int] = Query(None),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Retrieve transactions."""
    if start_date and end_date:
        transactions = crud.transaction.get_by_date_range(
            db=db,
            owner_id=current_user.id,
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.max.time()),
            skip=skip,
            limit=limit,
        )
    elif category_id:
        transactions = crud.transaction.get_by_category(
            db=db,
            owner_id=current_user.id,
            category_id=category_id,
            skip=skip,
            limit=limit,
        )
    else:
        transactions = crud.transaction.get_multi_by_owner(
            db=db, owner_id=current_user.id, skip=skip, limit=limit
        )
    
    # Enhance transactions with category names
    result = []
    for t in transactions:
        t_dict = schemas.Transaction.model_validate(t).model_dump()
        category = crud.category.get(db=db, id=t.category_id)
        t_dict["category_name"] = category.name if category else "Unknown"
        result.append(schemas.TransactionWithCategory(**t_dict))
    return result

@router.put("/{transaction_id}", response_model=schemas.Transaction)
def update_transaction(
    *,
    db: Session = Depends(deps.get_db),
    transaction_id: int,
    transaction_in: schemas.TransactionUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Update a transaction."""
    transaction = crud.transaction.get(db=db, id=transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    if transaction.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    if transaction_in.category_id:
        category = crud.category.get(db=db, id=transaction_in.category_id)
        if not category or category.owner_id != current_user.id:
            raise HTTPException(
                status_code=400,
                detail="Category not found or doesn't belong to user",
            )
    transaction = crud.transaction.update(db=db, db_obj=transaction, obj_in=transaction_in)
    return transaction

@router.delete("/{transaction_id}", response_model=schemas.Transaction)
def delete_transaction(
    *,
    db: Session = Depends(deps.get_db),
    transaction_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Delete a transaction."""
    transaction = crud.transaction.get(db=db, id=transaction_id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    if transaction.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    transaction = crud.transaction.remove(db=db, id=transaction_id)
    return transaction
