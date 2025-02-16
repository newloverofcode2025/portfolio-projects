from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()

@router.post("/", response_model=schemas.Category)
def create_category(
    *,
    db: Session = Depends(deps.get_db),
    category_in: schemas.CategoryCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Create new category."""
    # Check if category with same name exists
    category = crud.category.get_by_name(
        db=db, name=category_in.name, owner_id=current_user.id
    )
    if category:
        raise HTTPException(
            status_code=400,
            detail="Category with this name already exists",
        )
    category = crud.category.create_with_owner(
        db=db, obj_in=category_in, owner_id=current_user.id
    )
    return category

@router.get("/", response_model=List[schemas.Category])
def read_categories(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Retrieve categories."""
    categories = crud.category.get_multi_by_owner(
        db=db, owner_id=current_user.id, skip=skip, limit=limit
    )
    return categories

@router.put("/{category_id}", response_model=schemas.Category)
def update_category(
    *,
    db: Session = Depends(deps.get_db),
    category_id: int,
    category_in: schemas.CategoryUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Update a category."""
    category = crud.category.get(db=db, id=category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    if category.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    if category_in.name:
        existing_category = crud.category.get_by_name(
            db=db, name=category_in.name, owner_id=current_user.id
        )
        if existing_category and existing_category.id != category_id:
            raise HTTPException(
                status_code=400,
                detail="Category with this name already exists",
            )
    category = crud.category.update(db=db, db_obj=category, obj_in=category_in)
    return category

@router.delete("/{category_id}", response_model=schemas.Category)
def delete_category(
    *,
    db: Session = Depends(deps.get_db),
    category_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """Delete a category."""
    category = crud.category.get(db=db, id=category_id)
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    if category.owner_id != current_user.id:
        raise HTTPException(status_code=400, detail="Not enough permissions")
    # Check if category has any transactions
    transactions = crud.transaction.get_by_category(
        db=db, owner_id=current_user.id, category_id=category_id
    )
    if transactions:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete category with existing transactions",
        )
    category = crud.category.remove(db=db, id=category_id)
    return category
