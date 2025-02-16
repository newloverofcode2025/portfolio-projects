from typing import List
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.category import Category
from app.schemas.category import CategoryCreate, CategoryUpdate

class CRUDCategory(CRUDBase[Category, CategoryCreate, CategoryUpdate]):
    def create_with_owner(
        self, db: Session, *, obj_in: CategoryCreate, owner_id: int
    ) -> Category:
        obj_in_data = obj_in.model_dump()
        db_obj = self.model(**obj_in_data, owner_id=owner_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Category]:
        return (
            db.query(self.model)
            .filter(Category.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_name(
        self, db: Session, *, name: str, owner_id: int
    ) -> Category:
        return (
            db.query(self.model)
            .filter(Category.name == name, Category.owner_id == owner_id)
            .first()
        )

category = CRUDCategory(Category)
