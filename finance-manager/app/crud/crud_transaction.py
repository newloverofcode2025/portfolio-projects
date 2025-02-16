from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.crud.base import CRUDBase
from app.models.transaction import Transaction
from app.schemas.transaction import TransactionCreate, TransactionUpdate

class CRUDTransaction(CRUDBase[Transaction, TransactionCreate, TransactionUpdate]):
    def create_with_owner(
        self, db: Session, *, obj_in: TransactionCreate, owner_id: int
    ) -> Transaction:
        obj_in_data = obj_in.model_dump()
        db_obj = self.model(**obj_in_data, owner_id=owner_id)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[Transaction]:
        return (
            db.query(self.model)
            .filter(Transaction.owner_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_date_range(
        self,
        db: Session,
        *,
        owner_id: int,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0,
        limit: int = 100
    ) -> List[Transaction]:
        return (
            db.query(self.model)
            .filter(
                and_(
                    Transaction.owner_id == owner_id,
                    Transaction.date >= start_date,
                    Transaction.date <= end_date
                )
            )
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_category(
        self,
        db: Session,
        *,
        owner_id: int,
        category_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Transaction]:
        return (
            db.query(self.model)
            .filter(
                and_(
                    Transaction.owner_id == owner_id,
                    Transaction.category_id == category_id
                )
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

transaction = CRUDTransaction(Transaction)
