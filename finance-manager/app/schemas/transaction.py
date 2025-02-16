from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from app.db.models import TransactionType

class TransactionBase(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount must be positive")
    type: TransactionType
    description: Optional[str] = None
    category_id: int
    date: Optional[datetime] = None

class TransactionCreate(TransactionBase):
    pass

class TransactionUpdate(BaseModel):
    amount: Optional[float] = Field(None, gt=0)
    type: Optional[TransactionType] = None
    description: Optional[str] = None
    category_id: Optional[int] = None
    date: Optional[datetime] = None

class TransactionInDBBase(TransactionBase):
    id: int
    owner_id: int
    date: datetime

    class Config:
        from_attributes = True

class Transaction(TransactionInDBBase):
    pass

class TransactionWithCategory(Transaction):
    category_name: str
