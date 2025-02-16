from datetime import datetime
from typing import List
from sqlalchemy import Boolean, Column, Integer, String, Float, ForeignKey, DateTime, Enum
from sqlalchemy.orm import relationship
import enum

from app.db.base_class import Base

class TransactionType(str, enum.Enum):
    INCOME = "income"
    EXPENSE = "expense"

class User(Base):
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="owner")
    categories = relationship("Category", back_populates="owner")
    budgets = relationship("Budget", back_populates="owner")

class Category(Base):
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String)
    owner_id = Column(Integer, ForeignKey("user.id"))
    
    # Relationships
    owner = relationship("User", back_populates="categories")
    transactions = relationship("Transaction", back_populates="category")
    budgets = relationship("Budget", back_populates="category")

class Transaction(Base):
    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    type = Column(Enum(TransactionType), nullable=False)
    description = Column(String)
    date = Column(DateTime, default=datetime.utcnow)
    category_id = Column(Integer, ForeignKey("category.id"))
    owner_id = Column(Integer, ForeignKey("user.id"))
    
    # Relationships
    category = relationship("Category", back_populates="transactions")
    owner = relationship("User", back_populates="transactions")

class Budget(Base):
    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    category_id = Column(Integer, ForeignKey("category.id"))
    owner_id = Column(Integer, ForeignKey("user.id"))
    
    # Relationships
    category = relationship("Category", back_populates="budgets")
    owner = relationship("User", back_populates="budgets")
