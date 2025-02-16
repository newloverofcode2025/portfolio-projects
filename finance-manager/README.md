# AI-Powered Personal Finance Manager

A modern, intelligent personal finance management system that helps users track expenses, analyze spending patterns, and make better financial decisions using machine learning.

## Features

- 🔐 Secure user authentication and authorization
- 💰 Expense and income tracking
- 📊 Interactive dashboard with spending analytics
- 🤖 ML-powered spending pattern prediction
- 📱 RESTful API for mobile/web clients
- 📈 Budget planning and monitoring
- 🔍 Smart transaction categorization

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, PostgreSQL
- **ML/AI**: scikit-learn for predictive analytics
- **Authentication**: JWT with role-based access control
- **Testing**: pytest for unit and integration tests
- **Documentation**: OpenAPI (Swagger UI)
- **Database Migrations**: Alembic

## Getting Started

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```
5. Run migrations:
   ```bash
   alembic upgrade head
   ```
6. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```

## Project Structure

```
finance-manager/
├── app/
│   ├── api/            # API routes
│   ├── core/           # Core functionality
│   ├── db/             # Database models
│   ├── ml/             # Machine learning models
│   └── schemas/        # Pydantic models
├── tests/              # Test suite
├── alembic/            # Database migrations
└── README.md
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
