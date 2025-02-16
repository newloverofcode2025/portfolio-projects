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

### Backend
- **Framework**: FastAPI, SQLAlchemy
- **Database**: PostgreSQL 16
- **ML/AI**: scikit-learn for predictive analytics
- **Authentication**: JWT with role-based access control
- **Testing**: pytest for unit and integration tests
- **Documentation**: OpenAPI (Swagger UI)
- **Migrations**: Alembic

### Frontend
- **Framework**: React 18
- **UI Library**: Material-UI (MUI)
- **State Management**: Redux Toolkit
- **Charts**: Recharts
- **API Client**: Axios

## Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- PostgreSQL 16
- Git

## Getting Started

### Backend Setup

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

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
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
├── frontend/           # React frontend application
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── pages/      # Page components
│   │   ├── store/      # Redux store
│   │   └── api/        # API integration
├── tests/              # Test suite
├── alembic/            # Database migrations
└── README.md
```

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

### Code Style
- Backend: Follow PEP 8 guidelines
- Frontend: Follow ESLint configuration
- Commit messages: Use conventional commits format

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

- All endpoints are protected with JWT authentication
- Passwords are hashed using bcrypt
- Environment variables are used for sensitive data
- CORS is configured for security
- Rate limiting is implemented on sensitive endpoints

## Support

If you encounter any issues or have questions:
1. Check the [Issues](../../issues) page
2. Review the documentation
3. Create a new issue with detailed information

## Acknowledgments

- FastAPI for the excellent framework
- scikit-learn for ML capabilities
- Material-UI for the component library
