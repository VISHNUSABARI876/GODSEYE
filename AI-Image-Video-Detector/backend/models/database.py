from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def init_db(app):
    """
    Initialize SQLAlchemy and Flask-Migrate with Flask application context.
    """
    db.init_app(app)
    migrate.init_app(app, db)
