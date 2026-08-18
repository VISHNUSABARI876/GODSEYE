import os
import re
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables FIRST before importing config and models
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

from flask import Flask, jsonify, request
from flask_cors import CORS
from sqlalchemy import text

from config import config_by_name
from models import init_db, db
from routes import auth_bp, detection_bp, history_bp

def create_app(config_name=None):
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    app = Flask(__name__)
    app.config.from_object(config_by_name.get(config_name, config_by_name['default']))

    # Configure CORS with compiled regex for dynamic Vercel subdomains
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://godseye-three.vercel.app",
        re.compile(r"^https://.*\.vercel\.app$")
    ]

    frontend_env = app.config.get('FRONTEND_URL') or os.environ.get('FRONTEND_URL')
    if frontend_env:
        for url in frontend_env.split(','):
            cleaned = url.strip()
            if cleaned and cleaned != '*' and cleaned not in allowed_origins:
                allowed_origins.append(cleaned)

    CORS(
        app,
        resources={r"/api/*": {"origins": allowed_origins}},
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
        supports_credentials=True
    )

    @app.after_request
    def apply_cors_headers(response):
        origin = request.headers.get('Origin')
        if origin:
            is_allowed = False
            if origin in allowed_origins or (isinstance(origin, str) and origin.endswith('.vercel.app')):
                is_allowed = True
            else:
                for pattern in allowed_origins:
                    if hasattr(pattern, 'search') and pattern.search(origin):
                        is_allowed = True
                        break

            if is_allowed:
                response.headers['Access-Control-Allow-Origin'] = origin
                response.headers['Access-Control-Allow-Credentials'] = 'true'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
        return response

    # Initialize Database & Flask-Migrate
    init_db(app)

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(history_bp)

    # Main System Health Check Endpoint
    @app.route('/api/health', methods=['GET'])
    def health_check():
        db_status = 'disconnected'
        is_success = False

        if app.config.get('SQLALCHEMY_DATABASE_URI'):
            try:
                # Test PostgreSQL connectivity cleanly
                db.session.execute(text('SELECT 1'))
                db_status = 'connected'
                is_success = True
            except Exception as e:
                app.logger.error("Database health check failed.")

        return jsonify({
            'success': is_success,
            'backend': 'running',
            'database': db_status,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200

    # Development Database Test Endpoint
    @app.route('/api/health/database', methods=['GET'])
    def database_health_check():
        if not app.config.get('SQLALCHEMY_DATABASE_URI'):
            return jsonify({
                'success': False,
                'database': 'PostgreSQL',
                'connection': 'DATABASE_URL missing in environment'
            }), 500

        try:
            db.session.execute(text('SELECT 1'))
            return jsonify({
                'success': True,
                'database': 'PostgreSQL',
                'connection': 'working'
            }), 200
        except Exception as e:
            app.logger.error("PostgreSQL connection check failed.")
            return jsonify({
                'success': False,
                'database': 'PostgreSQL',
                'connection': 'failed'
            }), 500

    # General Error Handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'success': False, 'message': 'Resource not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'success': False, 'message': 'Internal server error'}), 500

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    print(f"Starting Flask backend server on http://{host}:{port} ...")
    app.run(host=host, port=port, debug=True)

