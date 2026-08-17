from flask import Blueprint, jsonify, request

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    return jsonify({
        'status': 'info',
        'message': 'Auth registration endpoint placeholder for Phase 2'
    }), 200

@auth_bp.route('/login', methods=['POST'])
def login():
    return jsonify({
        'status': 'info',
        'message': 'Auth login endpoint placeholder for Phase 2'
    }), 200

@auth_bp.route('/me', methods=['GET'])
def get_current_user():
    return jsonify({
        'status': 'info',
        'message': 'Auth user profile endpoint placeholder for Phase 2'
    }), 200
