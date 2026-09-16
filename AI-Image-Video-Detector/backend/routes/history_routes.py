from flask import Blueprint, jsonify

history_bp = Blueprint('history', __name__, url_prefix='/api/history')

@history_bp.route('/', methods=['GET'])
def get_history():
    return jsonify({
        'status': 'info',
        'history': [],
        'message': 'History endpoint placeholder for Phase 2'
    }), 200
