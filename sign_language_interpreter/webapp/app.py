from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import cv2
import numpy as np
import base64
from pathlib import Path

from config import Config
from models.user import Database, User
from services.isl_detector import ISLDetector
from services.tts_service import TTSService

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize database
db = Database(app.config['DATABASE_PATH'])

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize ISL Detector (lazy loading)
isl_detector = None

def get_isl_detector():
    global isl_detector
    if isl_detector is None:
        isl_detector = ISLDetector(
            app.config['MODEL_PATH'],
            app.config['ENCODER_PATH'],
            app.config['MAX_LENGTH_PATH']
        )
    return isl_detector

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(db, int(user_id))

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        else:
            return redirect(url_for('patient_dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.authenticate(db, email, password)
        
        if user:
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        
        # Determine role based on email
        role = 'doctor' if email in app.config['DOCTOR_EMAILS'] else 'patient'
        
        user = User.create_user(db, email, password, full_name, role)
        
        if user:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Email already exists', 'error')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/patient/dashboard')
@login_required
def patient_dashboard():
    if current_user.role != 'patient':
        return redirect(url_for('index'))
    return render_template('patient_dashboard.html', user=current_user)

@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor':
        return redirect(url_for('index'))
    return render_template('doctor_dashboard.html', user=current_user)

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    print(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Client disconnected: {request.sid}')

@socketio.on('join_session')
def handle_join_session(data):
    room = data.get('room')
    join_room(room)
    emit('joined', {'room': room}, room=request.sid)

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Decode base64 image
        img_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame with ISL detector
        detector = get_isl_detector()
        result = detector.process_frame(frame)
        
        if result['text'] and result['confidence'] > 0.7:
            # Generate speech
            audio_base64 = TTSService.text_to_speech(result['text'])
            
            # Send to doctor's room
            emit('translation', {
                'text': result['text'],
                'confidence': result['confidence'],
                'audio': audio_base64,
                'patient_id': data.get('patient_id')
            }, room=data.get('doctor_room'), broadcast=True)
            
            # Send confirmation to patient
            emit('detection_result', result, room=request.sid)
    
    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('error', {'message': str(e)}, room=request.sid)

@socketio.on('doctor_message')
def handle_doctor_message(data):
    """Send doctor's text message to patient"""
    emit('doctor_response', {
        'text': data['text'],
        'doctor_name': data.get('doctor_name', 'Doctor')
    }, room=data.get('patient_room'))

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
