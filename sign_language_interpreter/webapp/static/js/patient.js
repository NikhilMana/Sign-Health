const socket = io();
let videoElement = document.getElementById('videoElement');
let videoOverlay = document.getElementById('videoOverlay');
let startBtn = document.getElementById('startBtn');
let stopBtn = document.getElementById('stopBtn');
let detectionStatus = document.getElementById('detectionStatus');
let statusText = document.getElementById('statusText');
let currentSign = document.getElementById('currentSign');
let confidenceBar = document.getElementById('confidenceBar');
let confidenceText = document.getElementById('confidenceText');
let conversationHistory = document.getElementById('conversationHistory');
let videoStatus = document.getElementById('videoStatus');

let stream = null;
let isDetecting = false;
let frameInterval = null;

// Initialize webcam
async function initCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        videoElement.srcObject = stream;
        videoOverlay.style.display = 'none';
        videoStatus.textContent = 'Connected';
        videoStatus.className = 'badge bg-success';
        return true;
    } catch (error) {
        console.error('Camera error:', error);
        showAlert('Could not access camera. Please check permissions.', 'danger');
        videoOverlay.innerHTML = `
            <i class="fas fa-exclamation-triangle fa-3x text-danger mb-3"></i>
            <p class="text-danger">Camera access denied</p>
            <small class="text-muted">Please allow camera permissions and refresh</small>
        `;
        return false;
    }
}

// Start detection
startBtn.addEventListener('click', async () => {
    if (!stream) {
        const success = await initCamera();
        if (!success) return;
    }
    
    isDetecting = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    
    // Update UI
    detectionStatus.className = 'alert alert-success';
    detectionStatus.innerHTML = `
        <i class="fas fa-play-circle me-2"></i>
        <strong>Status:</strong> <span id="statusText">Detecting signs...</span>
    `;
    
    // Update status indicator
    const statusIndicator = document.querySelector('.status-indicator');
    statusIndicator.className = 'status-indicator status-active';
    statusIndicator.nextElementSibling.textContent = 'Detection active';
    
    // Join session room
    socket.emit('join_session', { room: 'doctor_room' });
    
    // Send frames every 100ms
    frameInterval = setInterval(() => {
        if (isDetecting) {
            captureAndSendFrame();
        }
    }, 100);
});

// Stop detection
stopBtn.addEventListener('click', () => {
    isDetecting = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    // Update UI
    detectionStatus.className = 'alert alert-secondary';
    detectionStatus.innerHTML = `
        <i class="fas fa-stop-circle me-2"></i>
        <strong>Status:</strong> <span id="statusText">Stopped</span>
    `;
    
    // Reset current sign
    currentSign.innerHTML = '<span class="text-muted">Waiting...</span>';
    confidenceBar.style.width = '0%';
    confidenceText.textContent = '0%';
    
    // Update status indicator
    const statusIndicator = document.querySelector('.status-indicator');
    statusIndicator.className = 'status-indicator status-inactive';
    statusIndicator.nextElementSibling.textContent = 'Ready to connect';
    
    if (frameInterval) {
        clearInterval(frameInterval);
    }
});

// Capture and send frame
function captureAndSendFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    
    const frameData = canvas.toDataURL('image/jpeg', 0.8);
    
    socket.emit('video_frame', {
        frame: frameData,
        patient_id: 'patient_1',
        doctor_room: 'doctor_room'
    });
}

// Handle detection results
socket.on('detection_result', (data) => {
    if (data.text) {
        currentSign.innerHTML = `<strong class="text-primary">${data.text}</strong>`;
        
        const confidence = Math.round(data.confidence * 100);
        confidenceBar.style.width = confidence + '%';
        confidenceText.textContent = confidence + '%';
        
        // Update progress bar color based on confidence
        if (confidence > 80) {
            confidenceBar.className = 'progress-bar bg-success';
        } else if (confidence > 60) {
            confidenceBar.className = 'progress-bar bg-warning';
        } else {
            confidenceBar.className = 'progress-bar bg-danger';
        }
        
        // Add to conversation history
        addToHistory('You', data.text, 'patient');
    }
});

// Handle doctor responses
socket.on('doctor_response', (data) => {
    addToHistory(data.doctor_name, data.text, 'doctor');
    
    // Speak the doctor's message
    speakText(data.text);
    
    // Show notification
    showNotification('New message from doctor', data.text);
});

// Add message to conversation history
function addToHistory(sender, message, type) {
    const timestamp = new Date().toLocaleTimeString();
    
    // Clear placeholder if it exists
    if (conversationHistory.querySelector('.text-center')) {
        conversationHistory.innerHTML = '';
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message-bubble message-${type} d-flex flex-column`;
    messageDiv.innerHTML = `
        <div class="d-flex justify-content-between align-items-center mb-1">
            <strong class="small">${sender}</strong>
            <small class="opacity-75">${timestamp}</small>
        </div>
        <div>${message}</div>
    `;
    
    conversationHistory.appendChild(messageDiv);
    conversationHistory.scrollTop = conversationHistory.scrollHeight;
}

// Text-to-speech for doctor's messages
function speakText(text) {
    if ('speechSynthesis' in window) {
        // Stop any ongoing speech
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        utterance.volume = 0.8;
        window.speechSynthesis.speak(utterance);
    }
}

// Show alert
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <i class="fas fa-${type === 'danger' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.main-content .container-fluid');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Show notification (if supported)
function showNotification(title, body) {
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(title, {
            body: body,
            icon: '/static/favicon.ico'
        });
    } else if ('Notification' in window && Notification.permission !== 'denied') {
        Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
                new Notification(title, {
                    body: body,
                    icon: '/static/favicon.ico'
                });
            }
        });
    }
}

// Initialize camera on page load
window.addEventListener('load', () => {
    initCamera();
    
    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    if (frameInterval) {
        clearInterval(frameInterval);
    }
});