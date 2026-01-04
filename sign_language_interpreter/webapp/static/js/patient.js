const socket = io();
let videoElement = document.getElementById('videoElement');
let startBtn = document.getElementById('startBtn');
let stopBtn = document.getElementById('stopBtn');
let detectionStatus = document.getElementById('detectionStatus');
let currentSign = document.getElementById('currentSign');
let confidenceBar = document.getElementById('confidenceBar');
let conversationHistory = document.getElementById('conversationHistory');

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
        return true;
    } catch (error) {
        console.error('Camera error:', error);
        alert('Could not access camera. Please check permissions.');
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
    detectionStatus.innerHTML = '<strong>Status:</strong> <span class="text-success">Detecting...</span>';
    
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
    detectionStatus.innerHTML = '<strong>Status:</strong> <span class="text-danger">Stopped</span>';
    
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
        currentSign.innerHTML = `<strong class="text-success">${data.text}</strong>`;
        
        const confidence = Math.round(data.confidence * 100);
        confidenceBar.style.width = confidence + '%';
        confidenceBar.textContent = confidence + '%';
        
        if (confidence > 70) {
            confidenceBar.className = 'progress-bar bg-success';
        } else if (confidence > 50) {
            confidenceBar.className = 'progress-bar bg-warning';
        } else {
            confidenceBar.className = 'progress-bar bg-danger';
        }
        
        // Add to history
        addToHistory('You', data.text);
    }
});

// Handle doctor responses
socket.on('doctor_response', (data) => {
    addToHistory(data.doctor_name, data.text);
    
    // Speak the doctor's message
    speakText(data.text);
});

// Add message to conversation history
function addToHistory(sender, message) {
    const timestamp = new Date().toLocaleTimeString();
    const messageDiv = document.createElement('div');
    messageDiv.className = 'mb-2 p-2 rounded ' + (sender === 'You' ? 'bg-primary text-white' : 'bg-secondary text-white');
    messageDiv.innerHTML = `<small><strong>${sender}</strong> (${timestamp})</small><br>${message}`;
    
    if (conversationHistory.querySelector('.text-muted')) {
        conversationHistory.innerHTML = '';
    }
    
    conversationHistory.appendChild(messageDiv);
    conversationHistory.scrollTop = conversationHistory.scrollHeight;
}

// Text-to-speech for doctor's messages
function speakText(text) {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        window.speechSynthesis.speak(utterance);
    }
}

// Initialize camera on page load
window.addEventListener('load', () => {
    initCamera();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
