const socket = io();
let translationFeed = document.getElementById('translationFeed');
let doctorMessage = document.getElementById('doctorMessage');
let sendMessageBtn = document.getElementById('sendMessageBtn');
let sessionStatus = document.getElementById('sessionStatus');
let messageCount = document.getElementById('messageCount');
let messageCountStat = document.getElementById('messageCountStat');
let sessionTime = document.getElementById('sessionTime');
let sessionDuration = document.getElementById('sessionDuration');
let consultationNotes = document.getElementById('consultationNotes');
let saveNotesBtn = document.getElementById('saveNotesBtn');
let exportNotesBtn = document.getElementById('exportNotesBtn');
let clearFeedBtn = document.getElementById('clearFeedBtn');
let charCount = document.getElementById('charCount');
let audioPlayer = document.getElementById('audioPlayer');

let messageCounter = 0;
let sessionStartTime = null;
let timerInterval = null;

// Join doctor room on load
window.addEventListener('load', () => {
    socket.emit('join_session', { room: 'doctor_room' });
    sessionStartTime = Date.now();
    startTimer();
    
    // Initialize UI
    updateSessionStats();
});

// Start session timer
function startTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        const timeString = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        
        sessionTime.textContent = timeString;
        sessionDuration.textContent = timeString;
    }, 1000);
}

// Receive translations from patient
socket.on('translation', (data) => {
    if (data.text) {
        messageCounter++;
        updateMessageCount();
        
        const timestamp = new Date().toLocaleTimeString();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message-item mb-3 p-3 rounded border-start border-primary border-4';
        messageDiv.style.background = 'var(--card-bg)';
        messageDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-start mb-2">
                <div class="d-flex align-items-center">
                    <i class="fas fa-user-injured text-primary me-2"></i>
                    <strong class="text-primary">Patient (Sign Language)</strong>
                </div>
                <small class="text-muted">${timestamp}</small>
            </div>
            <div class="message-text fs-5 mb-2">${data.text}</div>
            <div class="message-meta d-flex justify-content-between align-items-center">
                <span class="badge bg-info">
                    <i class="fas fa-chart-line me-1"></i>
                    Confidence: ${Math.round(data.confidence * 100)}%
                </span>
                <button class="btn btn-sm btn-outline-secondary copy-btn" onclick="copyToClipboard('${data.text}')">
                    <i class="fas fa-copy"></i>
                </button>
            </div>
        `;
        
        // Clear placeholder if it exists
        if (translationFeed.querySelector('.text-center')) {
            translationFeed.innerHTML = '';
        }
        
        translationFeed.appendChild(messageDiv);
        translationFeed.scrollTop = translationFeed.scrollHeight;
        
        // Play audio if available
        if (data.audio) {
            playAudio(data.audio);
        }
        
        // Auto-add to notes with timestamp
        const noteText = `[${timestamp}] Patient: ${data.text}\\n`;
        consultationNotes.value += noteText;
        
        // Show notification
        showNotification('New patient message', data.text);
        
        // Update stats
        updateSessionStats();
    }
});

// Play audio from base64
function playAudio(audioBase64) {
    try {
        audioPlayer.src = 'data:audio/mp3;base64,' + audioBase64;
        audioPlayer.play().catch(e => console.log('Audio play failed:', e));
    } catch (error) {
        console.error('Audio playback error:', error);
    }
}

// Send message to patient
function sendDoctorMessage() {
    const message = doctorMessage.value.trim();
    if (message) {
        socket.emit('doctor_message', {
            text: message,
            doctor_name: 'Doctor',
            patient_room: 'patient_room'
        });
        
        // Add to feed
        const timestamp = new Date().toLocaleTimeString();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message-item mb-3 p-3 rounded';
        messageDiv.style.background = 'var(--success-color)';
        messageDiv.style.color = 'white';
        messageDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-start mb-2">
                <div class="d-flex align-items-center">
                    <i class="fas fa-user-md me-2"></i>
                    <strong>You (Doctor)</strong>
                </div>
                <small class="opacity-75">${timestamp}</small>
            </div>
            <div class="message-text">${message}</div>
        `;
        translationFeed.appendChild(messageDiv);
        translationFeed.scrollTop = translationFeed.scrollHeight;
        
        // Add to notes
        const noteText = `[${timestamp}] Doctor: ${message}\\n`;
        consultationNotes.value += noteText;
        
        doctorMessage.value = '';
        updateCharCount();
    }
}

// Event listeners
sendMessageBtn.addEventListener('click', sendDoctorMessage);

doctorMessage.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendDoctorMessage();
    }
});

doctorMessage.addEventListener('input', updateCharCount);

// Character count update
function updateCharCount() {
    const count = doctorMessage.value.length;
    charCount.textContent = count;
    
    if (count > 180) {
        charCount.style.color = 'var(--danger-color)';
    } else if (count > 150) {
        charCount.style.color = 'var(--warning-color)';
    } else {
        charCount.style.color = 'var(--text-secondary)';
    }
}

// Update message count
function updateMessageCount() {
    messageCount.textContent = messageCounter;
    messageCountStat.textContent = messageCounter;
}

// Update session statistics
function updateSessionStats() {
    // Could add more stats here like average confidence, etc.
}

// Save consultation notes
saveNotesBtn.addEventListener('click', () => {
    const notes = consultationNotes.value;
    if (notes.trim()) {
        const timestamp = new Date().toISOString();
        const savedNotes = {
            timestamp: timestamp,
            notes: notes,
            messageCount: messageCounter,
            sessionDuration: sessionDuration.textContent
        };
        
        localStorage.setItem('consultation_' + timestamp, JSON.stringify(savedNotes));
        showAlert('Consultation notes saved successfully!', 'success');
    } else {
        showAlert('No notes to save.', 'warning');
    }
});

// Export session data
exportNotesBtn.addEventListener('click', () => {
    const sessionData = {
        timestamp: new Date().toISOString(),
        notes: consultationNotes.value,
        messageCount: messageCounter,
        sessionDuration: sessionDuration.textContent,
        messages: Array.from(translationFeed.querySelectorAll('.message-item')).map(msg => ({
            text: msg.querySelector('.message-text').textContent,
            timestamp: msg.querySelector('small').textContent,
            sender: msg.querySelector('strong').textContent
        }))
    };
    
    const dataStr = JSON.stringify(sessionData, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `session_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    showAlert('Session data exported successfully!', 'success');
});

// Clear translation feed
clearFeedBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to clear the translation feed?')) {
        translationFeed.innerHTML = `
            <div class="text-center text-muted">
                <i class="fas fa-user-injured fa-3x mb-3"></i>
                <h5>Feed cleared</h5>
                <p class="mb-0">New patient translations will appear here</p>
            </div>
        `;
    }
});

// Quick phrase buttons
document.getElementById('commonPhrase1').addEventListener('click', () => {
    doctorMessage.value = 'How are you feeling?';
    updateCharCount();
});

document.getElementById('commonPhrase2').addEventListener('click', () => {
    doctorMessage.value = 'Can you describe the pain?';
    updateCharCount();
});

document.getElementById('commonPhrase3').addEventListener('click', () => {
    doctorMessage.value = 'Thank you for explaining';
    updateCharCount();
});

// Utility functions
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showAlert('Text copied to clipboard!', 'success');
    }).catch(() => {
        showAlert('Failed to copy text', 'danger');
    });
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        <i class="fas fa-${type === 'danger' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : 'info-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('.main-content .container-fluid');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 3000);
}

function showNotification(title, body) {
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(title, {
            body: body,
            icon: '/static/favicon.ico'
        });
    }
}

// Initialize notifications
window.addEventListener('load', () => {
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (timerInterval) {
        clearInterval(timerInterval);
    }
});