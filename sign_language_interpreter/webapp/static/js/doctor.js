const socket = io();
let translationFeed = document.getElementById('translationFeed');
let doctorMessage = document.getElementById('doctorMessage');
let sendMessageBtn = document.getElementById('sendMessageBtn');
let sessionStatus = document.getElementById('sessionStatus');
let messageCount = document.getElementById('messageCount');
let sessionTime = document.getElementById('sessionTime');
let consultationNotes = document.getElementById('consultationNotes');
let saveNotesBtn = document.getElementById('saveNotesBtn');
let audioPlayer = document.getElementById('audioPlayer');

let messageCounter = 0;
let sessionStartTime = null;
let timerInterval = null;

// Join doctor room on load
window.addEventListener('load', () => {
    socket.emit('join_session', { room: 'doctor_room' });
    sessionStatus.textContent = 'Active';
    sessionStatus.className = 'badge bg-success';
    
    sessionStartTime = Date.now();
    startTimer();
});

// Start session timer
function startTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        sessionTime.textContent = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    }, 1000);
}

// Receive translations from patient
socket.on('translation', (data) => {
    if (data.text) {
        messageCounter++;
        messageCount.textContent = messageCounter;
        
        const timestamp = new Date().toLocaleTimeString();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'mb-3 p-3 rounded bg-white border-start border-primary border-4';
        messageDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-start mb-2">
                <strong class="text-primary">Patient (Sign Language)</strong>
                <small class="text-muted">${timestamp}</small>
            </div>
            <div class="fs-5">${data.text}</div>
            <div class="mt-2">
                <span class="badge bg-info">Confidence: ${Math.round(data.confidence * 100)}%</span>
            </div>
        `;
        
        if (translationFeed.querySelector('.text-muted')) {
            translationFeed.innerHTML = '';
        }
        
        translationFeed.appendChild(messageDiv);
        translationFeed.scrollTop = translationFeed.scrollHeight;
        
        // Play audio if available
        if (data.audio) {
            playAudio(data.audio);
        }
        
        // Auto-add to notes
        const noteText = `[${timestamp}] Patient: ${data.text}\n`;
        consultationNotes.value += noteText;
    }
});

// Play audio from base64
function playAudio(audioBase64) {
    try {
        audioPlayer.src = 'data:audio/mp3;base64,' + audioBase64;
        audioPlayer.play();
    } catch (error) {
        console.error('Audio playback error:', error);
    }
}

// Send message to patient
sendMessageBtn.addEventListener('click', () => {
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
        messageDiv.className = 'mb-3 p-3 rounded bg-success text-white';
        messageDiv.innerHTML = `
            <div class="d-flex justify-content-between align-items-start mb-2">
                <strong>You (Doctor)</strong>
                <small>${timestamp}</small>
            </div>
            <div>${message}</div>
        `;
        translationFeed.appendChild(messageDiv);
        translationFeed.scrollTop = translationFeed.scrollHeight;
        
        // Add to notes
        const noteText = `[${timestamp}] Doctor: ${message}\n`;
        consultationNotes.value += noteText;
        
        doctorMessage.value = '';
    }
});

// Allow Enter key to send message
doctorMessage.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessageBtn.click();
    }
});

// Save consultation notes
saveNotesBtn.addEventListener('click', () => {
    const notes = consultationNotes.value;
    if (notes) {
        // Save to localStorage for now (can be extended to save to database)
        const timestamp = new Date().toISOString();
        const savedNotes = {
            timestamp: timestamp,
            notes: notes
        };
        
        localStorage.setItem('consultation_' + timestamp, JSON.stringify(savedNotes));
        
        alert('Consultation notes saved successfully!');
    } else {
        alert('No notes to save.');
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (timerInterval) {
        clearInterval(timerInterval);
    }
});
