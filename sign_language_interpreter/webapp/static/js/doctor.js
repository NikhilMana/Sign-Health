/**
 * Doctor Dashboard — WebRTC video call + ISL translation feed.
 */

const socket = io();

// ── DOM refs ─────────────────────────────────────────
const remoteVideo = document.getElementById('remoteVideo');
const localVideo = document.getElementById('localVideo');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const patientOverlay = document.getElementById('patientOverlay');
const sessionTimerEl = document.getElementById('sessionTimerOverlay');
const sessionTimeEl = document.getElementById('sessionTime');
const pipContainer = document.getElementById('pipContainer');
const callControls = document.getElementById('callControls');
const callStatusBar = document.getElementById('callStatusBar');
const islToastContainer = document.getElementById('islToastContainer');

const startCallBtn = document.getElementById('startCallBtn');
const endCallBtn = document.getElementById('endCallBtn');
const toggleMicBtn = document.getElementById('toggleMicBtn');
const toggleCamBtn = document.getElementById('toggleCamBtn');

const translationFeed = document.getElementById('translationFeed');
const feedPlaceholder = document.getElementById('feedPlaceholder');
const doctorMessage = document.getElementById('doctorMessage');
const sendMessageBtn = document.getElementById('sendMessageBtn');
const messageCountEl = document.getElementById('messageCount');
const consultationNotes = document.getElementById('consultationNotes');
const saveNotesBtn = document.getElementById('saveNotesBtn');
const exportNotesBtn = document.getElementById('exportNotesBtn');
const audioPlayer = document.getElementById('audioPlayer');
const connectionDot = document.getElementById('connectionDot');
const connectionLabel = document.getElementById('connectionLabel');

// ── State ────────────────────────────────────────────
let messageCounter = 0;
let sessionStart = null;
let timerInterval = null;

// ── WebRTC setup ─────────────────────────────────────
const rtc = new SignHealthRTC(socket, {
    role: 'doctor',
    localVideo: localVideo,
    remoteVideo: remoteVideo,
    onCallStateChange: handleCallState,
    onConnectionStateChange: handleConnectionState,
});

// ── Initialise ───────────────────────────────────────
window.addEventListener('load', () => {
    socket.emit('join_session', { room: 'doctor_room' });

    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
});

// ── Call lifecycle ───────────────────────────────────

startCallBtn.addEventListener('click', async () => {
    try {
        setCallStatus('connecting', 'Connecting…');
        // Doctor uses audio-only — no camera needed (frees webcam for patient)
        await rtc.startAudioOnlyStream();
        showCallUI();
        rtc.initiateCall('patient_room');

        socket.emit('call_request', {
            doctor_name: 'Doctor',
            patient_room: 'patient_room',
        });
    } catch (err) {
        console.error('Start call error:', err);
        setCallStatus('disconnected', 'Microphone access denied');
    }
});

endCallBtn.addEventListener('click', () => {
    rtc.endCall('patient_room');
    hideCallUI();
    setCallStatus('disconnected', 'Call ended');
    stopTimer();
});

toggleMicBtn.addEventListener('click', () => {
    const muted = rtc.toggleMute();
    toggleMicBtn.innerHTML = muted
        ? '<i class="fas fa-microphone-slash"></i>'
        : '<i class="fas fa-microphone"></i>';
    toggleMicBtn.classList.toggle('active', muted);
});

toggleCamBtn.addEventListener('click', () => {
    const off = rtc.toggleCamera();
    toggleCamBtn.innerHTML = off
        ? '<i class="fas fa-video-slash"></i>'
        : '<i class="fas fa-video"></i>';
    toggleCamBtn.classList.toggle('active', off);
});

function handleCallState(state) {
    switch (state) {
        case 'calling':
            setCallStatus('connecting', 'Calling patient…');
            break;
        case 'in_call':
            setCallStatus('connected', 'In Call');
            startTimer();
            break;
        case 'remote_stream_connected':
            setCallStatus('connected', 'Patient connected');
            videoPlaceholder.style.display = 'none';
            remoteVideo.classList.add('connected');
            patientOverlay.style.display = 'flex';
            break;
        case 'call_ended':
            hideCallUI();
            setCallStatus('disconnected', 'Call ended');
            stopTimer();
            break;
        case 'connection_lost':
            setCallStatus('disconnected', 'Connection lost');
            connectionDot.className = 'dot dot-red';
            connectionLabel.textContent = 'Disconnected';
            break;
    }
}

function handleConnectionState(state) {
    if (state === 'connected') {
        connectionDot.className = 'dot dot-green';
        connectionLabel.textContent = 'Connected';
    } else if (state === 'connecting') {
        connectionDot.className = 'dot dot-amber';
        connectionLabel.textContent = 'Connecting…';
    }
}

function showCallUI() {
    // Doctor is audio-only — no PiP self-view, no camera toggle
    pipContainer.style.display = 'none';
    toggleCamBtn.style.display = 'none';
    callControls.style.display = 'flex';
    sessionTimerEl.style.display = 'flex';
    startCallBtn.style.display = 'none';
}

function hideCallUI() {
    videoPlaceholder.style.display = 'flex';
    patientOverlay.style.display = 'none';
    pipContainer.style.display = 'none';
    callControls.style.display = 'none';
    sessionTimerEl.style.display = 'none';
    startCallBtn.style.display = 'inline-flex';
    remoteVideo.classList.remove('connected');
    rtc.stopLocalStream();
}

function setCallStatus(type, text) {
    callStatusBar.className = `call-status-bar call-status-${type}`;
    callStatusBar.innerHTML = `<i class="fas fa-circle me-1" style="font-size:0.5rem;vertical-align:middle;"></i> ${text}`;
}

// ── Timer ─────────────────────────────────────────────

function startTimer() {
    sessionStart = Date.now();
    timerInterval = setInterval(() => {
        const s = Math.floor((Date.now() - sessionStart) / 1000);
        const m = Math.floor(s / 60);
        sessionTimeEl.textContent =
            `${String(m).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
    }, 1000);
}

function stopTimer() {
    if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
}

// ── ISL Translations ─────────────────────────────────

socket.on('translation', (data) => {
    if (!data.text) return;

    messageCounter++;
    messageCountEl.textContent = messageCounter;

    // Clear placeholder
    if (feedPlaceholder) feedPlaceholder.remove();

    // Add to feed
    const ts = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.className = 'feed-message feed-message-patient';
    div.innerHTML = `
        <div class="msg-header">
            <span class="msg-sender" style="color:var(--primary-color);">Patient (ISL)</span>
            <span class="msg-time">${ts}</span>
        </div>
        <div class="msg-body">${data.text}</div>
        <div class="msg-confidence">Confidence: ${Math.round(data.confidence * 100)}%</div>
    `;
    translationFeed.appendChild(div);
    translationFeed.scrollTop = translationFeed.scrollHeight;

    // ISL toast on video
    showIslToast(data.text);

    // Play audio
    if (data.audio) {
        try {
            audioPlayer.src = 'data:audio/mp3;base64,' + data.audio;
            audioPlayer.play().catch(() => { });
        } catch (e) { /* ignore */ }
    }

    // Auto-append to notes
    consultationNotes.value += `[${ts}] Patient: ${data.text}\n`;

    showNotification('Patient signed', data.text);
});

function showIslToast(text) {
    const toast = document.createElement('div');
    toast.className = 'isl-toast';
    toast.innerHTML = `<span class="toast-icon"><i class="fas fa-hand-paper"></i></span>${text}`;
    islToastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('fading');
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ── Doctor messaging ─────────────────────────────────

function sendDoctorMessage() {
    const msg = doctorMessage.value.trim();
    if (!msg) return;

    socket.emit('doctor_message', {
        text: msg,
        doctor_name: 'Doctor',
        patient_room: 'doctor_room',
    });

    // Add to feed
    if (feedPlaceholder) feedPlaceholder.remove();

    const ts = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.className = 'feed-message feed-message-doctor';
    div.innerHTML = `
        <div class="msg-header">
            <span class="msg-sender" style="color:var(--success-color);">You (Doctor)</span>
            <span class="msg-time">${ts}</span>
        </div>
        <div class="msg-body">${msg}</div>
    `;
    translationFeed.appendChild(div);
    translationFeed.scrollTop = translationFeed.scrollHeight;

    consultationNotes.value += `[${ts}] Doctor: ${msg}\n`;
    doctorMessage.value = '';
}

sendMessageBtn.addEventListener('click', sendDoctorMessage);
doctorMessage.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendDoctorMessage();
});

// Quick phrases
document.querySelectorAll('.phrase-pill').forEach(pill => {
    pill.addEventListener('click', () => {
        doctorMessage.value = pill.dataset.phrase;
        doctorMessage.focus();
    });
});

// ── Notes ────────────────────────────────────────────

saveNotesBtn.addEventListener('click', () => {
    const notes = consultationNotes.value;
    if (notes.trim()) {
        const ts = new Date().toISOString();
        localStorage.setItem('consultation_' + ts, JSON.stringify({
            timestamp: ts, notes, messageCount: messageCounter,
        }));
        showAlert('Notes saved!', 'success');
    }
});

exportNotesBtn.addEventListener('click', () => {
    const blob = new Blob([JSON.stringify({
        timestamp: new Date().toISOString(),
        notes: consultationNotes.value,
        messageCount: messageCounter,
    }, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `session_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
});

// ── Utilities ────────────────────────────────────────

function showAlert(message, type) {
    const div = document.createElement('div');
    div.className = `alert alert-${type} alert-dismissible fade show`;
    div.innerHTML = `<i class="fas fa-check-circle me-2"></i>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
    const container = document.querySelector('.main-content .container-fluid');
    container.insertBefore(div, container.firstChild);
    setTimeout(() => div.remove(), 3000);
}

function showNotification(title, body) {
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(title, { body, icon: '/static/favicon.ico' });
    }
}

// Cleanup
window.addEventListener('beforeunload', () => {
    rtc.endCall('patient_room');
    stopTimer();
});