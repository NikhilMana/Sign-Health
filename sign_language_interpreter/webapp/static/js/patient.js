/**
 * Patient Dashboard — WebRTC video call + ISL sign detection + frame streaming.
 *
 * Dual pipeline:
 *   1. WebRTC peer connection → live video to doctor
 *   2. Base64 frame capture  → server for ISL ML detection
 */

const socket = io();

// ── DOM refs ─────────────────────────────────────────
const remoteVideo = document.getElementById('remoteVideo');
const localVideo = document.getElementById('localVideo');
const videoPlaceholder = document.getElementById('videoPlaceholder');
const doctorOverlay = document.getElementById('doctorOverlay');
const sessionTimerEl = document.getElementById('sessionTimerOverlay');
const sessionTimeEl = document.getElementById('sessionTime');
const pipContainer = document.getElementById('pipContainer');
const callControls = document.getElementById('callControls');
const callStatusBar = document.getElementById('callStatusBar');

const toggleMicBtn = document.getElementById('toggleMicBtn');
const toggleCamBtn = document.getElementById('toggleCamBtn');
const endCallBtn = document.getElementById('endCallBtn');

const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const currentSign = document.getElementById('currentSign');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const detectionBadge = document.getElementById('detectionBadge');
const pipDetection = document.getElementById('pipDetection');
const pipSignText = document.getElementById('pipSignText');

const conversationHistory = document.getElementById('conversationHistory');
const convPlaceholder = document.getElementById('convPlaceholder');
const connectionDot = document.getElementById('connectionDot');
const connectionLabel = document.getElementById('connectionLabel');
const audioPlayer = document.getElementById('audioPlayer');

// ── State ────────────────────────────────────────────
let isDetecting = false;
let frameInterval = null;
let sessionStart = null;
let timerInterval = null;

// ── WebRTC setup ─────────────────────────────────────
const rtc = new SignHealthRTC(socket, {
    role: 'patient',
    localVideo: localVideo,
    remoteVideo: remoteVideo,
    onCallStateChange: handleCallState,
    onConnectionStateChange: handleConnectionState,
});

// ── Initialise ───────────────────────────────────────
window.addEventListener('load', () => {
    socket.emit('join_session', { room: 'patient_room' });

    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
});

// ── Incoming call from doctor ────────────────────────

socket.on('incoming_call', async (data) => {
    try {
        setCallStatus('connecting', 'Doctor is calling…');
        await rtc.startLocalStream({ width: 640, height: 480 });
        showCallUI();

        // Accept call — server tells doctor to send offer
        socket.emit('call_accepted', { doctor_room: 'doctor_room' });
    } catch (err) {
        console.error('Camera access error:', err);
        setCallStatus('disconnected', 'Camera access denied');
    }
});

// Doctor accepted → doctor sends offer, patient auto-answers via SignHealthRTC

endCallBtn.addEventListener('click', () => {
    rtc.endCall('doctor_room');
    hideCallUI();
    setCallStatus('disconnected', 'Call ended');
    stopTimer();
    stopISLDetection();
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
        case 'in_call':
            setCallStatus('connected', 'In Call');
            startTimer();
            break;
        case 'remote_stream_connected':
            setCallStatus('connected', 'Doctor connected');
            videoPlaceholder.style.display = 'none';
            doctorOverlay.style.display = 'flex';
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
    pipContainer.style.display = 'block';
    callControls.style.display = 'flex';
    sessionTimerEl.style.display = 'flex';
}

function hideCallUI() {
    videoPlaceholder.style.display = 'flex';
    doctorOverlay.style.display = 'none';
    pipContainer.style.display = 'none';
    callControls.style.display = 'none';
    sessionTimerEl.style.display = 'none';
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

// ── ISL Detection (base64 frame pipeline) ────────────

startBtn.addEventListener('click', async () => {
    // Ensure camera is on (may already be from WebRTC)
    if (!rtc.localStream) {
        try {
            await rtc.startLocalStream({ width: 640, height: 480 });
            pipContainer.style.display = 'block';
        } catch (err) {
            showAlert('Camera access required for sign detection.', 'danger');
            return;
        }
    }

    isDetecting = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    detectionBadge.textContent = 'Active';
    detectionBadge.className = 'badge bg-success ms-auto';
    pipDetection.style.display = 'block';

    socket.emit('join_session', { room: 'doctor_room' });

    // Capture frames at 5 FPS for ML detection
    frameInterval = setInterval(() => {
        if (isDetecting && rtc.localStream) {
            captureAndSendFrame();
        }
    }, 200);
});

stopBtn.addEventListener('click', () => {
    stopISLDetection();
});

function stopISLDetection() {
    isDetecting = false;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    detectionBadge.textContent = 'Inactive';
    detectionBadge.className = 'badge bg-secondary ms-auto';
    pipDetection.style.display = 'none';

    currentSign.innerHTML = '<span class="text-muted" style="font-weight:400;font-size:0.9rem;">Stopped</span>';
    confidenceBar.style.width = '0%';
    confidenceText.textContent = '0%';

    if (frameInterval) { clearInterval(frameInterval); frameInterval = null; }
}

function captureAndSendFrame() {
    const video = localVideo;
    if (!video || video.videoWidth === 0) return;

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const frameData = canvas.toDataURL('image/jpeg', 0.5);
    socket.emit('video_frame', {
        frame: frameData,
        patient_id: 'patient_1',
        doctor_room: 'doctor_room',
    });
}

// ── Detection results ────────────────────────────────

socket.on('detection_result', (data) => {
    if (!data.text) return;

    // Update sign display
    currentSign.innerHTML = `<strong style="color:var(--primary-color);">${data.text}</strong>`;

    const conf = Math.round(data.confidence * 100);
    confidenceBar.style.width = conf + '%';
    confidenceText.textContent = conf + '%';
    confidenceBar.className = conf > 80
        ? 'progress-bar bg-success'
        : conf > 60 ? 'progress-bar bg-warning' : 'progress-bar bg-danger';

    // PiP badge
    pipSignText.textContent = data.text;

    addToHistory('You (ISL)', data.text, 'patient');
});

// ── Doctor messages ──────────────────────────────────

socket.on('doctor_response', (data) => {
    addToHistory(data.doctor_name, data.text, 'doctor');
    speakText(data.text);
    showNotification('Doctor says', data.text);
});

function addToHistory(sender, message, type) {
    if (convPlaceholder) convPlaceholder.remove();

    const ts = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.className = `feed-message feed-message-${type}`;
    const senderColor = type === 'patient' ? 'var(--primary-color)' : 'var(--success-color)';
    div.innerHTML = `
        <div class="msg-header">
            <span class="msg-sender" style="color:${senderColor};">${sender}</span>
            <span class="msg-time">${ts}</span>
        </div>
        <div class="msg-body">${message}</div>
    `;
    conversationHistory.appendChild(div);
    conversationHistory.scrollTop = conversationHistory.scrollHeight;
}

// ── TTS ──────────────────────────────────────────────

function speakText(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 0.9; u.pitch = 1; u.volume = 0.8;
        window.speechSynthesis.speak(u);
    }
}

// ── Utilities ────────────────────────────────────────

function showAlert(message, type) {
    const div = document.createElement('div');
    div.className = `alert alert-${type} alert-dismissible fade show`;
    div.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
    const container = document.querySelector('.main-content .container-fluid');
    container.insertBefore(div, container.firstChild);
    setTimeout(() => div.remove(), 5000);
}

function showNotification(title, body) {
    if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(title, { body, icon: '/static/favicon.ico' });
    }
}

// ── Cleanup ──────────────────────────────────────────

window.addEventListener('beforeunload', () => {
    rtc.endCall('doctor_room');
    stopISLDetection();
    stopTimer();
});