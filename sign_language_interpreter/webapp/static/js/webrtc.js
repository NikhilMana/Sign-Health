/**
 * SignHealthRTC — WebRTC peer connection manager for telehealth video calls.
 *
 * Handles media acquisition, SDP negotiation, ICE candidate exchange,
 * and call lifecycle. Works with Socket.IO for signaling.
 *
 * Usage:
 *   const rtc = new SignHealthRTC(socket, { role: 'doctor', localVideo, remoteVideo });
 *   await rtc.startLocalStream();
 *   rtc.initiateCall();   // doctor side
 */

class SignHealthRTC {
    constructor(socket, options = {}) {
        this.socket = socket;
        this.role = options.role || 'patient';               // 'doctor' or 'patient'
        this.localVideoEl = options.localVideo || null;
        this.remoteVideoEl = options.remoteVideo || null;
        this.onRemoteStream = options.onRemoteStream || null;
        this.onCallStateChange = options.onCallStateChange || null;
        this.onConnectionStateChange = options.onConnectionStateChange || null;

        this.peerConnection = null;
        this.localStream = null;
        this.remoteStream = null;
        this.isMuted = false;
        this.isCameraOff = false;
        this.callActive = false;

        // ICE servers — Google's public STUN + optional TURN
        this.iceServers = [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' },
            { urls: 'stun:stun2.l.google.com:19302' },
        ];

        this._bindSocketEvents();
    }

    // ── Media ────────────────────────────────────────────

    async startLocalStream(videoConstraints = { width: 1280, height: 720 }) {
        try {
            this.localStream = await navigator.mediaDevices.getUserMedia({
                video: videoConstraints,
                audio: true,
            });

            if (this.localVideoEl) {
                this.localVideoEl.srcObject = this.localStream;
            }

            this._emitState('local_stream_ready');
            return this.localStream;
        } catch (err) {
            console.error('[RTC] Media access error:', err);
            this._emitState('media_error');
            throw err;
        }
    }

    async startAudioOnlyStream() {
        try {
            this.localStream = await navigator.mediaDevices.getUserMedia({
                video: false,
                audio: true,
            });
            this._emitState('local_stream_ready');
            return this.localStream;
        } catch (err) {
            console.error('[RTC] Audio access error:', err);
            this._emitState('media_error');
            throw err;
        }
    }

    stopLocalStream() {
        if (this.localStream) {
            this.localStream.getTracks().forEach(t => t.stop());
            this.localStream = null;
        }
        if (this.localVideoEl) this.localVideoEl.srcObject = null;
    }

    // ── Call lifecycle ────────────────────────────────────

    async initiateCall(patientRoom) {
        this._createPeerConnection();
        this._addLocalTracks();

        const offer = await this.peerConnection.createOffer();
        await this.peerConnection.setLocalDescription(offer);

        this.socket.emit('webrtc_offer', {
            sdp: this.peerConnection.localDescription,
            patient_room: patientRoom,
        });

        this.callActive = true;
        this._emitState('calling');
    }

    async _handleOffer(sdp) {
        this._createPeerConnection();
        this._addLocalTracks();

        await this.peerConnection.setRemoteDescription(new RTCSessionDescription(sdp));
        const answer = await this.peerConnection.createAnswer();
        await this.peerConnection.setLocalDescription(answer);

        this.socket.emit('webrtc_answer', {
            sdp: this.peerConnection.localDescription,
            doctor_room: 'doctor_room',
        });

        this.callActive = true;
        this._emitState('in_call');
    }

    async _handleAnswer(sdp) {
        await this.peerConnection.setRemoteDescription(new RTCSessionDescription(sdp));
        this.callActive = true;
        this._emitState('in_call');
    }

    endCall(targetRoom) {
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = null;
        }
        this.remoteStream = null;
        if (this.remoteVideoEl) this.remoteVideoEl.srcObject = null;

        this.callActive = false;
        this._emitState('call_ended');

        if (targetRoom) {
            this.socket.emit('end_call', { target_room: targetRoom });
        }
    }

    // ── Controls ─────────────────────────────────────────

    toggleMute() {
        if (!this.localStream) return false;
        this.isMuted = !this.isMuted;
        this.localStream.getAudioTracks().forEach(t => { t.enabled = !this.isMuted; });
        return this.isMuted;
    }

    toggleCamera() {
        if (!this.localStream) return false;
        const videoTracks = this.localStream.getVideoTracks();
        if (videoTracks.length === 0) return false;  // audio-only stream
        this.isCameraOff = !this.isCameraOff;
        videoTracks.forEach(t => { t.enabled = !this.isCameraOff; });
        return this.isCameraOff;
    }

    // ── Internal ─────────────────────────────────────────

    _createPeerConnection() {
        if (this.peerConnection) {
            this.peerConnection.close();
        }

        this.peerConnection = new RTCPeerConnection({ iceServers: this.iceServers });

        // ICE candidates → relay via signaling
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate) {
                const targetRoom = this.role === 'doctor' ? 'patient_room' : 'doctor_room';
                this.socket.emit('ice_candidate', {
                    candidate: event.candidate,
                    target_room: targetRoom,
                });
            }
        };

        // Remote stream arrives
        this.peerConnection.ontrack = (event) => {
            this.remoteStream = event.streams[0];
            if (this.remoteVideoEl) {
                this.remoteVideoEl.srcObject = this.remoteStream;
            }
            if (this.onRemoteStream) {
                this.onRemoteStream(this.remoteStream);
            }
            this._emitState('remote_stream_connected');
        };

        // Connection state monitoring
        this.peerConnection.onconnectionstatechange = () => {
            const state = this.peerConnection.connectionState;
            if (this.onConnectionStateChange) {
                this.onConnectionStateChange(state);
            }
            if (state === 'disconnected' || state === 'failed') {
                this._emitState('connection_lost');
            }
        };
    }

    _addLocalTracks() {
        if (this.localStream && this.peerConnection) {
            this.localStream.getTracks().forEach(track => {
                this.peerConnection.addTrack(track, this.localStream);
            });
        }
    }

    _bindSocketEvents() {
        this.socket.on('webrtc_offer', async (data) => {
            if (this.role === 'patient') {
                await this._handleOffer(data.sdp);
            }
        });

        this.socket.on('webrtc_answer', async (data) => {
            if (this.role === 'doctor') {
                await this._handleAnswer(data.sdp);
            }
        });

        this.socket.on('ice_candidate', async (data) => {
            if (this.peerConnection && data.candidate) {
                try {
                    await this.peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
                } catch (err) {
                    console.warn('[RTC] ICE candidate error:', err);
                }
            }
        });

        this.socket.on('call_ended', () => {
            this.endCall(null);
        });
    }

    _emitState(state) {
        if (this.onCallStateChange) {
            this.onCallStateChange(state);
        }
    }
}
