/**
 * Deep Learning Voice Recognition Interface
 * JavaScript Functionality
 */

// ============================================
// DOM Elements
// ============================================
const elements = {
    // Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    uploadTab: document.getElementById('uploadTab'),
    recordTab: document.getElementById('recordTab'),
    
    // Upload
    dropZone: document.getElementById('dropZone'),
    audioInput: document.getElementById('audioInput'),
    
    // Recording
    recordBtn: document.getElementById('recordBtn'),
    recordingTimer: document.getElementById('recordingTimer'),
    recordingIndicator: document.getElementById('recordingIndicator'),
    
    // Audio Player
    audioPlayerSection: document.getElementById('audioPlayerSection'),
    fileName: document.getElementById('fileName'),
    playPauseBtn: document.getElementById('playPauseBtn'),
    progressBar: document.getElementById('progressBar'),
    currentTime: document.getElementById('currentTime'),
    duration: document.getElementById('duration'),
    
    // Submit
    submitSection: document.getElementById('submitSection'),
    submitBtn: document.getElementById('submitBtn'),
    
    // Loading
    loadingSection: document.getElementById('loadingSection'),
    
    // Prediction
    predictionSection: document.getElementById('predictionSection'),
    // Modèle 1 (CNN)
    predictedCommand1: document.getElementById('predictedCommand1'),
    confidenceFill1: document.getElementById('confidenceFill1'),
    confidenceValue1: document.getElementById('confidenceValue1'),
    // Modèle 2 (LSTM)
    predictedCommand2: document.getElementById('predictedCommand2'),
    confidenceFill2: document.getElementById('confidenceFill2'),
    confidenceValue2: document.getElementById('confidenceValue2'),
    otherTags: document.getElementById('otherTags'),
    otherTags1: document.getElementById('otherTags1'),
    otherTags2: document.getElementById('otherTags2'),

    // Waveform
    waveform: document.getElementById('waveform'),
    waveformOverlay: document.getElementById('waveformOverlay')
};

// ============================================
// State Variables
// ============================================
let audioFile = null;
let audioElement = null;
let audioContext = null;
let analyser = null;
let animationId = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordingStartTime = null;
let recordingTimerInterval = null;
let isRecording = false;
let isPlaying = false;

// ============================================
// Initialize
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initializeWaveform();
    setupEventListeners();
});

// ============================================
// Waveform Visualization
// ============================================
function initializeWaveform() {
    const canvas = elements.waveform;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    // Draw initial state
    drawWaveform(ctx, canvas.offsetWidth, canvas.offsetHeight, []);
}

function drawWaveform(ctx, width, height, data) {
    ctx.clearRect(0, 0, width, height);
    
    if (data.length === 0) {
        // Draw placeholder bars
        const barCount = 50;
        const barWidth = width / barCount;
        const gap = 2;
        
        for (let i = 0; i < barCount; i++) {
            const barHeight = Math.random() * 20 + 10;
            const x = i * barWidth;
            const y = (height - barHeight) / 2;
            
            const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
            gradient.addColorStop(0, '#667eea');
            gradient.addColorStop(0.5, '#764ba2');
            gradient.addColorStop(1, '#4facfe');
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.roundRect(x + gap / 2, y, barWidth - gap, barHeight, 2);
            ctx.fill();
        }
    } else {
        // Draw real audio data
        const barCount = data.length;
        const barWidth = width / barCount;
        const gap = 2;
        
        for (let i = 0; i < barCount; i++) {
            const barHeight = Math.abs(data[i]) * height * 0.8;
            const x = i * barWidth;
            const y = (height - barHeight) / 2;
            
            const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
            gradient.addColorStop(0, '#43e97b');
            gradient.addColorStop(0.5, '#38f9d7');
            gradient.addColorStop(1, '#4facfe');
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.roundRect(x + gap / 2, y, barWidth - gap, Math.max(barHeight, 2), 2);
            ctx.fill();
        }
    }
}

function startWaveformAnimation() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    if (!analyser) {
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
    }
    
    const source = audioContext.createMediaElementSource(audioElement);
    source.connect(analyser);
    analyser.connect(audioContext.destination);
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    elements.waveformOverlay.classList.add('active');
    
    function animate() {
        animationId = requestAnimationFrame(animate);
        analyser.getByteFrequencyData(dataArray);
        
        const canvas = elements.waveform;
        const ctx = canvas.getContext('2d');
        drawWaveform(ctx, canvas.offsetWidth, canvas.offsetHeight, Array.from(dataArray));
    }
    
    animate();
}

function stopWaveformAnimation() {
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
    
    elements.waveformOverlay.classList.remove('active');
    
    // Reset waveform to placeholder
    const canvas = elements.waveform;
    const ctx = canvas.getContext('2d');
    setTimeout(() => {
        drawWaveform(ctx, canvas.offsetWidth, canvas.offsetHeight, []);
    }, 300);
}

// ============================================
// Event Listeners
// ============================================
function setupEventListeners() {
    // Tab switching
    elements.tabBtns.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });
    
    // Drag and drop
    elements.dropZone.addEventListener('click', () => elements.audioInput.click());
    elements.dropZone.addEventListener('dragover', handleDragOver);
    elements.dropZone.addEventListener('dragleave', handleDragLeave);
    elements.dropZone.addEventListener('drop', handleDrop);
    elements.audioInput.addEventListener('change', handleFileSelect);
    
    // Recording
    elements.recordBtn.addEventListener('click', toggleRecording);
    
    // Audio player - On utilise l'input range pour le seek
    elements.progressBar.addEventListener('input', seekAudio);
    elements.playPauseBtn.addEventListener('click', togglePlayPause);
    
    // Submit - ON FORCE LE PREVENT DEFAULT ICI
    elements.submitBtn.addEventListener('click', function(e) {
        e.preventDefault();
        submitAudio(e);
    });
}

// ============================================
// Tab Switching
// ============================================
function switchTab(tabName) {
    elements.tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    
    elements.uploadTab.classList.toggle('active', tabName === 'upload');
    elements.recordTab.classList.toggle('active', tabName === 'record');
    
    // Stop any active recording when switching tabs
    if (tabName === 'upload' && isRecording) {
        stopRecording();
    }
}

// ============================================
// File Upload - Drag & Drop
// ============================================
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.dropZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.dropZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.dropZone.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'audio/ogg', 'audio/m4a', 'audio/x-m4a'];
    const validExtensions = ['.wav', '.mp3', '.ogg', '.m4a'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(file.type) && !validExtensions.includes(fileExtension)) {
        alert('Please upload a valid audio file (.wav, .mp3, .ogg, .m4a)');
        return;
    }
    
    audioFile = file;
    
    // Create audio element
    if (audioElement) {
        audioElement.pause();
        audioElement = null;
    }
    
    audioElement = new Audio(URL.createObjectURL(file));
    audioElement.addEventListener('loadedmetadata', updateDuration);
    audioElement.addEventListener('timeupdate', updateProgress);
    audioElement.addEventListener('ended', handleAudioEnded);
    
    // Update UI
    elements.fileName.textContent = file.name;
    elements.audioPlayerSection.classList.add('active');
    elements.submitSection.classList.add('active');
    
    // Reset prediction
    hidePrediction();
    
    // Start waveform visualization
    startWaveformAnimation();
}

// ============================================
// Recording Functionality
// ============================================
async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        recordedChunks = [];
        
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) {
                recordedChunks.push(e.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'audio/webm' });
            const file = new File([blob], 'recorded_audio.webm', { type: 'audio/webm' });
            handleRecordedAudio(file);
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start(100);
        isRecording = true;
        
        // Update UI
        elements.recordBtn.classList.add('recording');
        elements.recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
        elements.recordingIndicator.classList.add('active');
        
        // Start timer
        recordingStartTime = Date.now();
        recordingTimerInterval = setInterval(updateRecordingTimer, 100);
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        alert('Unable to access microphone. Please check permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    
    isRecording = false;
    
    // Update UI
    elements.recordBtn.classList.remove('recording');
    elements.recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    elements.recordingIndicator.classList.remove('active');
    
    // Stop timer
    if (recordingTimerInterval) {
        clearInterval(recordingTimerInterval);
        recordingTimerInterval = null;
    }
}

function updateRecordingTimer() {
    const elapsed = Date.now() - recordingStartTime;
    const seconds = Math.floor((elapsed / 1000) % 60);
    const minutes = Math.floor((elapsed / 1000 / 60) % 60);
    elements.recordingTimer.textContent = 
        `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

function handleRecordedAudio(file) {
    audioFile = file;
    
    // Create audio element
    if (audioElement) {
        audioElement.pause();
        audioElement = null;
    }
    
    audioElement = new Audio(URL.createObjectURL(file));
    audioElement.addEventListener('loadedmetadata', updateDuration);
    audioElement.addEventListener('timeupdate', updateProgress);
    audioElement.addEventListener('ended', handleAudioEnded);
    
    // Update UI
    elements.fileName.textContent = 'Recorded Audio';
    elements.audioPlayerSection.classList.add('active');
    elements.submitSection.classList.add('active');
    
    // Reset prediction
    hidePrediction();
    
    // Start waveform visualization
    startWaveformAnimation();
}

// ============================================
// Audio Player Controls
// ============================================
function togglePlayPause() {
    if (!audioElement) return;
    
    if (isPlaying) {
        audioElement.pause();
        elements.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
    } else {
        audioElement.play();
        elements.playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
    }
    
    isPlaying = !isPlaying;
}

function seekAudio() {
    if (!audioElement) return;
    const time = (elements.progressBar.value / 100) * audioElement.duration;
    audioElement.currentTime = time;
}

function updateProgress() {
    if (!audioElement) return;
    const progress = (audioElement.currentTime / audioElement.duration) * 100;
    elements.progressBar.value = progress;
    elements.currentTime.textContent = formatTime(audioElement.currentTime);
}

function updateDuration() {
    if (!audioElement) return;
    elements.duration.textContent = formatTime(audioElement.duration);
}

function handleAudioEnded() {
    isPlaying = false;
    elements.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
    elements.progressBar.value = 0;
    elements.currentTime.textContent = '0:00';
}

function formatTime(seconds) {
    if (isNaN(seconds)) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ============================================
// Submit & API Integration
// ============================================
async function submitAudio(e) {
    if (e) {
        e.preventDefault();
        e.stopPropagation();
    }

    if (!audioFile) {
        alert('Please upload or record audio first');
        return false;
    }

    console.log("Envoi du fichier audio:", audioFile);

    try {
        if (elements.submitSection) elements.submitSection.classList.remove('active');
        if (elements.loadingSection) elements.loadingSection.classList.add('active');

        const formData = new FormData();
        formData.append("audio", audioFile, "recording.wav");

        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Erreur serveur : " + response.status);

        const data = await response.json();
        console.log("Données reçues du serveur:", data);

        displayPredictions(data);

    } catch (error) {
        console.error("Erreur détectée :", error);
        alert("Erreur lors de la prédiction : " + error.message);

        if (elements.loadingSection) elements.loadingSection.classList.remove('active');
        if (elements.submitSection) elements.submitSection.classList.add('active');
    }

    return false;
}

// Simulate prediction for demo
function simulatePrediction() {
    const commands = [
        { command: 'Turn on the lights', confidence: 0.94 },
        { command: 'Turn off the lights', confidence: 0.89 },
        { command: 'Play music', confidence: 0.85 },
        { command: 'Stop playing', confidence: 0.82 },
        { command: 'Increase volume', confidence: 0.78 }
    ];
    
    const mainResult = commands[0];
    const otherResults = commands.slice(1);
    
    setTimeout(() => {
        displayPrediction({
            predicted_command: mainResult.command,
            confidence: mainResult.confidence,
            other_predictions: otherResults
        });
    }, 2000);
}

function displayPredictions(data) {
    try {
        // --- MODÈLE 1 (CNN) ---
        if (elements.predictedCommand1 && data.cnn) {
            // Correction : utiliser predicted_command au lieu de command
            elements.predictedCommand1.textContent = data.cnn.predicted_command;
            const conf1 = Math.round(data.cnn.confidence * 100);
            elements.confidenceFill1.style.width = conf1 + "%";
            elements.confidenceValue1.textContent = conf1 + "%";
            
            // Correction : utiliser other_predictions au lieu de others
            const others1 = data.cnn.other_predictions || [];
            elements.otherTags1.innerHTML = others1
                .map(p => `<li>${p.command} <span class="tag-conf">${Math.round(p.confidence * 100)}%</span></li>`)
                .join('');
        }

        // --- MODÈLE 2 (LSTM) ---
        if (elements.predictedCommand2 && data.lstm) {
            // Correction : utiliser predicted_command au lieu de command
            elements.predictedCommand2.textContent = data.lstm.predicted_command; 
            const conf2 = Math.round(data.lstm.confidence * 100);
            elements.confidenceFill2.style.width = conf2 + "%";
            elements.confidenceValue2.textContent = conf2 + "%";

            // Correction : utiliser other_predictions au lieu de others
            const others2 = data.lstm.other_predictions || [];
            elements.otherTags2.innerHTML = others2
                .map(p => `<li>${p.command} <span class="tag-conf">${Math.round(p.confidence * 100)}%</span></li>`)
                .join('');
        }

        if (elements.loadingSection) elements.loadingSection.classList.remove('active');
        if (elements.predictionSection) elements.predictionSection.classList.add('active');

    } catch (error) {
        console.error("Erreur UI :", error);
    }
}
function hidePrediction() {
    elements.predictionSection.classList.remove('active');

    // Reset Model 1
    if (elements.predictedCommand1) {
        elements.predictedCommand1.innerHTML = '<span class="placeholder">Waiting for input...</span>';
        elements.confidenceFill1.style.width = '0%';
        elements.confidenceValue1.textContent = '0%';
    }

    // Reset Model 2
    if (elements.predictedCommand2) {
        elements.predictedCommand2.innerHTML = '<span class="placeholder">Waiting for input...</span>';
        elements.confidenceFill2.style.width = '0%';
        elements.confidenceValue2.textContent = '0%';
    }
}

// ============================================
// Window Resize Handler
// ============================================
window.addEventListener('resize', () => {
    const canvas = elements.waveform;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    if (!audioElement || audioElement.paused) {
        drawWaveform(ctx, canvas.offsetWidth, canvas.offsetHeight, []);
    }
});

