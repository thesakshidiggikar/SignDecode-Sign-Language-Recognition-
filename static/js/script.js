// --- Smooth scroll link handler ---
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// --- Mobile Menu Toggle ---
const mobileMenu = document.getElementById('mobile-menu');
const navList = document.getElementById('nav-list');

if (mobileMenu && navList) {
    mobileMenu.addEventListener('click', () => {
        navList.classList.toggle('active');
        mobileMenu.querySelector('i').classList.toggle('fa-bars');
        mobileMenu.querySelector('i').classList.toggle('fa-times');
    });

    // Close menu when a link is clicked
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => {
            navList.classList.remove('active');
            mobileMenu.querySelector('i').classList.add('fa-bars');
            mobileMenu.querySelector('i').classList.remove('fa-times');
        });
    });
}

// --- State Variables ---
let gameScore = 0;
let currentTarget = null;
let gameActive = false;
let outputText = "";
let lastPredictedLabel = null;
let frameCount = 0;
const THRESHOLD_FRAMES = 15;

const textBox = document.getElementById('recognizedText');
const scoreEl = document.getElementById('gameScore');
const targetEl = document.getElementById('targetSign');
const feedbackEl = document.getElementById('gameFeedback');
const speedRange = document.getElementById('speedRange');
const speedValue = document.getElementById('speedValue');
const translatorInput = document.getElementById('translatorInput');
const translateBtn = document.getElementById('runTranslateBtn');
const visualizerOutput = document.getElementById('visualizerOutput');
const visualizerChar = document.getElementById('visualizerChar');

// --- Sign Mapping (A-Z) ---
const signIcons = {
    'A': '👊', 'B': '✋', 'C': '👌', 'D': '☝️', 'E': '✊',
    'F': '👌', 'G': '👈', 'H': '👈', 'I': '☝️', 'J': '☝️',
    'K': '✌️', 'L': '👆', 'M': '✋', 'N': '✋', 'O': '👌',
    'P': '☝️', 'Q': '👈', 'R': '✌️', 'S': '👊', 'T': '👊',
    'U': '✌️', 'V': '✌️', 'W': '🖖', 'X': '☝️', 'Y': '🤙', 'Z': '☝️'
};

// --- Populate Sign Atlas ---
const atlasGrid = document.getElementById('signAtlas');
if (atlasGrid) {
    atlasGrid.innerHTML = ""; // Clear existing
    Object.entries(signIcons).forEach(([char, icon]) => {
        const item = document.createElement('div');
        item.className = 'atlas-item';
        item.innerHTML = `
            <span class="atlas-char">${char}</span>
            <span class="atlas-sign">${icon}</span>
        `;
        // Make atlas items interactive
        item.addEventListener('click', () => {
            if (visualizerOutput) {
                visualizerOutput.innerText = icon;
                visualizerChar.innerText = char;
            }
        });
        atlasGrid.appendChild(item);
    });
}

// --- Speed Control ---
if (speedRange) {
    speedRange.addEventListener('input', () => {
        speedValue.innerText = speedRange.value + "x";
    });
}

// --- Text-to-Sign Translator (Sequential) ---
let translationInterval = null;

if (translateBtn) {
    translateBtn.addEventListener('click', () => {
        const text = translatorInput.value.toUpperCase().replace(/[^A-Z ]/g, "");
        if (!text) return;

        // Reset and start sequence
        if (translationInterval) clearInterval(translationInterval);

        let index = 0;
        translateBtn.disabled = true;
        translateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';

        translationInterval = setInterval(() => {
            if (index >= text.length) {
                clearInterval(translationInterval);
                visualizerChar.innerText = "Done";
                translateBtn.disabled = false;
                translateBtn.innerHTML = '<i class="fas fa-play"></i> Run Sequence';
                return;
            }

            const char = text[index];
            if (signIcons[char]) {
                visualizerOutput.innerText = signIcons[char];
                visualizerChar.innerText = char;
            } else if (char === " ") {
                visualizerOutput.innerText = "⏳";
                visualizerChar.innerText = "Space";
            } else {
                // Skip unknown
                index++;
                return;
            }

            index++;
        }, 800); // 800ms for readable speed
    });
}

// --- MediaPipe Setup ---
const videoElement = document.getElementById('input_video');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({ image: videoElement });
    },
    width: 640,
    height: 480
});
camera.start();

/**
 * Heuristic Helper for recognition
 * @param {LandmarkList} landmarks 
 * @param {string} handedness 'Left' or 'Right'
 */
function recognizeSign(landmarks, handedness) {
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const middleTip = landmarks[12];
    const ringTip = landmarks[16];
    const pinkyTip = landmarks[20];

    const thumbBase = landmarks[2];

    const isIndexUp = indexTip.y < landmarks[6].y;
    const isMiddleUp = middleTip.y < landmarks[10].y;
    const isRingUp = ringTip.y < landmarks[14].y;
    const isPinkyUp = pinkyTip.y < landmarks[18].y;

    // A: All fingers folded, thumb on side
    if (!isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) return "A";

    // B: All fingers up, thumb folded
    if (isIndexUp && isMiddleUp && isRingUp && isPinkyUp) return "B";

    // L: Thumb and Index up
    const thumbIsOurHand = handedness === 'Left' ? thumbTip.x > thumbBase.x : thumbTip.x < thumbBase.x;
    if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && thumbIsOurHand) return "L";

    // V: Index and Middle up
    if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) return "V";

    return null;
}

const labels = ['A', 'B', 'L', 'V']; // Demo subset for game

function onResults(results) {
    canvasCtx.save();

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks && results.multiHandedness) {
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const landmarks = results.multiHandLandmarks[i];
            const handedness = results.multiHandedness[i].label; // 'Left' or 'Right'

            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#4f46e5', lineWidth: 5 });
            drawLandmarks(canvasCtx, landmarks, { color: '#10b981', lineWidth: 2 });

            const predicted = recognizeSign(landmarks, handedness);
            if (predicted) {
                // Stability logic
                if (predicted === lastPredictedLabel) {
                    frameCount++;
                } else {
                    frameCount = 0;
                    lastPredictedLabel = predicted;
                }

                if (frameCount >= THRESHOLD_FRAMES) {
                    outputText += predicted;
                    textBox.innerText = outputText;
                    frameCount = 0;

                    // Game Check
                    if (gameActive && predicted === currentTarget) {
                        celebrateMatch();
                    }
                }
            }
        }
    } else {
        if (outputText === "") textBox.innerText = "Waiting...";
    }
    canvasCtx.restore();
}

// --- Game Functions ---
function getNewTarget() {
    feedbackEl.innerText = "Processing...";
    const randomIndex = Math.floor(Math.random() * labels.length);
    currentTarget = labels[randomIndex];
    targetEl.innerText = currentTarget;
    feedbackEl.innerText = "Try to sign: " + currentTarget;
}

function celebrateMatch() {
    confetti({
        particleCount: 150,
        spread: 70,
        origin: { y: 0.6 },
        colors: ['#4f46e5', '#7d2ae8', '#10b981']
    });
    gameScore++;
    scoreEl.innerText = gameScore;
    feedbackEl.innerText = "Great job! You got it!";

    // Pause briefly before next sign
    setTimeout(() => {
        if (gameActive) getNewTarget();
    }, 2000);
}

document.getElementById('startGameBtn').addEventListener('click', () => {
    gameActive = true;
    gameScore = 0;
    scoreEl.innerText = "0";
    getNewTarget();
    document.getElementById('game').scrollIntoView({ behavior: 'smooth' });
});

// --- Controls ---
if (document.getElementById('speakBtn')) {
    document.getElementById('speakBtn').addEventListener('click', () => {
        const text = textBox.innerText;
        if (text && text !== "Waiting..." && !text.includes("Error")) {
            let speech = new SpeechSynthesisUtterance(text);
            speech.rate = parseFloat(speedRange.value);
            window.speechSynthesis.speak(speech);
        }
    });
}

if (document.getElementById('spaceBtn')) {
    document.getElementById('spaceBtn').addEventListener('click', () => {
        if (outputText !== "") {
            outputText += " ";
            textBox.innerText = outputText;
        }
    });
}

if (document.getElementById('clearBtn')) {
    document.getElementById('clearBtn').addEventListener('click', () => {
        outputText = "";
        textBox.innerText = "Waiting...";
    });
}
