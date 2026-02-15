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

// --- Sign Recognition Heuristics ---
// Simple rule-based recognition for demonstration
// In a production app, you would use a TensorFlow.js model here.
function recognizeSign(landmarks) {
    // Basic finger state detection
    // landmarks[i] : x, y, z (normalized)

    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const middleTip = landmarks[12];
    const ringTip = landmarks[16];
    const pinkyTip = landmarks[20];

    const indexBase = landmarks[5];
    const middleBase = landmarks[9];
    const ringBase = landmarks[13];
    const pinkyBase = landmarks[17];

    // Check if fingers are extended (y-coordinate is smaller than base)
    const isIndexUp = indexTip.y < indexBase.y;
    const isMiddleUp = middleTip.y < middleBase.y;
    const isRingUp = ringTip.y < ringBase.y;
    const isPinkyUp = pinkyTip.y < pinkyBase.y;

    // A: All fingers folded, thumb on side
    if (!isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) return "A";

    // B: All fingers up, thumb folded
    if (isIndexUp && isMiddleUp && isRingUp && isPinkyUp) return "B";

    // C: Curved hand (approximate)
    if (indexTip.x > thumbTip.x && Math.abs(indexTip.y - thumbTip.y) > 0.1) {
        // Simple C logic
    }

    // L: Thumb and Index up
    if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp && thumbTip.x < indexBase.x) return "L";

    // V: Index and Middle up
    if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) return "V";

    // 0: All fingers down (same as A in this simple logic)
    // 5: All fingers spread (same as B in this simple logic)

    return null;
}

const labels = ['A', 'B', 'L', 'V', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']; // Demo subset

function onResults(results) {
    canvasCtx.save();
    // Mirror the display horizontally
    canvasCtx.translate(canvasElement.width, 0);
    canvasCtx.scale(-1, 1);

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#4f46e5', lineWidth: 5 });
            drawLandmarks(canvasCtx, landmarks, { color: '#10b981', lineWidth: 2 });

            const predicted = recognizeSign(landmarks);
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
document.getElementById('speakBtn').addEventListener('click', () => {
    const text = textBox.innerText;
    if (text && text !== "Waiting..." && !text.includes("Error")) {
        let speech = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(speech);
    }
});

document.getElementById('clearBtn').addEventListener('click', () => {
    outputText = "";
    textBox.innerText = "Waiting...";
});
