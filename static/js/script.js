// Smooth scroll link handler
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// --- State Variables ---
let gameScore = 0;
let currentTarget = null;
let gameActive = false;
const textBox = document.getElementById('recognizedText');
const scoreEl = document.getElementById('gameScore');
const targetEl = document.getElementById('targetSign');
const feedbackEl = document.getElementById('gameFeedback');

// --- Polling (Studio & Game) ---
function updateApp() {
    // 1. Fetch live recognition
    fetch('/status')
        .then(res => res.json())
        .then(data => {
            if (data.text) textBox.innerText = data.text;
            else textBox.innerText = "Waiting...";
        });

    // 2. Game logic (if active)
    if (gameActive) {
        fetch('/game_status')
            .then(res => res.json())
            .then(data => {
                if (data.match) {
                    celebrateMatch();
                    gameScore = data.score;
                    scoreEl.innerText = gameScore;
                    getNewTarget();
                }
            });
    }
}

setInterval(updateApp, 800);

// --- Game Functions ---
function getNewTarget() {
    feedbackEl.innerText = "Processing...";
    fetch('/get_new_sign')
        .then(res => res.json())
        .then(data => {
            currentTarget = data.sign;
            targetEl.innerText = currentTarget;
            feedbackEl.innerText = "Try to sign: " + currentTarget;
        });
}

function celebrateMatch() {
    confetti({
        particleCount: 150,
        spread: 70,
        origin: { y: 0.6 },
        colors: ['#4f46e5', '#7d2ae8', '#10b981']
    });
    feedbackEl.innerText = "Greak job! You got it!";
    setTimeout(() => {
        if (gameActive) feedbackEl.innerText = "Try to sign: " + currentTarget;
    }, 2000);
}

document.getElementById('startGameBtn').addEventListener('click', () => {
    gameActive = true;
    gameScore = 0;
    scoreEl.innerText = "0";
    getNewTarget();
    document.getElementById('game').scrollIntoView({ behavior: 'smooth' });
});

// --- Existing Controls ---
document.getElementById('speakBtn').addEventListener('click', () => {
    const text = textBox.innerText;
    if (text && text !== "Waiting...") {
        let speech = new SpeechSynthesisUtterance(text);
        window.speechSynthesis.speak(speech);
    }
});

document.getElementById('clearBtn').addEventListener('click', () => {
    fetch('/clear_text', { method: 'POST' }).then(() => {
        textBox.innerText = "Waiting...";
    });
});

// Smooth Scroll
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({ behavior: 'smooth' });
    });
});
