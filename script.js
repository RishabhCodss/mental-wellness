// MindSpace - AI Mental Health Platform JavaScript - Frontend Only

// Global state management
const AppState = {
    currentPage: 'chat',
    userMood: 'neutral',
    chatHistory: [],
    testProgress: 0,
    currentQuestion: 0,
    testAnswers: [],
    isLoggedIn: false,
    userProfile: null,
    sessionStartTime: new Date(),
    lastActivityTime: new Date()
};

// Mood system configuration
const MoodSystem = {
    neutral: { color: '#6366f1', text: 'Neutral' },
    positive: { color: '#10b981', text: 'Positive' },
    anxious: { color: '#f59e0b', text: 'Anxious' },
    sad: { color: '#ef4444', text: 'Low' },
    excited: { color: '#8b5cf6', text: 'Energetic' },
    calm: { color: '#06b6d4', text: 'Calm' },
    stressed: { color: '#f97316', text: 'Stressed' },
    angry: { color: '#dc2626', text: 'Frustrated' }
};

// Personality test questions
const PersonalityQuestions = [
    {
        id: 1,
        text: "I often feel energized when I'm around other people",
        trait: "extraversion",
        type: "likert"
    },
    {
        id: 2,
        text: "I tend to worry about things that might go wrong",
        trait: "neuroticism",
        type: "likert"
    },
    {
        id: 3,
        text: "I enjoy trying new experiences and activities",
        trait: "openness",
        type: "likert"
    },
    {
        id: 4,
        text: "I am usually organized and keep things tidy",
        trait: "conscientiousness",
        type: "likert"
    },
    {
        id: 5,
        text: "I generally trust others and believe in their good intentions",
        trait: "agreeableness",
        type: "likert"
    },
    {
        id: 6,
        text: "I prefer to work alone rather than in groups",
        trait: "extraversion",
        type: "likert",
        reverse: true
    },
    {
        id: 7,
        text: "I often feel stressed or overwhelmed by daily tasks",
        trait: "neuroticism",
        type: "likert"
    },
    {
        id: 8,
        text: "I enjoy creative activities like art, music, or writing",
        trait: "openness",
        type: "likert"
    },
    {
        id: 9,
        text: "I always complete tasks on time",
        trait: "conscientiousness",
        type: "likert"
    },
    {
        id: 10,
        text: "I find it easy to forgive others when they make mistakes",
        trait: "agreeableness",
        type: "likert"
    },
    {
        id: 11,
        text: "I enjoy being the center of attention",
        trait: "extraversion",
        type: "likert"
    },
    {
        id: 12,
        text: "I rarely feel blue or depressed",
        trait: "neuroticism",
        type: "likert",
        reverse: true
    }
];

// Keywords for mood detection
const MoodKeywords = {
    positive: ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'love', 'grateful', 'blessed', 'peaceful', 'content', 'proud', 'accomplished'],
    anxious: ['anxious', 'worried', 'nervous', 'panic', 'fear', 'scared', 'frightened', 'overwhelmed', 'stressed', 'tense', 'restless', 'uneasy', 'apprehensive'],
    sad: ['sad', 'depressed', 'down', 'blue', 'upset', 'hurt', 'lonely', 'empty', 'hopeless', 'discouraged', 'disappointed', 'grief', 'loss', 'crying'],
    angry: ['angry', 'mad', 'furious', 'frustrated', 'irritated', 'annoyed', 'rage', 'pissed'],
    calm: ['calm', 'relaxed', 'peaceful', 'serene', 'tranquil', 'centered', 'balanced'],
    excited: ['excited', 'energetic', 'pumped', 'motivated', 'enthusiastic', 'thrilled']
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    updateMoodDisplay();
    loadUserData();
    startActivityTracking();
});

function initializeApp() {
    console.log('Initializing MindSpace application...');
    
    // Set initial page
    showPage('chat');
    
    // Hide typing indicator initially
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }
    
    // Initialize personality test
    initializePersonalityTest();
    
    console.log('MindSpace initialized successfully');
}

function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.getAttribute('data-page');
            showPage(page);
            updateNavigation(page);
            updateActivityTime();
        });
    });
    
    // Chat input handling
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    
    if (chatInput && sendButton) {
        // Auto-resize textarea
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
            
            // Enable/disable send button
            const hasContent = this.value.trim().length > 0;
            sendButton.disabled = !hasContent;
            sendButton.style.opacity = hasContent ? '1' : '0.5';
        });
        
        // Send message on Enter (but not Shift+Enter)
        chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Send button click
        sendButton.addEventListener('click', (e) => {
            e.preventDefault();
            sendMessage();
        });
        
        // Focus handling
        chatInput.addEventListener('focus', updateActivityTime);
    }
    
    // Mobile sidebar handling
    document.addEventListener('click', (e) => {
        const sidebar = document.getElementById('sidebar');
        const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
        
        if (sidebar && !sidebar.contains(e.target) && !mobileMenuBtn?.contains(e.target)) {
            sidebar.classList.remove('mobile-open');
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Alt + C to focus chat input
        if (e.altKey && e.key === 'c') {
            e.preventDefault();
            const chatInput = document.getElementById('chatInput');
            if (chatInput && AppState.currentPage === 'chat') {
                chatInput.focus();
            }
        }
        
        // Escape to close mobile sidebar
        if (e.key === 'Escape') {
            const sidebar = document.getElementById('sidebar');
            if (sidebar?.classList.contains('mobile-open')) {
                sidebar.classList.remove('mobile-open');
            }
        }
    });
}

function startActivityTracking() {
    const activityEvents = ['mousedown', 'mousemove', 'keypress', 'scroll', 'touchstart', 'click'];
    const debouncedUpdateActivity = debounce(updateActivityTime, 1000);
    
    activityEvents.forEach(event => {
        document.addEventListener(event, debouncedUpdateActivity, true);
    });
}

function updateActivityTime() {
    AppState.lastActivityTime = new Date();
}

function showPage(pageId) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show selected page
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
    }
    
    // Update page title
    const titles = {
        chat: 'AI Mental Health Chat',
        test: 'Personality Assessment',
        docs: 'Documentation',
        about: 'About MindSpace',
        faq: 'WHO Guidelines & FAQ'
    };
    
    const pageTitle = document.getElementById('pageTitle');
    if (pageTitle && titles[pageId]) {
        pageTitle.textContent = titles[pageId];
    }
    
    AppState.currentPage = pageId;
    
    // Special handling for personality test page
    if (pageId === 'test') {
        setTimeout(() => displayCurrentQuestion(), 100);
    }
    
    // Close mobile sidebar when navigating
    const sidebar = document.getElementById('sidebar');
    if (sidebar?.classList.contains('mobile-open')) {
        sidebar.classList.remove('mobile-open');
    }
}

function updateNavigation(activePageId) {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    
    const activeNavItem = document.querySelector(`[data-page="${activePageId}"]`);
    if (activeNavItem) {
        activeNavItem.classList.add('active');
    }
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        sidebar.classList.toggle('mobile-open');
        updateActivityTime();
    }
}

function toggleLogin() {
    if (AppState.isLoggedIn) {
        logout();
    } else {
        simulateLogin();
    }
}

function simulateLogin() {
    const username = prompt('Enter username (demo):') || 'Demo User';
    
    AppState.isLoggedIn = true;
    AppState.userProfile = {
        name: username,
        joinDate: new Date().toISOString(),
        loginTime: new Date().toISOString()
    };
    
    updateLoginDisplay();
    saveUserData();
}

function updateLoginDisplay() {
    const loginSection = document.getElementById('loginSection');
    if (!loginSection) return;
    
    if (AppState.isLoggedIn && AppState.userProfile) {
        loginSection.innerHTML = `
            <div class="user-info">
                <div class="user-avatar">👤</div>
                <div class="user-details">
                    <div class="user-name">${AppState.userProfile.name}</div>
                    <button class="logout-btn" onclick="logout()">Logout</button>
                </div>
            </div>
        `;
    } else {
        loginSection.innerHTML = '<button class="login-btn" onclick="toggleLogin()">Login / Sign Up</button>';
    }
}

function logout() {
    if (confirm('Are you sure you want to logout?')) {
        AppState.isLoggedIn = false;
        AppState.userProfile = null;
        updateLoginDisplay();
        saveUserData();
        
        // Clear chat on logout
        clearChat();
    }
}

function clearChat() {
    AppState.chatHistory = [];
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to MindSpace 🤗</h2>
                <p>I'm here to provide a safe space for you to share your thoughts and feelings. How can I support you today?</p>
                
                <div class="suggestions">
                    <div class="suggestion-chip" onclick="sendSuggestion('I\\'m feeling anxious today')">I'm feeling anxious</div>
                    <div class="suggestion-chip" onclick="sendSuggestion('I need coping strategies')">Coping strategies</div>
                    <div class="suggestion-chip" onclick="sendSuggestion('I feel overwhelmed')">Feeling overwhelmed</div>
                    <div class="suggestion-chip" onclick="sendSuggestion('I need someone to talk to')">Need to talk</div>
                </div>
            </div>
        `;
    }
}

// Chat functionality - Frontend only
async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    
    if (!chatInput || !sendButton) return;
    
    const message = chatInput.value.trim();
    if (!message) return;
    
    // Disable input during processing
    chatInput.disabled = true;
    sendButton.disabled = true;
    
    // Add user message to chat
    addMessageToChat(message, 'user');
    
    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Update mood based on message
    analyzeMoodFromMessage(message);
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        // Send to backend LLM - Replace with your API endpoint
        const response = await sendMessageToBackend(message);
        
        hideTypingIndicator();
        addMessageToChat(response, 'ai');
        
        // Save to chat history
        AppState.chatHistory.push(
            { text: message, sender: 'user', timestamp: new Date().toISOString() },
            { text: response, sender: 'ai', timestamp: new Date().toISOString() }
        );
        
    } catch (error) {
        console.error('Error sending message to backend:', error);
        hideTypingIndicator();
        addMessageToChat('Sorry, I\'m having trouble connecting right now. Please try again.', 'ai');
    }
    
    // Re-enable input
    chatInput.disabled = false;
    sendButton.disabled = false;
    chatInput.focus();
    
    updateActivityTime();
}

// Backend API call - Replace with your actual backend endpoint
async function sendMessageToBackend(message) {
    // Example API call structure - modify according to your backend
    const apiUrl = '/api/chat'; // Replace with your backend URL
    
    const requestBody = {
        message: message,
        userId: AppState.userProfile?.name || 'anonymous',
        mood: AppState.userMood,
        sessionId: generateSessionId(),
        timestamp: new Date().toISOString()
    };
    
    const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            // Add any authentication headers if needed
        },
        body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data.response || data.message || 'I apologize, but I couldn\'t process your message right now.';
}

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function sendSuggestion(suggestionText) {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.value = suggestionText;
        sendMessage();
        
        // Hide welcome message after first interaction
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }
    }
}

function addMessageToChat(message, sender) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    // Hide welcome message after first user message
    if (sender === 'user') {
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.style.display = 'none';
        }
    }
    
    const messageElement = document.createElement('div');
    messageElement.className = `message ${sender}-message`;
    
    const avatar = sender === 'user' ? '👤' : '🤗';
    const timestamp = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    messageElement.innerHTML = `
        <div class="message-avatar"><span>${avatar}</span></div>
        <div class="message-content">
            <div class="message-text">${escapeHtml(message)}</div>
            <div class="message-time">${timestamp}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageElement);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function showTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'flex';
        
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }
}

function analyzeMoodFromMessage(message) {
    const lowerMessage = message.toLowerCase();
    let newMood = 'neutral';
    let maxMatches = 0;
    
    // Check each mood category
    Object.entries(MoodKeywords).forEach(([mood, keywords]) => {
        const matches = keywords.filter(keyword => lowerMessage.includes(keyword)).length;
        if (matches > maxMatches) {
            maxMatches = matches;
            newMood = mood;
        }
    });
    
    if (newMood !== AppState.userMood) {
        AppState.userMood = newMood;
        updateMoodDisplay();
    }
}

function updateMoodDisplay() {
    const moodText = document.getElementById('moodText');
    const moodLight1 = document.getElementById('moodLight1');
    const moodLight2 = document.getElementById('moodLight2');
    
    const currentMood = MoodSystem[AppState.userMood];
    
    if (moodText && currentMood) {
        moodText.textContent = `Mood: ${currentMood.text}`;
    }
    
    if (moodLight1 && moodLight2 && currentMood) {
        moodLight1.style.backgroundColor = currentMood.color;
        moodLight2.style.backgroundColor = currentMood.color;
        
        // Add appropriate animation based on mood
        const animationClass = AppState.userMood === 'anxious' ? 'pulse-fast' : 
                              AppState.userMood === 'excited' ? 'pulse-energetic' : 'pulse';
        
        moodLight1.style.animation = `${animationClass} 2s infinite`;
        moodLight2.style.animation = `${animationClass} 2s infinite`;
    }
}

// Personality Test functionality
function initializePersonalityTest() {
    AppState.currentQuestion = 0;
    AppState.testAnswers = [];
    AppState.testProgress = 0;
    
    const testResults = document.getElementById('testResults');
    if (testResults) {
        testResults.style.display = 'none';
    }
}

function displayCurrentQuestion() {
    const questionnaire = document.getElementById('questionnaire');
    const progressBar = document.getElementById('progressBar');
    
    if (!questionnaire) return;
    
    if (AppState.currentQuestion >= PersonalityQuestions.length) {
        showTestResults();
        return;
    }
    
    const question = PersonalityQuestions[AppState.currentQuestion];
    const progress = (AppState.currentQuestion / PersonalityQuestions.length) * 100;
    
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
    }
    
    questionnaire.innerHTML = `
        <div class="question-card">
            <div class="question-number">Question ${AppState.currentQuestion + 1} of ${PersonalityQuestions.length}</div>
            <div class="question-text">${question.text}</div>
            <div class="answer-options">
                <button class="answer-btn" onclick="answerQuestion(1)">Strongly Disagree</button>
                <button class="answer-btn" onclick="answerQuestion(2)">Disagree</button>
                <button class="answer-btn" onclick="answerQuestion(3)">Neutral</button>
                <button class="answer-btn" onclick="answerQuestion(4)">Agree</button>
                <button class="answer-btn" onclick="answerQuestion(5)">Strongly Agree</button>
            </div>
        </div>
    `;
}

function answerQuestion(score) {
    const currentQuestion = PersonalityQuestions[AppState.currentQuestion];
    
    let finalScore = currentQuestion.reverse ? (6 - score) : score;
    
    AppState.testAnswers.push({
        questionId: currentQuestion.id,
        trait: currentQuestion.trait,
        score: finalScore,
        originalScore: score
    });
    
    AppState.currentQuestion++;
    
    if (AppState.currentQuestion >= PersonalityQuestions.length) {
        showTestResults();
    } else {
        displayCurrentQuestion();
    }
}

function calculatePersonalityScores() {
    const traits = {
        extraversion: [],
        neuroticism: [],
        openness: [],
        conscientiousness: [],
        agreeableness: []
    };
    
    AppState.testAnswers.forEach(answer => {
        if (traits[answer.trait]) {
            traits[answer.trait].push(answer.score);
        }
    });
    
    const scores = {};
    Object.keys(traits).forEach(trait => {
        const traitScores = traits[trait];
        const average = traitScores.reduce((sum, score) => sum + score, 0) / traitScores.length;
        scores[trait] = Math.round(average * 20);
    });
    
    return scores;
}

function getPersonalityType(scores) {
    const types = {
        'ANALYST': {
            condition: scores.openness > 70 && scores.conscientiousness > 60,
            description: 'You are analytical, strategic, and enjoy solving complex problems. You tend to be independent and value competence highly.'
        },
        'DIPLOMAT': {
            condition: scores.agreeableness > 70 && scores.openness > 60,
            description: 'You are empathetic, idealistic, and focused on human potential. You value harmony and are driven by your values.'
        },
        'SENTINEL': {
            condition: scores.conscientiousness > 70 && scores.neuroticism < 50,
            description: 'You are practical, fact-minded, and reliable. You prefer structure and stability in your environment.'
        },
        'EXPLORER': {
            condition: scores.extraversion > 60 && scores.openness > 60,
            description: 'You are spontaneous, energetic, and adaptable. You enjoy new experiences and thinking on your feet.'
        },
        'HARMONIZER': {
            condition: scores.agreeableness > 80,
            description: 'You are cooperative, trusting, and focused on maintaining positive relationships with others.'
        },
        'BALANCED': {
            condition: true,
            description: 'You show a balanced personality profile with moderate scores across different traits, indicating adaptability and flexibility.'
        }
    };
    
    for (const [typeName, typeInfo] of Object.entries(types)) {
        if (typeInfo.condition) {
            return { name: typeName, description: typeInfo.description };
        }
    }
    
    return types.BALANCED;
}

function showTestResults() {
    const questionnaire = document.getElementById('questionnaire');
    const testResults = document.getElementById('testResults');
    const progressBar = document.getElementById('progressBar');
    
    if (progressBar) {
        progressBar.style.width = '100%';
    }
    
    if (questionnaire) {
        questionnaire.style.display = 'none';
    }
    
    const scores = calculatePersonalityScores();
    const personalityType = getPersonalityType(scores);
    
    const personalityTypeEl = document.getElementById('personalityType');
    const personalityDescEl = document.getElementById('personalityDescription');
    
    if (personalityTypeEl) {
        personalityTypeEl.textContent = `Your Personality Type: ${personalityType.name}`;
    }
    
    if (personalityDescEl) {
        personalityDescEl.textContent = personalityType.description;
    }
    
    const traitsContainer = document.getElementById('traitsContainer');
    if (traitsContainer) {
        const traitNames = {
            extraversion: 'Extraversion',
            neuroticism: 'Emotional Stability',
            openness: 'Openness to Experience',
            conscientiousness: 'Conscientiousness',
            agreeableness: 'Agreeableness'
        };
        
        traitsContainer.innerHTML = Object.keys(scores).map(trait => {
            const score = scores[trait];
            const adjustedScore = trait === 'neuroticism' ? (100 - score) : score;
            
            return `
                <div class="trait-score">
                    <div class="trait-name">${traitNames[trait]}</div>
                    <div class="trait-bar">
                        <div class="trait-fill" style="width: ${adjustedScore}%"></div>
                    </div>
                    <div class="trait-percentage">${adjustedScore}%</div>
                </div>
            `;
        }).join('');
    }
    
    if (testResults) {
        testResults.style.display = 'block';
    }
}

function retakeTest() {
    const questionnaire = document.getElementById('questionnaire');
    const testResults = document.getElementById('testResults');
    
    if (questionnaire) {
        questionnaire.style.display = 'block';
    }
    
    if (testResults) {
        testResults.style.display = 'none';
    }
    
    initializePersonalityTest();
    displayCurrentQuestion();
}

// FAQ functionality
function toggleFAQ(element) {
    const answer = element.nextElementSibling;
    const isOpen = answer.style.display === 'block';
    
    document.querySelectorAll('.faq-answer').forEach(faqAnswer => {
        faqAnswer.style.display = 'none';
    });
    
    document.querySelectorAll('.faq-question span').forEach(span => {
        span.textContent = '+';
    });
    
    if (!isOpen) {
        answer.style.display = 'block';
        element.querySelector('span').textContent = '−';
    }
}

// Data persistence functions
function saveUserData() {
    try {
        const userData = {
            isLoggedIn: AppState.isLoggedIn,
            userProfile: AppState.userProfile,
            lastActive: new Date().toISOString()
        };
        console.log('User data saved:', userData);
    } catch (error) {
        console.error('Error saving user data:', error);
    }
}

function loadUserData() {
    try {
        AppState.isLoggedIn = false;
        AppState.userProfile = null;
        updateLoginDisplay();
        console.log('User data loaded');
    } catch (error) {
        console.error('Error loading user data:', error);
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Export functions for global access
window.toggleSidebar = toggleSidebar;
window.toggleLogin = toggleLogin;
window.logout = logout;
window.sendSuggestion = sendSuggestion;
window.answerQuestion = answerQuestion;
window.retakeTest = retakeTest;
window.toggleFAQ = toggleFAQ;

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}