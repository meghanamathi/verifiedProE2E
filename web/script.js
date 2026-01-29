// Product Fraud Detection System - Frontend JavaScript

const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
const productIdInput = document.getElementById('product-id');
const uploadBtn = document.getElementById('upload-btn');
const cameraBtn = document.getElementById('camera-btn');
const fileInput = document.getElementById('file-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const removeImageBtn = document.getElementById('remove-image');
const cameraContainer = document.getElementById('camera-container');
const cameraVideo = document.getElementById('camera-video');
const cameraCanvas = document.getElementById('camera-canvas');
const captureBtn = document.getElementById('capture-btn');
const closeCameraBtn = document.getElementById('close-camera-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const dismissErrorBtn = document.getElementById('dismiss-error-btn');
const newAnalysisBtn = document.getElementById('new-analysis-btn');

// State
let currentImage = null;
let cameraStream = null;

// Event Listeners
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
removeImageBtn.addEventListener('click', clearImage);
cameraBtn.addEventListener('click', startCamera);
captureBtn.addEventListener('click', captureImage);
closeCameraBtn.addEventListener('click', stopCamera);
analyzeBtn.addEventListener('click', analyzeProduct);
dismissErrorBtn.addEventListener('click', hideError);
newAnalysisBtn.addEventListener('click', resetForm);
productIdInput.addEventListener('input', validateForm);

// File Selection
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        if (!file.type.startsWith('image/')) {
            showError('Please select a valid image file');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            currentImage = file;
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            validateForm();
        };
        reader.readAsDataURL(file);
    }
}

// Clear Image
function clearImage() {
    currentImage = null;
    previewImage.src = '';
    previewContainer.style.display = 'none';
    fileInput.value = '';
    validateForm();
}

// Camera Functions
async function startCamera() {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        cameraVideo.srcObject = cameraStream;
        cameraContainer.style.display = 'block';
        uploadBtn.disabled = true;
    } catch (error) {
        showError('Unable to access camera. Please check permissions.');
        console.error('Camera error:', error);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraContainer.style.display = 'none';
    uploadBtn.disabled = false;
}

function captureImage() {
    const canvas = cameraCanvas;
    const video = cameraVideo;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob((blob) => {
        currentImage = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
        
        canvas.toDataURL('image/jpeg');
        previewImage.src = canvas.toDataURL('image/jpeg');
        previewContainer.style.display = 'block';
        
        stopCamera();
        validateForm();
    }, 'image/jpeg', 0.9);
}

// Form Validation
function validateForm() {
    const hasProductId = productIdInput.value.trim() !== '';
    const hasImage = currentImage !== null;
    
    analyzeBtn.disabled = !(hasProductId && hasImage);
}

// Analyze Product
async function analyzeProduct() {
    const productId = productIdInput.value.trim();
    
    if (!productId || !currentImage) {
        showError('Please provide both Product ID and image');
        return;
    }
    
    // Hide previous results/errors
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    // Show loading
    loading.style.display = 'block';
    analyzeBtn.disabled = true;
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('product_id', productId);
        formData.append('returned_image', currentImage);
        
        // Make API request
        const response = await fetch(`${API_BASE_URL}/full-analysis`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || 'Analysis failed');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showError(error.message || 'Failed to analyze product. Please ensure the API server is running.');
        console.error('Analysis error:', error);
    } finally {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

// Display Results
function displayResults(data) {
    // Update status banner
    const statusBanner = document.getElementById('status-banner');
    const statusTitle = document.getElementById('status-title');
    const statusRecommendation = document.getElementById('status-recommendation');
    
    statusTitle.textContent = data.status;
    statusRecommendation.textContent = data.recommendation;
    
    // Set banner color
    statusBanner.className = 'status-banner';
    if (data.color === 'green') {
        statusBanner.classList.add('success');
    } else if (data.color === 'orange') {
        statusBanner.classList.add('warning');
    } else {
        statusBanner.classList.add('danger');
    }
    
    // Authentication results
    const authStatus = document.getElementById('auth-status');
    const similarityScore = document.getElementById('similarity-score');
    const authConfidence = document.getElementById('auth-confidence');
    const similarityProgress = document.getElementById('similarity-progress');
    
    authStatus.textContent = data.authentication.is_authentic ? '✓ Authentic' : '✗ Not Authentic';
    authStatus.style.color = data.authentication.is_authentic ? 'var(--success-color)' : 'var(--danger-color)';
    
    const scorePercent = (data.authentication.similarity_score * 100).toFixed(1);
    similarityScore.textContent = `${scorePercent}%`;
    authConfidence.textContent = data.authentication.confidence;
    
    similarityProgress.style.width = `${scorePercent}%`;
    
    // Damage assessment results
    const damageLevel = document.getElementById('damage-level');
    const damageConfidence = document.getElementById('damage-confidence');
    const damageBreakdownContent = document.getElementById('damage-breakdown-content');
    
    damageLevel.textContent = data.damage_assessment.damage_level;
    
    // Color code damage level
    if (data.damage_assessment.damage_level === 'No Damage') {
        damageLevel.style.color = 'var(--success-color)';
    } else if (data.damage_assessment.damage_level === 'Minor Damage') {
        damageLevel.style.color = 'var(--warning-color)';
    } else {
        damageLevel.style.color = 'var(--danger-color)';
    }
    
    const confidencePercent = (data.damage_assessment.confidence * 100).toFixed(1);
    damageConfidence.textContent = `${confidencePercent}%`;
    
    // Damage breakdown
    damageBreakdownContent.innerHTML = '';
    const predictions = data.damage_assessment.all_predictions;
    
    for (const [className, probability] of Object.entries(predictions)) {
        const percent = (probability * 100).toFixed(1);
        
        const item = document.createElement('div');
        item.className = 'breakdown-item';
        item.innerHTML = `
            <span>${className}</span>
            <div class="breakdown-bar">
                <div class="breakdown-bar-fill" style="width: ${percent}%"></div>
            </div>
            <span>${percent}%</span>
        `;
        damageBreakdownContent.appendChild(item);
    }
    
    // Show results section with animation
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Error Handling
function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorSection.style.display = 'none';
}

// Reset Form
function resetForm() {
    productIdInput.value = '';
    clearImage();
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Check API Health on Load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (!data.siamese_model_loaded || !data.damage_model_loaded) {
            console.warn('Models not loaded. Please train the models first.');
        }
    } catch (error) {
        console.warn('API server not reachable. Please start the server with: python api/app.py');
    }
}

// Initialize
checkAPIHealth();
