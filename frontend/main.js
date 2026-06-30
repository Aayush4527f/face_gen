import * as ort from 'onnxruntime-web';

const LATENT_DIM = 16;
let decoderSession = null;
let encoderSession = null;
let isGenerating = false;

// Initialize state to 0 for all latent dimensions
const latentSpace = new Float32Array(LATENT_DIM).fill(0);

// Facing direction vector (extracted left ⟷ right facing control)
const FACING_DIRECTION_VECTOR = [
  0.2899738848209381, 0.009440995752811432, -0.026998644694685936, 0.19809047877788544,
  -0.14305420219898224, -0.5916984677314758, 0.16607166826725006, 0.27834460139274597,
  -0.1228371188044548, 0.15463581681251526, -0.049575693905353546, -0.16682018339633942,
  0.038495831191539764, -0.5326293706893921, -0.17295007407665253, -0.125915065407753
];
let facingValue = 0.0;
let oldFacingValue = 0.0;

// DOM Elements
const slidersContainer = document.getElementById('sliders-container');
const randomizeBtn = document.getElementById('randomize-btn');
const resetBtn = document.getElementById('reset-btn');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('loading-overlay');

// Semantic Controls DOM Elements
const controlFacing = document.getElementById('control-facing');
const valFacing = document.getElementById('val-facing');

// Tab Navigation Elements
const tabSliders = document.getElementById('tab-sliders');
const tabEncoder = document.getElementById('tab-encoder');
const paneSliders = document.getElementById('pane-sliders');
const paneEncoder = document.getElementById('pane-encoder');

// Upload and Preview Elements
const uploadArea = document.getElementById('upload-area');
const imageLoader = document.getElementById('image-loader');
const inputPlaceholder = document.getElementById('input-placeholder');
const inputPreview = document.getElementById('input-preview');
const encoderFeedback = document.getElementById('encoder-feedback');

// Request animation frame ID for debouncing
let rafId = null;

// Tab Switch Logic
tabSliders.addEventListener('click', () => {
  tabSliders.classList.add('active');
  tabEncoder.classList.remove('active');
  paneSliders.classList.add('active');
  paneEncoder.classList.remove('active');
});

tabEncoder.addEventListener('click', () => {
  tabEncoder.classList.add('active');
  tabSliders.classList.remove('active');
  paneEncoder.classList.add('active');
  paneSliders.classList.remove('active');
});

// Facing Direction Slider Listener
controlFacing.addEventListener('input', (e) => {
  facingValue = parseFloat(e.target.value);
  valFacing.innerText = facingValue.toFixed(2);
  
  const delta = facingValue - oldFacingValue;
  oldFacingValue = facingValue;

  // Apply delta across base dimensions and update manual slider positions in DOM
  for (let i = 0; i < LATENT_DIM; i++) {
    const bounds = sliderBounds[i];
    let val = latentSpace[i] + delta * FACING_DIRECTION_VECTOR[i];
    val = Math.max(bounds.min, Math.min(bounds.max, val));
    latentSpace[i] = val;

    // Update DOM inputs and value labels
    const inputEl = slidersContainer.querySelector(`input[data-index="${i}"]`);
    if (inputEl) {
      inputEl.value = val.toString();
    }
    const valDisplay = document.getElementById(`val-${i}`);
    if (valDisplay) {
      valDisplay.innerText = val.toFixed(2);
    }
  }
  
  if (rafId) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(generateImage);
});

// Custom bounds for each of the 16 latent dimensions
const sliderBounds = [
  { min: -3.0, max: 3.0 },
  { min: -1.5, max: 1.5 },
  { min: -1.5, max: 3.0 },
  { min: -1.5, max: 2.0 },
  { min: -1.5, max: 1.5 },
  { min: -2.0, max: 2.0 },
  { min: -1.5, max: 1.5 },
  { min: -1.5, max: 1.5 },
  { min: -2.0, max: 2.0 },
  { min: -2.0, max: 1.5 },
  { min: -1.5, max: 1.5 },
  { min: -2.0, max: 2.0 },
  { min: -1.0, max: 1.0 },
  { min: -2.0, max: 2.0 },
  { min: -1.0, max: 1.0 },
  { min: -1.5, max: 1.5 }
];

function createSliders() {
  slidersContainer.innerHTML = '';
  
  // Create details tag for the entire flat list of sliders
  const groupDetails = document.createElement('details');
  groupDetails.className = 'sliders-group';
  // Kept collapsed/closed by default
  
  // Create summary tag
  const groupSummary = document.createElement('summary');
  groupSummary.className = 'sliders-group-title';
  groupSummary.innerText = "Latent Space Sliders (Dimensions 1-16)";
  groupDetails.appendChild(groupSummary);
  
  // Grid inside collapsible group
  const gridDiv = document.createElement('div');
  gridDiv.className = 'sliders-group-content';
  
  // Populate all 16 sliders
  for (let i = 0; i < LATENT_DIM; i++) {
    const bounds = sliderBounds[i];
    const wrapper = document.createElement('div');
    wrapper.className = 'slider-container';

    const labelRow = document.createElement('div');
    labelRow.className = 'slider-label-row';
    
    const label = document.createElement('span');
    label.innerText = `Dim ${i + 1}`;
    
    const valDisplay = document.createElement('span');
    valDisplay.className = 'slider-val';
    valDisplay.id = `val-${i}`;
    valDisplay.innerText = '0.00';

    labelRow.appendChild(label);
    labelRow.appendChild(valDisplay);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = bounds.min.toString();
    input.max = bounds.max.toString();
    input.step = '0.05';
    input.value = '0';
    input.dataset.index = i;

    // Event listener for user slide interaction
    input.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      valDisplay.innerText = val.toFixed(2);
      latentSpace[i] = val;
      
      // Reset facing slider base reference since user is manual tuning
      facingValue = 0.0;
      oldFacingValue = 0.0;
      if (controlFacing) controlFacing.value = '0.0';
      if (valFacing) valFacing.innerText = '0.00';
      
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(generateImage);
    });

    wrapper.appendChild(labelRow);
    wrapper.appendChild(input);
    gridDiv.appendChild(wrapper);
  }
  
  groupDetails.appendChild(gridDiv);
  slidersContainer.appendChild(groupDetails);
}

// Randomize all sliders to normally distributed values scaled to individual bounds
randomizeBtn.addEventListener('click', () => {
  const inputs = slidersContainer.querySelectorAll('input');
  inputs.forEach((input) => {
    const i = parseInt(input.dataset.index);
    const bounds = sliderBounds[i];
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    
    // Scale normal variable to span the bounds: mid + std_dev * (half_width / 2)
    const mid = (bounds.min + bounds.max) / 2.0;
    const halfWidth = (bounds.max - bounds.min) / 2.0;
    let val = mid + num * (halfWidth / 2.0);
    
    // Clamp to bounds
    val = Math.max(bounds.min, Math.min(bounds.max, val));
    
    latentSpace[i] = val;
    input.value = val.toString();
    const valDisplay = document.getElementById(`val-${i}`);
    if (valDisplay) valDisplay.innerText = val.toFixed(2);
  });
  
  // Reset facing slider on randomize
  facingValue = 0.0;
  oldFacingValue = 0.0;
  controlFacing.value = '0.0';
  valFacing.innerText = '0.00';
  
  if (rafId) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(generateImage);
});

// Reset all sliders back to mean (0)
resetBtn.addEventListener('click', () => {
  latentSpace.fill(0);
  
  // Reset facing slider on reset
  facingValue = 0.0;
  oldFacingValue = 0.0;
  controlFacing.value = '0.0';
  valFacing.innerText = '0.00';
  
  const inputs = slidersContainer.querySelectorAll('input');
  inputs.forEach((input) => {
    const i = parseInt(input.dataset.index);
    input.value = '0';
    const valDisplay = document.getElementById(`val-${i}`);
    if (valDisplay) valDisplay.innerText = '0.00';
  });
  if (rafId) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(generateImage);
});

// Load the ONNX model files
async function initModel() {
  try {
    // Load Decoder model
    decoderSession = await ort.InferenceSession.create('/decoderv4.1.onnx', { executionProviders: ['wasm'] });
    console.log('Decoder loaded successfully');
    
    // Load Encoder model
    try {
      encoderSession = await ort.InferenceSession.create('/encoderv4.1.onnx', { executionProviders: ['wasm'] });
      console.log('Encoder loaded successfully');
      setupEncoderEvents();
    } catch (encErr) {
      console.error('Failed to load encoder model. Upload tab will be locked.', encErr);
      showFeedback('Encoder model unavailable. Check console.', 'error');
      tabEncoder.style.pointerEvents = 'none';
      tabEncoder.style.opacity = '0.5';
    }
    
    overlay.classList.remove('active');
    generateImage();
  } catch (err) {
    console.error('Failed to load model session', err);
    overlay.querySelector('p').innerText = 'Error loading models. Check console.';
    overlay.querySelector('.spinner').style.display = 'none';
  }
}

// Generate the image over the latent vector using the Decoder session
async function generateImage() {
  if (!decoderSession || isGenerating) return;
  isGenerating = true;

  try {
    // Decode base latentSpace array directly (which in-turn gets edited by semantic sliders)
    const tensorBuffer = Float32Array.from(latentSpace);
    const tensor = new ort.Tensor('float32', tensorBuffer, [1, LATENT_DIM]);

    const inputName = decoderSession.inputNames[0];
    const feedsObj = {};
    feedsObj[inputName] = tensor;
    
    const results = await decoderSession.run(feedsObj);
    const outputName = decoderSession.outputNames[0];
    const outputTensor = results[outputName];
    
    renderToCanvas(outputTensor.data);
  } catch (err) {
    console.error('Decoder inference error', err);
  } finally {
    isGenerating = false;
  }
}

// Paint pixel colors from reconstructed data tensor
function renderToCanvas(data) {
  const width = 64;
  const height = 64;
  const spatialSize = width * height;
  
  const imgData = ctx.createImageData(width, height);
  for (let i = 0; i < spatialSize; i++) {
    // CHW arrangement: R is first spatialSize, G is second, B is third
    const r = Math.min(255, Math.max(0, Math.floor(data[i] * 255)));
    const g = Math.min(255, Math.max(0, Math.floor(data[spatialSize + i] * 255)));
    const b = Math.min(255, Math.max(0, Math.floor(data[2 * spatialSize + i] * 255)));
    
    const offset = i * 4;
    imgData.data[offset] = r;
    imgData.data[offset + 1] = g;
    imgData.data[offset + 2] = b;
    imgData.data[offset + 3] = 255;
  }
  
  ctx.putImageData(imgData, 0, 0);
}

// ==========================================
// IMAGE ENCODER DRAG-AND-DROP & UPLOAD
// ==========================================

function setupEncoderEvents() {
  // Click on upload area to open file explorer
  uploadArea.addEventListener('click', () => imageLoader.click());
  
  imageLoader.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  });

  // Drag over effects
  ['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      uploadArea.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      uploadArea.classList.remove('dragover');
    }, false);
  });

  uploadArea.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      processFile(files[0]);
    }
  });
}

function showFeedback(msg, type = '') {
  encoderFeedback.innerText = msg;
  encoderFeedback.className = `encoder-feedback-msg ${type}`;
  encoderFeedback.style.display = 'block';
}

// Preprocessor for face image upload
function processFile(file) {
  if (!file.type.startsWith('image/')) {
    showFeedback('Please select a valid image file.', 'error');
    return;
  }

  showFeedback('Reading file...', 'info');

  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = () => {
    const img = new Image();
    img.src = reader.result;
    img.onload = async () => {
      // 1. Show preview of original input
      inputPreview.src = img.src;
      inputPreview.style.display = 'block';
      inputPlaceholder.style.display = 'none';
      
      showFeedback('Preprocessing image to 64x64...', 'info');
      
      try {
        // 2. Preprocess: resize to 64x64 on a temporary canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 64;
        tempCanvas.height = 64;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(img, 0, 0, 64, 64);
        
        // 3. Extract pixels
        const imgData = tempCtx.getImageData(0, 0, 64, 64);
        const tensorData = new Float32Array(3 * 64 * 64);
        
        // 4. Fill in NCHW format
        const spatialSize = 64 * 64;
        for (let i = 0; i < spatialSize; i++) {
          const r = imgData.data[i * 4] / 255.0;
          const g = imgData.data[i * 4 + 1] / 255.0;
          const b = imgData.data[i * 4 + 2] / 255.0;
          
          tensorData[i] = r;
          tensorData[spatialSize + i] = g;
          tensorData[2 * spatialSize + i] = b;
        }
        
        showFeedback('Encoding face features...', 'info');
        
        // 5. Build input tensor
        const tensorInput = new ort.Tensor('float32', tensorData, [1, 3, 64, 64]);
        const inputName = encoderSession.inputNames[0];
        const feeds = {};
        feeds[inputName] = tensorInput;
        
        // 6. Run ONNX Encoder
        const results = await encoderSession.run(feeds);
        const outputName = encoderSession.outputNames[0];
        const zData = results[outputName].data; // Float32Array(16)
        
        // 7. Update sliders and state (clamped to individual bounds)
        for (let i = 0; i < LATENT_DIM; i++) {
          const bounds = sliderBounds[i];
          const val = Math.max(bounds.min, Math.min(bounds.max, zData[i]));
          latentSpace[i] = val;
          
          // Find matching input element and val label in DOM
          const inputEl = slidersContainer.querySelector(`input[data-index="${i}"]`);
          if (inputEl) {
            inputEl.value = val.toString();
          }
          const valDisplay = document.getElementById(`val-${i}`);
          if (valDisplay) {
            valDisplay.innerText = val.toFixed(2);
          }
        }
        
        // Reset facing slider on new image encoding (since it is already baked into zData)
        facingValue = 0.0;
        oldFacingValue = 0.0;
        controlFacing.value = '0.0';
        valFacing.innerText = '0.00';
        
        // 8. Reconstruct image using decoder
        await generateImage();
        showFeedback('Success! Face encoded and reconstructed.', 'success');
      } catch (err) {
        console.error('Encoder inference failed', err);
        showFeedback('Inference failed. Check console.', 'error');
      }
    };
  };
}

// Modal Event Listeners
const infoBtn = document.getElementById('info-btn');
const vectorModal = document.getElementById('vector-modal');
const closeModal = document.getElementById('close-modal');
const vectorCode = document.getElementById('vector-code');

if (infoBtn && vectorModal && closeModal && vectorCode) {
  // Show vector values formatted in list
  vectorCode.innerText = JSON.stringify(FACING_DIRECTION_VECTOR, null, 2);
  
  infoBtn.addEventListener('click', (e) => {
    e.preventDefault();
    vectorModal.classList.add('active');
  });
  
  closeModal.addEventListener('click', () => {
    vectorModal.classList.remove('active');
  });
  
  // Close modal on click outside content
  window.addEventListener('click', (event) => {
    if (event.target === vectorModal) {
      vectorModal.classList.remove('active');
    }
  });
}

// Dynamic loader for the raw README.txt log file
async function loadReadmeLog() {
  const container = document.getElementById('raw-log-container');
  if (!container) return;
  try {
    const response = await fetch('/README.txt');
    if (!response.ok) throw new Error('File not found');
    const text = await response.text();
    container.innerText = text;
  } catch (err) {
    console.error('Failed to load README.txt', err);
    container.innerText = 'Failed to load chronological log file. Check console details.';
  }
}

// Dark Mode Toggle Logic
const themeToggle = document.getElementById('theme-toggle');
if (themeToggle) {
  const savedTheme = localStorage.getItem('theme');
  const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  if (savedTheme === 'dark' || (!savedTheme && systemPrefersDark)) {
    document.documentElement.classList.add('dark-theme');
    document.body.classList.add('dark-theme');
  } else {
    document.documentElement.classList.remove('dark-theme');
    document.body.classList.remove('dark-theme');
  }
  
  themeToggle.addEventListener('click', () => {
    document.documentElement.classList.toggle('dark-theme');
    document.body.classList.toggle('dark-theme');
    const isDark = document.documentElement.classList.contains('dark-theme');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  });
}

// Boot up sequence
createSliders();
initModel();
loadReadmeLog();

