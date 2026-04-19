import * as ort from 'onnxruntime-web';

const LATENT_DIM = 16;
let session = null;
let isGenerating = false;

// Initialize state to 0 for all knobs
const latentSpace = new Float32Array(LATENT_DIM).fill(0);

// Elements
const slidersContainer = document.getElementById('sliders-container');
const randomizeBtn = document.getElementById('randomize-btn');
const resetBtn = document.getElementById('reset-btn');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('loading-overlay');

// Request animation frame ID for debouncing
let rafId = null;

function createSliders() {
  slidersContainer.innerHTML = '';
  for (let i = 0; i < LATENT_DIM; i++) {
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
    input.min = '-3';
    input.max = '3';
    input.step = '0.05';
    input.value = '0';
    input.dataset.index = i;

    // Fast response using input event
    input.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      valDisplay.innerText = val.toFixed(2);
      latentSpace[i] = val;
      
      // Request generating new frame smoothly
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(generateImage);
    });

    wrapper.appendChild(labelRow);
    wrapper.appendChild(input);
    slidersContainer.appendChild(wrapper);
  }
}

// Randomize all sliders to normal distribution approx mapping to [-3, 3]
randomizeBtn.addEventListener('click', () => {
  const inputs = slidersContainer.querySelectorAll('input');
  inputs.forEach((input, i) => {
    // Generate normally distributed value mapped correctly
    let u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    let num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    
    // Clamp to min/max
    num = Math.max(-3, Math.min(3, num));
    
    latentSpace[i] = num;
    input.value = num;
    document.getElementById(`val-${i}`).innerText = num.toFixed(2);
  });
  
  if (rafId) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(generateImage);
});

// Reset all sliders back to mean (0)
resetBtn.addEventListener('click', () => {
  latentSpace.fill(0);
  const inputs = slidersContainer.querySelectorAll('input');
  inputs.forEach((input, i) => {
    input.value = '0';
    document.getElementById(`val-${i}`).innerText = '0.00';
  });
  if (rafId) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(generateImage);
});

// Load the model
async function initModel() {
  try {
    // Determine automatically or set path implicitly
    // We copied the model files to /decoder.onnx serving public output.
    session = await ort.InferenceSession.create('/decoderv3.4.onnx', { executionProviders: ['wasm'] });
    
    console.log('Model loaded successfully');
    overlay.classList.remove('active');
    
    // Generate initial flat face (mean face)
    generateImage();
  } catch (err) {
    console.error('Failed to load model', err);
    overlay.querySelector('p').innerText = 'Error loading model. Check console.';
    overlay.querySelector('.spinner').style.display = 'none';
  }
}

// Generate the image over the given features
async function generateImage() {
  if (!session || isGenerating) return;
  isGenerating = true;

  try {
    // Generate a new array every time to prevent detached buffer errors with WASM
    const tensorBuffer = Float32Array.from(latentSpace);
    const tensor = new ort.Tensor('float32', tensorBuffer, [1, LATENT_DIM]);

    const inputName = session.inputNames[0]; // e.g., 'z'
    const feedsObj = {};
    feedsObj[inputName] = tensor;
    
    const results = await session.run(feedsObj);
    const outputName = session.outputNames[0]; // e.g., 'sigmoid'
    const outputTensor = results[outputName];
    
    renderToCanvas(outputTensor.data);
  } catch (err) {
    console.error('Inference error', err);
  } finally {
    isGenerating = false;
  }
}

// Paint pixel colors
function renderToCanvas(data) {
  const width = 48;
  const height = 48;
  const spatialSize = width * height;
  
  const imgData = ctx.createImageData(width, height);
  for (let i = 0; i < spatialSize; i++) {
    // Data is NCHW, so R is first spatialSize elements, G is second, B is third
    const r = Math.min(255, Math.max(0, Math.floor(data[i] * 255)));
    const g = Math.min(255, Math.max(0, Math.floor(data[spatialSize + i] * 255)));
    const b = Math.min(255, Math.max(0, Math.floor(data[2 * spatialSize + i] * 255)));
    
    const offset = i * 4;
    imgData.data[offset] = r;     // R
    imgData.data[offset + 1] = g; // G
    imgData.data[offset + 2] = b; // B
    imgData.data[offset + 3] = 255; // A
  }
  
  ctx.putImageData(imgData, 0, 0);
}

// Boot up sequence
createSliders();
initModel();
