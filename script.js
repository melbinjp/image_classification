import {
  AutoTokenizer,
  AutoProcessor,
  CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection,
  env,
  RawImage,
  dot,
  softmax
} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@latest/dist/transformers.min.js';

// Configure environment to ONLY use the remote Hugging Face Hub (prevent local /models/ 404s)
env.allowLocalModels = false;
env.useBrowserCache = true;

const MODEL_NAME = 'Xenova/clip-vit-base-patch16';
let textModel, visionModel, tokenizer, processor, textEmbeddings, cachedLabels;
let isClassifying = false;

// Pretext for high-performance layout
const pretext = window.pretext || { layout: (text) => text };

// UI Elements
const elements = {
  uploadZone: document.getElementById('dropZone'),
  imageUpload: document.getElementById('imageUpload'),
  previewContainer: document.getElementById('previewContainer'),
  preview: document.getElementById('preview'),
  clearBtn: document.getElementById('clearBtn'),
  resultsSection: document.getElementById('resultsSection'),
  modelStatus: document.getElementById('modelStatus'),
  confidenceMeter: document.getElementById('confidenceMeter'),
  labelInput: document.getElementById('labelInput'),
};

// Model initialization
async function initializeModel() {
  try {
    // Load models (text and vision)
    textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_NAME, {
      quantized: true,
      progress_callback: downloadProgress,
    });
    visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME, {
      quantized: true,
      progress_callback: downloadProgress,
    });

    // Load tokenizer and processor
    tokenizer = await AutoTokenizer.from_pretrained(MODEL_NAME);
    processor = await AutoProcessor.from_pretrained(MODEL_NAME);

    elements.modelStatus.className = 'model-status ready';
    elements.modelStatus.innerHTML = `<i class="fas fa-check-circle"></i> AI Model Ready`;
  } catch (error) {
    elements.modelStatus.className = 'model-status error';
    elements.modelStatus.innerHTML = `<i class="fas fa-times-circle"></i> Model failed to load`;
    console.error('Model loading error:', error);
  }
}

// Progress callback function
function downloadProgress(progress) {
  if (progress.status === 'downloading') return; // Skip granular download ticks for cleaner UI
  const percent = Math.round(progress * 100);
  if (!isNaN(percent)) {
    elements.modelStatus.innerHTML = `<i class="fas fa-circle-notch fa-spin"></i> Loading AI model... ${percent}%`;
  }
}

// Process text labels
async function processLabels(labels) {
  if (textEmbeddings) {
    textEmbeddings = null; // Let GC handle memory
  }

  const textInputs = await tokenizer(labels, {
    padding: true,
    truncation: true,
    return_tensors: true,
  });

  const outputs = await textModel(textInputs);
  const tempEmbeddings = outputs.text_embeds;

  // Normalize embeddings using built-in method and convert to nested list
  textEmbeddings = tempEmbeddings.normalize(2, -1).tolist();
  cachedLabels = labels.join(',');
}

// Classify image
async function classifyImage(imageElement) {
  if (isClassifying) return;
  isClassifying = true;
  let imageEmbedding, similarity;

  try {
    elements.resultsSection.style.display = 'block';
    elements.confidenceMeter.innerHTML = '<div class="model-status" style="margin:0"><i class="fas fa-circle-notch fa-spin"></i> Analyzing image...</div>';

    const labels = elements.labelInput.value.split(',').map((s) => s.trim()).filter(s => s !== '');
    if (labels.length === 0) throw new Error("Please provide at least one text label");

    if (!textEmbeddings || labels.join(',') !== cachedLabels) {
      await processLabels(labels);
    }

    const image = await RawImage.fromURL(imageElement.src);
    const imageInputs = await processor(image);
    const imageOutputs = await visionModel(imageInputs);

    imageEmbedding = imageOutputs.image_embeds;
    const normalizedImage = imageEmbedding.normalize(2, -1).tolist()[0];

    // Compute similarity using dot product
    // See `model.logit_scale` parameter of original model
    const exp_logit_scale = Math.exp(4.6052);

    const similarities = textEmbeddings.map(
      x => dot(x, normalizedImage) * exp_logit_scale
    );

    const sortedIndices = softmax(similarities)
      .map((x, i) => ({
        label: labels[i],
        score: x * 100
      }))
      .sort((a, b) => b.score - a.score);

    displayResults(sortedIndices);
  } catch (error) {
    console.error('Classification error:', error);
    elements.confidenceMeter.innerHTML = `<div class="model-status error" style="margin:0"><i class="fas fa-exclamation-triangle"></i> Error: ${error.message}</div>`;
  } finally {
    isClassifying = false;
  }
}

// Display results dynamically
function displayResults(probabilities) {
  // Use Pretext layout for the container header
  const headerContent = pretext.layout("Analysis Results");
  
  elements.confidenceMeter.innerHTML = '';

  probabilities.forEach((item, index) => {
    const isTop = index === 0;
    const scoreFormatted = item.score.toFixed(1) + '%';
    
    // Pre-layout the label using Pretext
    const laidOutLabel = pretext.layout(item.label);

    const row = document.createElement('div');
    row.className = `result-row ${isTop ? 'top-result' : ''}`;

    row.innerHTML = `
            <div class="result-label" title="${item.label}">${laidOutLabel}</div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width: 0%"></div>
            </div>
            <div class="result-score">${scoreFormatted}</div>
        `;

    elements.confidenceMeter.appendChild(row);

    // Trigger animation after a tiny delay
    setTimeout(() => {
      const fill = row.querySelector('.score-bar-fill');
      if (fill) fill.style.width = `${item.score}%`;
    }, 50);
  });
}

// File Handling Logic
function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    elements.preview.src = e.target.result;
    elements.uploadZone.style.display = 'none';
    elements.previewContainer.style.display = 'flex';

    elements.preview.onload = () => {
      if (textModel && visionModel) {
        classifyImage(elements.preview);
      }
    };
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  elements.preview.src = '';
  elements.uploadZone.style.display = 'flex';
  elements.previewContainer.style.display = 'none';
  elements.resultsSection.style.display = 'none';
  elements.imageUpload.value = '';
}

// Event listeners
elements.uploadZone.addEventListener('click', () => elements.imageUpload.click());

elements.uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  elements.uploadZone.classList.add('dragover');
});

elements.uploadZone.addEventListener('dragleave', () => {
  elements.uploadZone.classList.remove('dragover');
});

elements.uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  elements.uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files.length) {
    handleFile(e.dataTransfer.files[0]);
  }
});

elements.imageUpload.addEventListener('change', (e) => {
  if (e.target.files.length) {
    handleFile(e.target.files[0]);
  }
});

elements.clearBtn.addEventListener('click', clearImage);

// Trigger re-classification if labels change and an image is already loaded
elements.labelInput.addEventListener('change', () => {
  if (elements.preview.src && elements.previewContainer.style.display !== 'none') {
    classifyImage(elements.preview);
  }
});

// Start initialization
initializeModel();