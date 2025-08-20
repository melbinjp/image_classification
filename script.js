import {
    AutoTokenizer,
    AutoProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    AutoModelForVision2Seq,
    AutoModelForImageClassification,
} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@latest/dist/transformers.min.js';

// --- Model Names ---
const ZERO_SHOT_MODEL_NAME = 'Xenova/clip-vit-base-patch16';
const CAPTIONING_MODEL_NAME = 'Xenova/vit-gpt2-image-captioning';
const CLASSIFICATION_MODEL_NAME = 'Xenova/google/vit-base-patch16-224';

// --- Model Variables ---
let textModel, visionModel, zeroShotTokenizer, zeroShotProcessor, textEmbeddings, cachedLabels;
let captioningModel, captioningProcessor;
let classificationModel, classificationProcessor;

// --- UI Elements ---
const elements = {
    uploadZone: document.getElementById('dropZone'),
    imageUpload: document.getElementById('imageUpload'),
    preview: document.getElementById('preview'),
    modelStatus: document.querySelector('.model-status'),
    labelInput: document.querySelector('.label-input'),
    confidenceMeter: document.querySelector('.confidence-meter'),
    describeBtn: document.getElementById('describe-btn'),
    classifyBtn: document.getElementById('classify-btn'),
    descriptionResult: document.getElementById('description-result'),
    classificationResult: document.getElementById('classification-result'),
};

// --- Model Initialization ---
async function initializeModel() {
    elements.modelStatus.innerHTML = `<span><i class="fas fa-circle-notch fa-spin"></i> Loading models...</span>`;
    try {
        const updateProgress = (modelName, progress) => {
            const percent = Math.round(progress.progress * 100);
            elements.modelStatus.innerHTML = `<span><i class="fas fa-circle-notch fa-spin"></i> Loading ${modelName}... ${percent}%</span>`;
        };

        [textModel, visionModel, zeroShotTokenizer, zeroShotProcessor, captioningModel, captioningProcessor, classificationModel, classificationProcessor] = await Promise.all([
            CLIPTextModelWithProjection.from_pretrained(ZERO_SHOT_MODEL_NAME, { quantized: true, progress_callback: p => updateProgress('Zero-Shot Text', p) }),
            CLIPVisionModelWithProjection.from_pretrained(ZERO_SHOT_MODEL_NAME, { quantized: true, progress_callback: p => updateProgress('Zero-Shot Vision', p) }),
            AutoTokenizer.from_pretrained(ZERO_SHOT_MODEL_NAME),
            AutoProcessor.from_pretrained(ZERO_SHOT_MODEL_NAME),
            AutoModelForVision2Seq.from_pretrained(CAPTIONING_MODEL_NAME, { quantized: true, progress_callback: p => updateProgress('Captioning Model', p) }),
            AutoProcessor.from_pretrained(CAPTIONING_MODEL_NAME),
            AutoModelForImageClassification.from_pretrained(CLASSIFICATION_MODEL_NAME, { quantized: true, progress_callback: p => updateProgress('Classification Model', p) }),
            AutoProcessor.from_pretrained(CLASSIFICATION_MODEL_NAME),
        ]);

        elements.modelStatus.innerHTML = `<span><i class="fas fa-check-circle" style="color: #10b981"></i> Models ready!</span>`;
        elements.describeBtn.disabled = false;
        elements.classifyBtn.disabled = false;

    } catch (error) {
        elements.modelStatus.innerHTML = `<span><i class="fas fa-times-circle" style="color: #ef4444"></i> Models failed to load</span>`;
        console.error('Model loading error:', error);
    }
}

// --- Image Description ---
async function describeImage() {
    if (!elements.preview.src || !captioningModel) {
        elements.descriptionResult.innerHTML = 'Please upload an image first.';
        return;
    }
    elements.descriptionResult.innerHTML = 'Generating description...';
    try {
        const image = elements.preview;
        const pixel_values = (await captioningProcessor(image)).pixel_values;
        const outputs = await captioningModel.generate(pixel_values, { max_length: 16 });
        const description = captioningProcessor.decode(outputs[0], { skip_special_tokens: true });
        elements.descriptionResult.innerHTML = description;
    } catch (error) {
        console.error('Description error:', error);
        elements.descriptionResult.innerHTML = `Error: ${error.message}`;
    }
}

// --- Direct Image Classification ---
async function classifyImage() {
    if (!elements.preview.src || !classificationModel) {
        elements.classificationResult.innerHTML = 'Please upload an image first.';
        return;
    }
    elements.classificationResult.innerHTML = 'Classifying...';
    try {
        const image = elements.preview;
        const inputs = await classificationProcessor(image, { return_tensors: "pt" });
        const outputs = await classificationModel(inputs);
        const results = outputs.logits.softmax().topk(5);
        const topClasses = results.map((result, i) => ({
            label: classificationModel.config.id2label[results.indices[i]],
            score: Math.round(result * 100),
        }));
        displayClassificationResults(topClasses);
    } catch (error) {
        console.error('Classification error:', error);
        elements.classificationResult.innerHTML = `Error: ${error.message}`;
    }
}

function displayClassificationResults(results) {
    elements.classificationResult.innerHTML = '';
    results.forEach(({ label, score }) => {
        const item = document.createElement('div');
        item.className = 'confidence-item';
        item.innerHTML = `
            <span>${label.replaceAll('_', ' ')}</span>
            <div class="confidence-bar-container">
                <div class="confidence-bar" style="width: ${score}%;"></div>
            </div>
            <span>${score}%</span>`;
        elements.classificationResult.appendChild(item);
    });
}


// --- Zero-Shot Classification ---
async function processLabels(labels) {
    if (textEmbeddings) textEmbeddings.dispose();
    const textInputs = await zeroShotTokenizer(labels, { padding: true, truncation: true, return_tensors: 'pt' });
    const outputs = await textModel(textInputs);
    const tempEmbeddings = outputs.text_embeds;
    const norms = tempEmbeddings.norm({ p: 2, dim: -1, keepdim: true });
    textEmbeddings = tempEmbeddings.div(norms);
    tempEmbeddings.dispose();
    cachedLabels = labels.join(',');
}

async function zeroShotClassifyImage() {
    if (!elements.preview.src || !visionModel) {
        elements.confidenceMeter.innerHTML = 'Please upload an image first.';
        return;
    }
    const labels = elements.labelInput.value.split(',').map(s => s.trim()).filter(s => s);
    if (labels.length === 0) {
        elements.confidenceMeter.innerHTML = 'Please enter some labels to classify.';
        return;
    }
    elements.confidenceMeter.innerHTML = 'Classifying...';
    let imageEmbedding, similarity;
    try {
        if (!textEmbeddings || labels.join(',') !== cachedLabels) {
            await processLabels(labels);
        }
        const imageInputs = await zeroShotProcessor(elements.preview, { return_tensors: 'pt' });
        const imageOutputs = await visionModel(imageInputs);
        imageEmbedding = imageOutputs.image_embeds;
        const imageNorms = imageEmbedding.norm({ p: 2, dim: -1, keepdim: true });
        const normalizedImage = imageEmbedding.div(imageNorms);
        similarity = normalizedImage.matMul(textEmbeddings.transpose(0, 1));
        const scores = await similarity.data();
        const confidences = labels.map((label, i) => ({
            label: label,
            confidence: Math.round(scores[i] * 100),
        }));
        confidences.sort((a, b) => b.confidence - a.confidence);
        displayZeroShotResults(confidences);
    } catch (error) {
        console.error('Zero-shot classification error:', error);
        elements.confidenceMeter.innerHTML = `Error: ${error.message}`;
    } finally {
        if (imageEmbedding) imageEmbedding.dispose();
        if (similarity) similarity.dispose();
    }
}

function displayZeroShotResults(confidences) {
    elements.confidenceMeter.innerHTML = '';
    confidences.forEach(({ label, confidence }) => {
        const item = document.createElement('div');
        item.className = 'confidence-item';
        item.innerHTML = `
            <span>${label}</span>
            <div class="confidence-bar-container">
                <div class="confidence-bar" style="width: ${confidence}%;"></div>
            </div>
            <span>${confidence}%</span>`;
        elements.confidenceMeter.appendChild(item);
    });
}

// --- Event Handlers ---
function handleImageUpload(file) {
    if (file && file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.preview.src = e.target.result;
            elements.preview.style.display = 'block';
            elements.descriptionResult.innerHTML = '';
            elements.classificationResult.innerHTML = '';
            elements.confidenceMeter.innerHTML = '';
        };
        reader.readAsDataURL(file);
    }
}

elements.uploadZone.addEventListener('click', () => elements.imageUpload.click());
elements.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elements.uploadZone.style.borderColor = 'var(--primary)';
});
elements.uploadZone.addEventListener('dragleave', () => {
    elements.uploadZone.style.borderColor = '#cbd5e1';
});
elements.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    elements.uploadZone.style.borderColor = '#cbd5e1';
    handleImageUpload(e.dataTransfer.files[0]);
});
elements.imageUpload.addEventListener('change', (e) => handleImageUpload(e.target.files[0]));

elements.describeBtn.addEventListener('click', describeImage);
elements.classifyBtn.addEventListener('click', classifyImage);
elements.labelInput.addEventListener('keyup', (e) => {
    if (e.key === 'Enter') {
        zeroShotClassifyImage();
    }
});

// --- Initial State ---
elements.describeBtn.disabled = true;
elements.classifyBtn.disabled = true;
initializeModel();