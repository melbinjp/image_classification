import {
    AutoTokenizer,
    AutoProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
  } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@latest/dist/transformers.min.js';
  
  const MODEL_NAME = 'Xenova/clip-vit-base-patch16';
  let textModel, visionModel, tokenizer, processor, textEmbeddings, cachedLabels;
  
  // UI Elements
  const elements = {
    uploadZone: document.getElementById('dropZone'),
    preview: document.getElementById('preview'),
    result: document.getElementById('result'),
    progress: document.querySelector('.progress'),
    modelStatus: document.querySelector('.model-status'),
    confidenceMeter: document.querySelector('.confidence-meter'),
    labelInput: document.querySelector('.label-input'),
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
  
      elements.modelStatus.innerHTML =
        `<i class="fas fa-check-circle" style="color: #10b981"></i> Model ready!`;
    } catch (error) {
      elements.modelStatus.innerHTML =
        `<i class="fas fa-times-circle" style="color: #ef4444"></i> Model failed to load`;
      console.error('Model loading error:', error);
    }
  }
  
  // Progress callback function
  function downloadProgress(progress) {
    const percent = Math.round(progress * 100);
    elements.modelStatus.innerHTML = `Loading model... ${percent}%`;
  }
  
  // Process text labels
  async function processLabels(labels) {
    if (textEmbeddings) {
      textEmbeddings.dispose();
    }
  
    // Tokenize text inputs
    const textInputs = await tokenizer(labels, {
      padding: true,
      truncation: true,
      return_tensors: true,
    });
  
    // Get text embeddings from the text model
    const outputs = await textModel(textInputs);
  
    const tempEmbeddings = outputs.text_embeds;
  
    // Normalize embeddings
    const norms = tempEmbeddings.norm({ dim: -1, keepdim: true });
    textEmbeddings = tempEmbeddings.div(norms);
    tempEmbeddings.dispose();
    cachedLabels = labels.join(',');
  }
  
  // Classify image
  async function classifyImage(imageElement) {
    let imageEmbedding, similarity;
  
    try {
      const labels = elements.labelInput.value.split(',').map((s) => s.trim());
  
      // Check if labels have changed or textEmbeddings is uninitialized
      if (!textEmbeddings || labels.join(',') !== cachedLabels) {
        await processLabels(labels);
      }
  
      // Use processor to process image
      const imageInputs = await processor(imageElement, {
        return_tensors: true,
      });
  
      // Get image embeddings from the vision model
      const imageOutputs = await visionModel(imageInputs);
  
      imageEmbedding = imageOutputs.image_embeds;
  
      // Normalize image embeddings
      const imageNorms = imageEmbedding.norm({ dim: -1, keepdim: true });
      const normalizedImage = imageEmbedding.div(imageNorms);
  
      // Compute similarity between image and text embeddings
      similarity = normalizedImage.matmul(textEmbeddings.transpose());
  
      const results = await similarity.data();
  
      const confidences = labels.map((label, i) => ({
        label: label,
        confidence: Math.round(results[i] * 100),
      }));
  
      confidences.sort((a, b) => b.confidence - a.confidence);
  
      // Display results
      displayResults(
        confidences.map((c) => c.label),
        confidences.map((c) => c.confidence / 100),
      );
    } catch (error) {
      console.error('Classification error:', error);
      elements.result.innerHTML = `Error: ${error.message}`;
    } finally {
      // Cleanup tensors
      if (imageEmbedding) imageEmbedding.dispose();
      if (similarity) similarity.dispose();
    }
  }
  
  // Display results (remains the same)
  function displayResults(labels, scores) {
    // Your existing code
  }
  
  // Event listeners (remain the same)
  
  // Initialize the model
  initializeModel();