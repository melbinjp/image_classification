// Import TensorFlow.js and the Transformers.js CLIP model & processor.
import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.esm.js';
import { CLIPModel, CLIPProcessor } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@latest/dist/transformers.js';

// Use a quantized version for smaller model size.
const MODEL_NAME = 'Xenova/clip-vit-base-patch16';
const quantized = true;

let model, processor;

// Define candidate textual descriptions for zero-shot classification.
const labels = ["a cat", "a dog", "a car"];
let textEmbeddings; // To hold pre-computed text embeddings

// Load the CLIP model and processor.
async function loadModel() {
  model = await CLIPModel.from_pretrained(MODEL_NAME, { quantized });
  processor = await CLIPProcessor.from_pretrained(MODEL_NAME);
  console.log('CLIP model and processor loaded.');

  // Pre-compute text embeddings.
  await computeTextEmbeddings();
}
  
// Compute normalized text embeddings for the candidate labels.
async function computeTextEmbeddings() {
  // Process the text labels. The processor automatically tokenizes them.
  const textInputs = await processor({ text: labels });
  // Use the model's text encoder to obtain embeddings.
  textEmbeddings = model.get_text_features(textInputs);
  // Normalize each embedding (divide by its L2 norm).
  textEmbeddings = tf.div(textEmbeddings, tf.norm(textEmbeddings, 'euclidean', 1, true));
  console.log('Text embeddings computed.');
}

loadModel();

// Get references to UI elements.
const imageUpload = document.getElementById('imageUpload');
const preview = document.getElementById('preview');
const classifyButton = document.getElementById('classifyButton');
const resultElem = document.getElementById('result');

let uploadedImage; // Will store the uploaded image

// When the user selects an image, display a preview.
imageUpload.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
  uploadedImage = new Image();
  uploadedImage.src = url;
  uploadedImage.onload = () => {
    console.log('Image loaded.');
  };
});

// When the classify button is clicked, process the image and classify.
classifyButton.addEventListener('click', async () => {
  if (!model || !processor || !textEmbeddings) {
    resultElem.innerText = 'Model is still loading. Please wait...';
    return;
  }
  if (!uploadedImage) {
    resultElem.innerText = 'Please upload an image first.';
    return;
  }
  resultElem.innerText = 'Processing image...';

  // Process the image using the processor.
  // The processor accepts an HTMLImageElement and returns a tensor (or a dict of tensors)
  const imageInputs = await processor({ image: uploadedImage });
  
  // Compute image features using the vision encoder.
  let imageEmbedding = model.get_image_features(imageInputs);
  // Normalize the image embedding.
  imageEmbedding = tf.div(imageEmbedding, tf.norm(imageEmbedding, 'euclidean', 1, true));

  // Compute cosine similarities between image embedding and each text embedding.
  // imageEmbedding shape: [1, dim], textEmbeddings shape: [num_labels, dim]
  const similarities = tf.matMul(imageEmbedding, textEmbeddings, false, true);
  const similarityArray = await similarities.data();

  // Find the label with the highest similarity.
  let maxIndex = 0;
  for (let i = 1; i < similarityArray.length; i++) {
    if (similarityArray[i] > similarityArray[maxIndex]) {
      maxIndex = i;
    }
  }
  resultElem.innerText = `Predicted: ${labels[maxIndex]}\nSimilarities: ${JSON.stringify(similarityArray, null, 2)}`;

  // Dispose tensors to free memory.
  imageEmbedding.dispose();
  similarities.dispose();
});
