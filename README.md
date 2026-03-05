# Zero-Shot Image Classifier

A powerful AI-powered image classification tool that can identify objects in images without being pre-trained on specific categories. Uses state-of-the-art machine learning models to classify images with custom labels.

## 🌟 Features

- **Zero-Shot Learning**: Classify images with custom labels without pre-training
- **Drag & Drop Interface**: Easy image upload with drag and drop support
- **Real-Time Processing**: Instant classification results with confidence scores
- **Custom Labels**: Define your own classification categories
- **Confidence Meter**: Visual representation of classification confidence
- **Progress Tracking**: Real-time progress bar during model loading and processing
- **Mobile Friendly**: Responsive design that works on all devices

## 🚀 Usage

1. **Upload Image**: Drag and drop an image or click to select a file
2. **Set Labels**: Enter comma-separated labels (e.g., "cat, dog, car, bird")
3. **Wait for Processing**: The AI model will analyze your image
4. **View Results**: See classification results with confidence scores

## 🛠️ How It Works

This tool uses **zero-shot learning**, which means:
- No pre-training on specific image categories
- Can classify images into any category you define
- Uses advanced AI models to understand image content
- Provides confidence scores for each classification

## 🔧 Technical Details

- **Transformers.js**: Runs AI models directly in the browser
- **Zero-Shot Classification**: Uses CLIP or similar models
- **WebGL Acceleration**: GPU-accelerated processing when available
- **Progressive Loading**: Model loads in background (~15MB)
- **Modern JavaScript**: ES6+ features and async processing

## 📱 Browser Requirements

- **Modern Browser**: Chrome, Firefox, Safari, or Edge (latest versions)
- **JavaScript Enabled**: Required for AI model execution
- **WebGL Support**: Recommended for faster processing
- **Sufficient Memory**: ~50MB RAM recommended for model loading

## 🎯 Example Use Cases

- **Content Moderation**: Identify inappropriate content in images
- **Product Classification**: Categorize products in e-commerce
- **Animal Recognition**: Identify different types of animals
- **Object Detection**: Find specific objects in photos
- **Scene Understanding**: Classify different types of scenes

## 📊 Performance

- **Model Size**: ~15MB (downloads once, cached locally)
- **Processing Time**: 1-3 seconds per image (depends on device)
- **Accuracy**: High accuracy for common objects and scenes
- **Memory Usage**: Efficient memory management

## 📄 License

This project is open source and available under the MIT License.

## 🌐 Live Demo

Try the zero-shot image classifier: [Live Demo](https://melbinjp.github.io/image_classification/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 Tips for Best Results

- Use clear, well-lit images
- Provide specific and relevant labels
- Include multiple related labels for better classification
- Use high-resolution images when possible
- Ensure good internet connection for initial model download

## 🔮 Future Enhancements

- Support for video classification
- Batch processing of multiple images
- More advanced AI models
- Custom model training capabilities 