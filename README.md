# Product Fraud Detection System

An AI-powered CNN-based system for detecting product fraud and damage in online market delivery returns. The system uses deep learning to verify returned products against original purchases and assess their condition.

## 🎯 Features

- **Product Authentication**: Siamese Network compares returned products against database images
- **Damage Detection**: Multi-class CNN classifier assesses product condition
- **Database Integration**: Automatically retrieves original product images by product ID
- **Web Interface**: Modern, responsive UI with camera capture support
- **REST API**: Flask-based API for easy integration

## 🏗️ Architecture

### Models

1. **Siamese Network** - Product Authentication
   - Twin CNN branches with shared weights
   - Computes similarity between original and returned product images
   - Threshold: 85% similarity for authentic products

2. **Damage Detection CNN** - Condition Assessment
   - Transfer learning with MobileNetV2 base
   - 4 damage classes: No Damage, Minor, Major, Severely Damaged
   - Custom classification head with dropout for robustness

### Technology Stack

- **Backend**: Python, TensorFlow/Keras, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Deep Learning**: CNN, Siamese Networks, Transfer Learning
- **Image Processing**: PIL, OpenCV, NumPy

## 📦 Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd d:/e2e
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate sample data** (for testing)
   ```bash
   python scripts/generate_sample_data.py
   ```

4. **Train the models**
   ```bash
   # Full training (50 epochs)
   python models/train.py
   
   # Quick test (2 epochs)
   python models/train.py --test-mode
   ```

## 🚀 Usage

### 1. Start the API Server

```bash
python api/app.py
```

The server will start on `http://localhost:5000`

### 2. Open the Web Interface

Open `web/index.html` in your browser

### 3. Test the System

1. Enter a Product ID (e.g., `PROD001`)
2. Upload or capture an image of the returned product
3. Click "Analyze Product"
4. View the authentication and damage assessment results

## 📡 API Endpoints

### Health Check
```
GET /api/health
```

### Verify Product Authenticity
```
POST /api/verify-product
Body: 
  - product_id: string
  - returned_image: file
```

### Assess Damage
```
POST /api/assess-damage
Body:
  - product_image: file
```

### Full Analysis
```
POST /api/full-analysis
Body:
  - product_id: string
  - returned_image: file
```

## 🗂️ Project Structure

```
d:/e2e/
├── models/
│   ├── __init__.py
│   ├── model.py              # CNN architectures
│   ├── preprocessing.py      # Image preprocessing
│   └── train.py             # Training script
├── api/
│   ├── __init__.py
│   ├── app.py               # Flask API
│   └── utils.py             # API utilities
├── web/
│   ├── index.html           # Web interface
│   ├── style.css            # Styles
│   └── script.js            # Frontend logic
├── scripts/
│   └── generate_sample_data.py  # Sample data generator
├── data/
│   ├── original_products/   # Database of original product images
│   └── sample_data/         # Generated test data
├── saved_models/            # Trained model files
├── config.py                # Configuration
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🔧 Configuration

Edit `config.py` to customize:

- Image size and batch size
- Model paths
- Similarity threshold
- Training parameters
- API settings

## 📊 Training Your Own Models

### Using Real Data

1. **Organize your product images**:
   ```
   data/original_products/
   ├── PROD001.png
   ├── PROD002.jpg
   └── ...
   ```

2. **Prepare training data**:
   - Create pairs of (original, returned) images
   - Label them as authentic (1) or fraudulent (0)
   - Create damage-labeled images

3. **Modify `models/train.py`**:
   - Replace `generate_synthetic_training_data()` with your data loader
   - Adjust hyperparameters in `config.py`

4. **Train**:
   ```bash
   python models/train.py --epochs 100
   ```

## 🎨 Web Interface Features

- **Dual Input Methods**: Upload files or use camera
- **Real-time Preview**: See images before analysis
- **Detailed Results**: Authentication scores and damage breakdown
- **Color-coded Status**: Visual feedback for quick decisions
- **Responsive Design**: Works on desktop and mobile

## 🔍 How It Works

### Product Verification Flow

1. Customer initiates return with product ID
2. System retrieves original product image from database
3. Customer uploads/captures image of returned product
4. **Siamese Network** compares both images:
   - Similarity > 85% → Authentic
   - Similarity < 85% → Potential fraud
5. **Damage CNN** assesses condition:
   - No Damage → Full refund
   - Minor Damage → Partial refund
   - Major/Severe → Reject return
6. System provides recommendation

## 📈 Model Performance

The models are trained with:
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Data augmentation for robustness
- Transfer learning for better accuracy

Training metrics are saved as plots in `saved_models/`

## 🛠️ Troubleshooting

### Models not loading
```bash
# Train the models first
python models/train.py --test-mode
```

### API connection errors
- Ensure Flask server is running: `python api/app.py`
- Check the API URL in `web/script.js` matches your server

### Camera not working
- Grant camera permissions in browser
- Use HTTPS or localhost for camera access

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to enhance the system with:
- Additional damage categories
- Multi-angle product verification
- Barcode/QR code integration
- Database backend (PostgreSQL, MongoDB)
- User authentication

## 📧 Support

For issues or questions, please check:
1. Models are trained (`saved_models/` contains .h5 files)
2. API server is running
3. Original product images exist in `data/original_products/`
4. Dependencies are installed correctly

---

**Built with ❤️ using TensorFlow and Flask**
