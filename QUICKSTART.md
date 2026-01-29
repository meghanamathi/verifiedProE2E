# Quick Start Guide - Product Fraud Detection System

## ⚡ Fast Setup (5 minutes)

### 1. Install Dependencies
```bash
cd d:/e2e
pip install tensorflow numpy pillow flask flask-cors matplotlib scikit-learn opencv-python
```

### 2. Generate Sample Data
```bash
python scripts/generate_sample_data.py
```
This creates 5 sample products with test images.

### 3. Train Models (Quick Test)
```bash
python models/train.py --test-mode
```
Trains both models in ~2 minutes (2 epochs only).

### 4. Start API Server
```bash
python api/app.py
```
Server runs on http://localhost:5000

### 5. Open Web Interface
Open `web/index.html` in your browser.

## 🧪 Test the System

1. **Product ID**: Enter `PROD001`
2. **Image**: Upload from `data/sample_data/authentic_returns/PROD001_return.png`
3. **Click**: "Analyze Product"
4. **Result**: Should show ✓ Authentic + No Damage

### More Test Cases

| Product ID | Image Path | Expected Result |
|------------|-----------|-----------------|
| PROD001 | `sample_data/authentic_returns/PROD001_return.png` | ✓ Authentic, No Damage |
| PROD001 | `sample_data/fraudulent_returns/PROD001_fraud.png` | ✗ Not Authentic |
| PROD002 | `sample_data/damaged_products/PROD002_minor.png` | ✓ Authentic, Minor Damage |
| PROD003 | `sample_data/damaged_products/PROD003_major.png` | ✓ Authentic, Major Damage |

## 🔧 Troubleshooting

### "Models not loaded"
```bash
python models/train.py --test-mode
```

### "Product image not found"
```bash
python scripts/generate_sample_data.py
```

### "API not reachable"
Make sure Flask server is running:
```bash
python api/app.py
```

## 📱 Using Camera

1. Click "📷 Use Camera" button
2. Allow camera permissions
3. Point at product
4. Click "📸 Capture"
5. Enter product ID
6. Analyze!

## 🎯 Production Deployment

### Replace Sample Data with Real Products

1. Add product images to `data/original_products/`:
   ```
   data/original_products/
   ├── PROD12345.jpg
   ├── PROD67890.png
   └── ...
   ```

2. Train with real data:
   - Collect pairs of (original, returned) images
   - Label as authentic (1) or fraud (0)
   - Collect damaged product images
   - Modify `models/train.py` to load your data

3. Deploy API:
   ```bash
   # Production mode
   gunicorn -w 4 -b 0.0.0.0:5000 api.app:app
   ```

## 💡 Key Features

- ✅ **Automatic Database Lookup**: Customer only uploads return image
- ✅ **Dual Analysis**: Authentication + Damage in one request
- ✅ **Smart Recommendations**: Color-coded decisions
- ✅ **Camera Support**: Mobile-friendly capture
- ✅ **Premium UI**: Dark mode, animations, responsive

---

**Need help?** Check [README.md](file:///d:/e2e/README.md) for full documentation.
