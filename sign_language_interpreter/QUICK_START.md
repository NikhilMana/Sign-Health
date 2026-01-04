# 🚀 Quick Start Guide - ISL Interpreter

## Current Status
✅ MP_DATA deleted - ready for fresh training
✅ Optimized code in place
✅ Videos folder ready (if you have labeled videos)

---

## 📋 Step-by-Step Instructions

### **Option 1: Automated Pipeline (Recommended)**

Run everything automatically:

```bash
cd sign_language_interpreter
python run_full_pipeline.py
```

This will:
1. Extract features from all videos in `videos/` folder
2. Train the optimized LSTM model
3. Save the trained model to `models/`

**Time estimate:** 20-60 minutes depending on dataset size

---

### **Option 2: Manual Step-by-Step**

#### Step 1: Feature Extraction
```bash
cd sign_language_interpreter/data_collection
python process_videos.py
```

**What it does:**
- Processes all videos in `videos/` folder
- Extracts MediaPipe keypoints (pose + hands)
- Saves to `MP_DATA/` as .npy files
- Uses multiprocessing for speed

**Expected output:**
```
Found 200 sign labels
Processing 1500 videos...
[1/1500] ✓ hello/0.mp4: Saved 45 frames
[2/1500] ✓ thank_you/0.mp4: Saved 38 frames
...
```

---

#### Step 2: Model Training
```bash
cd sign_language_interpreter/model_training
python train_model.py
```

**What it does:**
- Loads data from `MP_DATA/`
- Applies data augmentation (if < 500 samples)
- Trains Bidirectional LSTM model
- Saves model to `models/sign_model.keras`

**Expected output:**
```
Found 200 sign labels
Total sequences loaded: 1500
Augmented from 1500 to 4500 sequences
Training...
Epoch 1/300: loss: 4.2 - accuracy: 0.15 - val_accuracy: 0.20
...
Epoch 50/300: loss: 0.3 - accuracy: 0.92 - val_accuracy: 0.85
Test Accuracy: 0.8500
```

---

#### Step 3: Test Real-Time Detection
```bash
cd sign_language_interpreter/real_time_inference
python detect_live.py
```

**What it does:**
- Opens your webcam
- Detects ISL signs in real-time
- Shows predictions with confidence scores

**Controls:**
- Press `q` to quit

---

## 🎯 What You Should See

### After Feature Extraction:
```
MP_DATA/
├── hello/
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
├── thank_you/
│   ├── 0.npy
│   └── ...
└── ...
```

### After Training:
```
models/
├── sign_model.keras          # Main model
├── best_model.keras          # Best checkpoint
├── label_encoder.pkl         # Label mappings
├── max_length.txt            # Sequence length
└── training_history.png      # Training curves
```

---

## ⚠️ Troubleshooting

### "No videos found"
- Make sure you have videos in `videos/` folder
- Each sign should have its own subfolder
- Run `auto_labeler.py` first if you haven't

### "Out of memory"
- Reduce `MAX_SEQ_LENGTH_CAP` in train_model.py
- Close other applications
- Process fewer videos at once

### "Low accuracy"
- Need more training data (aim for 10+ samples per sign)
- Ensure good video quality (clear signing, good lighting)
- Let training run longer (EarlyStopping will find optimal point)

### "Slow real-time detection"
- Lower webcam resolution in detect_live.py
- Reduce MediaPipe model_complexity to 0
- Close other applications

---

## 📊 Expected Performance

| Dataset Size | Feature Extraction | Training Time | Expected Accuracy |
|--------------|-------------------|---------------|-------------------|
| 500 videos   | 5-10 min          | 15-30 min     | 70-80%           |
| 1500 videos  | 15-25 min         | 30-60 min     | 80-90%           |
| 3000+ videos | 30-50 min         | 60-120 min    | 85-95%           |

---

## 🎉 Next Steps After Training

1. **Check training curves**: Open `models/training_history.png`
2. **Test with webcam**: Run `detect_live.py`
3. **Collect more data**: Add more videos for low-accuracy signs
4. **Fine-tune**: Adjust hyperparameters in train_model.py

---

## 💡 Pro Tips

- **More data = better accuracy**: Aim for 10-20 samples per sign
- **Diverse signers**: Different people, lighting, backgrounds
- **Quality over quantity**: Clear signing is more important than many samples
- **Monitor training**: Watch for overfitting (val_loss increases while train_loss decreases)
- **Use augmentation**: Enabled automatically for small datasets

---

## 🆘 Need Help?

Check the optimization summary:
```bash
cat OPTIMIZATION_SUMMARY.md
```

Happy signing! 🤟
