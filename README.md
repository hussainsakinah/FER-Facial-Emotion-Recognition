# Facial Expression Recognition
## VGG + CBAM Attention + Focal Loss · FER2013

Recognises 7 facial emotions from photos using a deep learning
pipeline trained on FER2013.

---

## Architecture

| Component | Detail |
|-----------|--------|
| Backbone | VGG-style · 4 blocks (64→128→256→512 filters) |
| Attention | CBAM after each block (Channel + Spatial) |
| Loss | Focal Loss γ=2.0, α=0.25 |
| Preprocessing | dlib 68-pt face alignment |
| Input | 48×48 grayscale |
| Classes | 7 (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) |

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **67.38%** |
| Dataset | FER2013 (28,709 train / 3,589 test) |

## How to Use

### Run inference on your own photo

1. Open **FER_Inference.ipynb** in Colab
2. Run all cells top to bottom (takes ~3 min first time)
3. In Cell 4, upload any face photo to get a prediction

### Use the web frontend

1. Open **FER_Inference.ipynb** and run the Flask backend cell
2. Copy the ngrok URL that appears
3. Open the [live demo](https://hussainsakinah2611.github.io/FER-VGG-CBAM-FocalLoss/)
4. Paste the URL into the connection bar and click Test
5. Upload any photo

### Run locally

```bash
git clone https://github.com/hussainsakinah2611/FER-VGG-CBAM-FocalLoss.git
cd FER-VGG-CBAM-FocalLoss
python serve.py      # opens http://localhost:8000
```

## Dataset

[FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

## Requirements

See `requirements.txt`. Training was done on Google Colab with T4 GPU.

---

*Built with TensorFlow · OpenCV · dlib · Flask*
