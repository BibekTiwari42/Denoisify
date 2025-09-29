# Denoisify - Audio Denoising System

AN audio denoising system that combines WaveUNet neural network architecture with MMSE-STSA as post-processing for superior noise reduction in speech signals.

## Features

- **Hybrid Denoising**: Combines WaveUNet architecture with SSBSE post-processing
- **Multiple Processing Modes**:
  - WaveUNet + SSBSE (recommended)
  - SSBSE-only processing
- **Web Interface**: Django-based web application for easy audio processing
- **Batch Processing**: Support for processing multiple audio files
- **Model Training**: Complete training pipeline with evaluation metrics
- **GPU Support**: CUDA acceleration for faster processing

## Architecture

### Core Components

1. **WaveUNet Model** (`model_training/model.py`)

   - U-Net architecture adapted for 1D audio signals
   - Encoder-decoder structure with skip connections
   - Configurable depth and channel dimensions

2. **SSBSE Post-Processing** (`model_training/postprocess.py`)

   - Spectral Subtraction with Bispectral Spectral Enhancement
   - Advanced noise reduction technique
   - Can be used standalone or with WaveUNet

3. **Django Web Interface** (`denoiser/`)
   - User-friendly web interface
   - File upload and download functionality - Real-time processing status

## Requirements

### System Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- 4GB+ RAM (8GB+ recommended)
- Storage space for audio datasets

### Python Dependencies

```
torch>=1.9.0
torchaudio>=0.9.0
django>=3.2.0
numpy>=1.21.0
scipy>=1.7.0
librosa>=0.8.0
soundfile>=0.10.0
matplotlib>=3.3.0
tqdm>=4.62.0
```

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/denoisify.git
   cd denoisify
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Django database**

   ```bash
   python manage.py migrate
   ```

5. **Download pre-trained model** (optional)
   ```bash
   # Place your trained model in model_training/checkpoints/   # Or use the provided best_model.pth
   ```

## Dataset Structure

Organize your audio data as follows:

```
Data/
├── train/
│   ├── clean_trainset_28spk/     # Clean training audio
│   └── noisy_trainset_28spk/     # Noisy training audio
└── test/
    ├── clean_testset/            # Clean test audio
    └── noisy_testset/            # Noisy test audio
```

### Supported Audio Formats

- WAV (16-bit, 48kHz recommended)
- Sample rate: 16kHz - 48kHz
- Channels: Mono (single channel)

## Usage

### Web Interface

1. **Start the Django server**

   ```bash
   python manage.py runserver
   ```

2. **Open browser** and navigate to `http://localhost:8000`

3. **Upload audio file** through the web interface

4. **Select processing mode**:

   - WaveUNet + SSBSE (recommended)

5. **Download processed audio**

## Model Training

### Prepare Dataset

1. Organize your training data in the `Data/` directory
2. Ensure clean and noisy audio pairs are properly aligned

### Training Process

```bash
cd model_training
python train.py --epochs 100 --batch_size 16 --lr 0.001
```

### Training Parameters

- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate (default: 0.001)
- `--device`: Training device ('cuda' or 'cpu')
- `--checkpoint_dir`: Directory to save model checkpoints

### Evaluation

```bash
python evaluate.py --model_path checkpoints/best_model.pth
```

## Performance Metrics

The system is evaluated using standard audio quality metrics:

- **PESQ** (Perceptual Evaluation of Speech Quality)
- **STOI** (Short-Time Objective Intelligibility)
- **SNR** (Signal-to-Noise Ratio)
- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio)

### Typical Results

| Method           | PESQ | STOI | SNR (dB) |
| ---------------- | ---- | ---- | -------- |
| Noisy Input      | 1.97 | 0.91 | 0.0      |
| SSBSE Only       | 2.45 | 0.94 | 8.2      |
| WaveUNet + SSBSE | 2.89 | 0.96 | 12.1     |

## Configuration

### Model Configuration

Edit `model_training/model.py` to adjust:

- Network depth
- Base channel count
- Kernel sizes
- Activation functions

## Project Structure

```
denoisify/
├── Data/                          # Training and test datasets
├── backend_django/                # Django project settings
├── denoiser/                      # Django app for web interface
│   ├── templates/
│   ├── views.py
│   ├── models.py
│   └── urls.py
├── model_training/                # Model training and evaluation
│   ├── model.py                   # WaveUNet architecture
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   ├── dataset.py                 # Dataset loader
│   ├── postprocess.py             # SSBSE post-processing
│   └── checkpoints/               # Saved model weights
├── media/                         # Processed audio outputs
├── manage.py                      # Django management
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Technical Details

### WaveUNet Architecture

- **Input**: Raw audio waveform (1D signal)
- **Encoder**: Downsampling with strided convolutions
- **Decoder**: Upsampling with transposed convolutions
- **Skip Connections**: Preserve high-frequency details
- **Output**: Denoised audio waveform

### SSBSE Post-Processing

- **Spectral Analysis**: STFT-based frequency domain processing
- **Bispectral Enhancement**: Phase information preservation
- **Noise Estimation**: Adaptive noise floor estimation
- **Spectral Subtraction**: Selective noise reduction

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
