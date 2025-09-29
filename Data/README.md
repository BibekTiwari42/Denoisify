# Data Directory Structure

This directory should contain the training and testing datasets for the audio denoising model.

## Directory Structure

```
Data/
├── train/
│   ├── clean_trainset_28spk/
│   │   └── clean_trainset_28spk_wav/
│   │       └── *.wav files (clean audio samples)
│   └── noisy_trainset_28spk/
│       └── noisy_trainset_28spk_wav/
│           └── *.wav files (corresponding noisy audio samples)
└── test/
    ├── clean_testset/
    │   └── *.wav files (clean test audio samples)
    └── noisy_testset/
        └── *.wav files (corresponding noisy test audio samples)
```

## Dataset Requirements

### Training Data (`train/`)

- **clean_trainset_28spk/clean_trainset_28spk_wav/**: Contains clean audio files for training
- **noisy_trainset_28spk/noisy_trainset_28spk_wav/**: Contains corresponding noisy versions of the same audio files
- File naming should match between clean and noisy directories
- Recommended format: 16-bit WAV files at 16kHz sample rate

### Test Data (`test/`)

- **clean_testset/**: Contains clean audio files for evaluation
- **noisy_testset/**: Contains corresponding noisy versions for testing
- Used for model evaluation and performance metrics

## How to Setup

1. **Download Dataset**: Obtain a suitable audio denoising dataset (e.g., DNS Challenge dataset, VCTK + noise)

2. **Organize Files**:

   - Place clean training audio in `train/clean_trainset_28spk/clean_trainset_28spk_wav/`
   - Place noisy training audio in `train/noisy_trainset_28spk/noisy_trainset_28spk_wav/`
   - Place clean test audio in `test/clean_testset/`
   - Place noisy test audio in `test/noisy_testset/`

3. **File Format**: Ensure all audio files are in WAV format, preferably:

   - Sample rate: 16kHz
   - Bit depth: 16-bit
   - Channels: Mono (1 channel)

4. **Naming Convention**: Clean and noisy files should have matching names
   - Example: `clean_audio_001.wav` ↔ `noisy_audio_001.wav`

## Dataset Used for Training

This model was trained using a **28-speaker dataset** with the following characteristics:

### **Training Dataset Specifications:**

- **Dataset Name**: 28-Speaker Clean/Noisy Training Set
- **Speakers**: 28 different speakers (male and female voices)
- **Audio Format**: WAV files, 16-bit, 16kHz sample rate, mono channel
- **Duration**: Multiple hours of paired clean/noisy audio data
- **Language**: English speech corpus
- **Content**: Natural speech recordings with various acoustic conditions

### **Dataset Structure:**

- **Clean Training Data**: `clean_trainset_28spk_wav/` - High-quality, noise-free speech recordings
- **Noisy Training Data**: `noisy_trainset_28spk_wav/` - Corresponding noisy versions with added background noise, reverb, and distortions
- **Test Data**: Separate clean and noisy test sets for model evaluation

### **Noise Characteristics:**

The noisy training data includes various types of acoustic interference:

- Background noise (office, street, cafe environments)
- Reverb and echo effects
- Low SNR conditions (-5dB to 20dB)
- Multiple noise sources and acoustic scenarios

### **Model Training Details:**

- **Sample Rate**: 16kHz (model processes audio at this rate)
- **Segment Length**: 16384 samples (~1 second segments)
- **Training Epochs**: 30 epochs with early stopping
- **Architecture**: WaveUNet (depth=5, base_channels=24)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=1e-4)

## Alternative Datasets for Retraining

If you want to retrain the model with different data:

- **DNS Challenge Dataset**: Microsoft's DNS Challenge provides high-quality clean speech and diverse noise samples
- **VCTK Corpus + Background Noise**: Clean speech corpus with added synthetic noise
- **LibriSpeech + Noise**: Large-scale English speech corpus with artificial noise addition
- **Custom Dataset**: Record your own clean audio and add noise programmatically

## Notes

- Audio files are not included in the repository due to size constraints
- Total dataset size typically ranges from 10-100GB depending on the corpus
- Ensure you have proper licensing rights to use any downloaded datasets
- For training, you typically need several hours of paired clean/noisy audio data

## Training Configuration

Update the paths in `model_training/train.py` to match your data structure:

```python
CONFIG = {
    "train_noisy_dir": "Data/train/noisy_trainset_28spk/noisy_trainset_28spk_wav",
    "train_clean_dir": "Data/train/clean_trainset_28spk/clean_trainset_28spk_wav",
    "valid_noisy_dir": "Data/test/noisy_testset",
    "valid_clean_dir": "Data/test/clean_testset",
    # ... other config options
}
```
