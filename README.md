# EEG Numerical Cognition Decoding Project

This project aims to use deep learning to decode cognitive states from EEG data related to numerical processing.

## Project Setup

This project uses a Conda environment to manage dependencies.

### 1. Create and Activate the Conda Environment

First, ensure you have Anaconda or Miniconda installed. Then, create the environment using the following command:

```bash
conda create --name torcheeg-env python=3.11 -y
```

Activate the environment:

```bash
conda activate torcheeg-env
```

### 2. Install Dependencies

Once the environment is activated, install the required packages using pip:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torcheeg
pip install mne scikit-learn matplotlib pandas
```

### 3. Running the Analysis

The analysis scripts are located in the `code/` directory.

1.  `01_prepare_for_nn.py`: Preprocesses the raw data. (This has already been run).
2.  `02_train_torcheeg_decoder.py`: Trains the TorchEEG model.

To run the training script:

```bash
python code/02_train_torcheeg_decoder.py
``` 