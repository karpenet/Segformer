# Segformer Training Pipeline

This repository contains the implementation of a Segformer model for semantic segmentation using PyTorch. The training pipeline is designed to work with the ADE20K dataset and uses Weights & Biases (wandb) for experiment tracking.

## Prerequisites

- Python 3.8 or higher
- CUDA (if using GPU)

## Setup

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the required packages:**

   Ensure you have `pip` installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Weights & Biases (wandb):**

   - Create an account on [wandb.ai](https://wandb.ai/).
   - Install the wandb CLI tool if not already installed:

     ```bash
     pip install wandb
     ```

   - Log in to wandb:

     ```bash
     wandb login
     ```

   - Optionally, you can set up a default project and entity in your wandb configuration file (`~/.config/wandb/settings`) or by using environment variables:

     ```bash
     export WANDB_PROJECT=<your-project-name>
     export WANDB_ENTITY=<your-entity-name>
     ```

## Training

1. **Prepare the configuration file:**

   The training configuration is stored in `train_config.yaml`. You can modify the hyperparameters such as `batch_size`, `epochs`, `learning_rate`, and model architecture (`arch`) as needed.

2. **Run the training script:**

   Execute the following command to start training:

   ```bash
   python train.py
   ```

   This will initialize a wandb run, load the dataset, and start the training process. The model's performance metrics will be logged to your wandb project.

## Code Structure

- `train.py`: Contains the main training loop and functions for logging metrics.
- `dataset.py`: Handles dataset downloading, preprocessing, and loading.
- `Segformer.py`: Defines the Segformer model architecture.
- `train_config.yaml`: Configuration file for training hyperparameters and model architecture.

## Logging and Monitoring

- Training and validation metrics are logged to wandb. You can monitor the training process in real-time by visiting your wandb project dashboard.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.