# train_grpo.py
GRPO Training Script for Qwen Model on GSM8K Dataset
# GRPO Training Script for Qwen Model on GSM8K Dataset

## Overview

This script trains a **Qwen** model using the **GRPO (Group Relative Policy Optimization)** method on the **GSM8K** (Grade School Math 8K) dataset. The script leverages **transformers**, **PEFT** (Parameter-Efficient Fine-Tuning), and **TRL** (Transformer Reinforcement Learning) libraries to fine-tune the model with reinforcement learning techniques focused on one-shot prompting.

## Features

- **Dataset**: Loads the GSM8K dataset and applies one-shot prompting.
- **Reward Functions**: Defines multiple reward functions to guide the model's training.
- **Model Setup**: Configures and loads the Qwen model using the `AutoModelForCausalLM` class.
- **PEFT Configuration**: Uses Lora (Low-Rank Adaptation) for efficient fine-tuning.
- **Training Setup**: Configures GRPO training arguments and initializes the `GRPOTrainer`.
- **Logging**: Implements logging to track the training process.

## Prerequisites

- **Python**: 3.8 or higher.
- **PyTorch**: Installed and configured for CUDA (recommended).
- **Transformers**: Version 4.28.1 or higher.
- **PEFT**: Version 0.3.0 or higher.
- **TRL**: Version 0.3.0 or higher.
- **Hugging Face Datasets**: Version 2.10.1 or higher.
- **wandb**: (optional) For logging if `report_to="wandb"`.

You can install the required libraries using `pip`:

```bash
pip install \
    torch \
    transformers>=4.28.1 \
    peft>=0.3.0 \
    trl>=0.3.0 \
    datasets>=2.10.1 \
    wandb
```
# Setting Environment Variables

Before running the script, set the following environment variables:

* `MODEL_NAME`: Name of the Hugging Face model (default: `Qwen/Qwen2.5-1.5B-Instruct`).
* `OUTPUT_DIR`: Directory to save training outputs (default: `outputs/default-GRPO`).
* `RUN_NAME`: Name of the training run (default: `default-GRPO-gsm8k`).

For example:

```bash
export MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export OUTPUT_DIR="./outputs/default-GRPO"
export RUN_NAME="default-GRPO-gsm8k"
```
# Running the Script
To train the model, execute the following command in your terminal:
```python
python train_grpo.py
```
# Key Modifications
PEFT Configuration: Uncomment the peft_config line in the script if you want to apply LoRA for efficient parameter tuning.
Reward Functions: Customize the reward functions to fit your training objectives better.
Training Arguments: Adjust parameters in GRPOConfig to optimize training performance.

# Code Structure
Main Components
- **Logging Setup** :
Configures logging to track the training process and handle errors.
- **Dataset Preparation** :
Loads the GSM8K dataset.
Applies one-shot prompting using a system prompt and optional example.
- **Reward Functions** :
Correctness Reward: Rewards correct responses based on the extracted answer.
Integer Reward: Rewards responses that produce digit answers.
Format Reward: Rewards responses that follow the XML formatting.
XML Count Reward: Rewards responses are based on XML tag counts and penalize extra content.
- **Model Loading** :
Loads the pre-trained Qwen model and tokenizer.
Sets up the model for efficient fine-tuning using LoRA (optional).
- **Training Setup** :
Configures GRPO training arguments.
Initializes the GRPOTrainer with the model, tokenizer, reward functions, and dataset.
- **Training Process** :
Runs the training loop.
Saves the trained model and logs to the specified output directory.

# Utility Functions
**extract_xml_answer**: Extracts the answer from XML-formatted responses.
**extract_hash_answer**: Extracts the answer from hash-formatted responses.
**count_xml**: Counts XML tags and applies penalties for extra content.

# Example Output
During training, you will see logs similar to the following:
```python
INFO: [train_grpo.py] Question:
What is 68 divided by the product of 2 and 4?
Answer:
34
Response:
Th
<answer>
34
</answer>
Extracted:
34
```
# Additional Notes
**Environment Configuration**: Ensure that CUDA is correctly configured if you are using a GPU.
**Hyperparameter Tuning**: To optimise performance, experiment with different learning rates, batch sizes, and reward function weights.
**Logging and Monitoring**: Use `wandb` for better visualization and monitoring of the training process.

Feel free to contribute to this project, report issues, and provide feedback.