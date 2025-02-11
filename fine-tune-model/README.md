# 🚀 Fine-Tuning a Custom Model for Ollama

This script demonstrates how to fine-tune a custom language model using the **Unsloth library** with LoRA (Low-Rank Adaptation) and prepare it for use in **Ollama**. Below is an overview of the steps involved:

## 🛠️ Steps to Fine-Tune a Custom Model

### 1. 📚 Load the Base Model
- Loads a pre-trained model (`unsloth/llama-3.2-1b-bnb-4bit`) using `FastLanguageModel`.
- Uses 4-bit precision for memory efficiency.

### 2. 🔧 Apply LoRA for Fine-Tuning
- Applies LoRA to enable efficient fine-tuning.
- Targets specific modules and uses RS-LoRA for optimization.

### 3. 🗣️ Customize the Tokenizer
- Applies a chat template to format conversations.

### 4. 📊 Load and Preprocess the Dataset
- Loads dataset in ShareGPT format.
- Splits into training and validation sets.

### 5. 🔠 Tokenize and Format Data
- Formats conversations using the chat template.

### 6. 🏋️ Train the Model with SFT
- Uses `SFTTrainer` for fine-tuning.
- Configures training arguments (batch size, learning rate, etc.).

### 7. 📈 Evaluate the Model
- Evaluates on the validation set.

### 8. 💾 Save and Convert the Model
- Saves the fine-tuned model in GGUF format for Ollama compatibility.

## 🚀 How to Use This Script

1. **Install Required Libraries**:
    
    `pip install unsloth transformers trl datasets scikit-learn`

2. **Prepare Your Dataset**:
- Ensure it's in ShareGPT format or modify the loading step.

3. **Run the Script**:

    `python fine_tune.py`

4. **Use Your Custom Model in Ollama**:
- Convert to GGUF format (included in Step 8).
- Place in Ollama models directory (e.g., `/root/.ollama/models`).

## 🎉 Output
- Fine-tuned GGUF model in `ggufmodel` directory.
- Training logs and evaluation metrics.
- Ready for deployment in Ollama!

## 🌟 Key Features
- Efficient fine-tuning with LoRA.
- Memory-efficient 4-bit quantization.
- Support for chat-based datasets.
- Easy integration with Ollama.

This script provides an end-to-end pipeline for creating custom models tailored to conversational AI tasks! 🤖💬

---

## ⚙️ Environment Setup

Before running this script, ensure your environment is properly set up:

1. **Install Required Libraries**:
   Use the following command to install all necessary Python libraries:

2. **Set Up llama.cpp**:
- To set up `llama.cpp`, you must run the provided `setup_llama.sh` script:

    `./setup_llama.sh`

- This script will:
  - Clone the `llama.cpp` repository.
  - Build it using `CMake`.
  - Create a symlink to the `quantize` executable.
- Ensure that all dependencies (e.g., `git`, `cmake`, `make`) are installed before running this script.

3. **Verify llama.cpp Setup**:
After running `setup_llama.sh`, ensure that the `quantize` executable is available in your system's path or as a symlink in the specified directory.

---