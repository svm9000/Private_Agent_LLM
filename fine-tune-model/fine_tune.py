# Import necessary libraries
import os
import subprocess
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from transformers import TrainingArguments
from trl import SFTTrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the base model and tokenizer
model_name = "unsloth/llama-3.2-1b-bnb-4bit"  # Specify the base model name
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,  # Maximum sequence length for the model
    dtype=None,           # Default data type (e.g., fp32 or bf16)
    load_in_4bit=True,    # Load the model in 4-bit precision for memory efficiency
    local_files_only=True # Use local files only (no downloading from the internet)
)

# Step 2: Apply LoRA (Low-Rank Adaptation) to the model for fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                           # Rank of the LoRA matrices
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # Target modules for LoRA
    lora_alpha=32,                  # Scaling factor for LoRA updates
    lora_dropout=0.05,              # Dropout rate applied to LoRA updates
    bias="none",                    # Bias configuration ("none" means no bias adaptation)
    use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing to save memory
    random_state=3407,              # Random seed for reproducibility
    use_rslora=True                 # Use RS-LoRA (a variant of LoRA)
)

# Step 3: Customize the tokenizer with a chat template
tokenizer = get_chat_template(
    tokenizer,
    mapping={
        "role": "from",         # Map role to "from"
        "content": "value",     # Map content to "value"
        "user": "human",        # Map user role to "human"
        "assistant": "gpt"      # Map assistant role to "gpt"
    }
)

# Step 4: Load and preprocess the dataset
# Load the dataset with conversations in ShareGPT format
origdataset = load_dataset("philschmid/guanaco-sharegpt-style", split="train")

# Select only the 'conversations' column from the dataset
conversations_dataset = origdataset.select_columns(['conversations'])

# Ensure the dataset is not empty
if len(conversations_dataset) == 0:
    raise ValueError("The dataset is empty. Please check the data source.")

# Convert the dataset to a list format
conversations_list = conversations_dataset['conversations']

# Split the dataset into training and validation sets
train_conversations, val_conversations = train_test_split(conversations_list, test_size=0.1, random_state=42)

# Ensure the splits are not empty
if not train_conversations or not val_conversations:
    raise ValueError("The dataset split resulted in empty subsets. Please check the data and split parameters.")

# Step 5: Format conversations using the chat template and tokenize them
# Create new datasets from the split lists
train_dataset = origdataset.filter(lambda x: x['conversations'] in train_conversations, desc="Filtering training conversations")
val_dataset = origdataset.filter(lambda x: x['conversations'] in val_conversations, desc="Filtering validation conversations")

# Format conversations using the chat template and tokenize them
train_dataset = train_dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(
            x["conversations"],
            tokenize=False,           # Do not tokenize during formatting
            add_generation_prompt=False  # Do not add generation prompts
        )
    },
    batched=True,                    # Process data in batches for efficiency
    batch_size=100,                  # Batch size for processing
    desc="Formatting training conversations"  # Description for progress bar/logging
)

val_dataset = val_dataset.map(
    lambda x: {
        "text": tokenizer.apply_chat_template(
            x["conversations"],
            tokenize=False,           # Do not tokenize during formatting
            add_generation_prompt=False  # Do not add generation prompts
        )
    },
    batched=True,                    # Process data in batches for efficiency
    batch_size=100,                  # Batch size for processing
    desc="Formatting validation conversations"  # Description for progress bar/logging
)

# Step 6: Define and initialize the Trainer for supervised fine-tuning (SFT)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",       # Specify which field contains text data in the dataset

    args=TrainingArguments(          # Training arguments configuration
        per_device_train_batch_size=2,      # Batch size per device (GPU/CPU)
        gradient_accumulation_steps=4,      # Accumulate gradients over multiple steps
        warmup_steps=5,                     # Number of warmup steps for learning rate scheduler
        max_steps=60,                       # Total number of training steps
        learning_rate=2e-4,                 # Learning rate for optimizer
        fp16=not is_bfloat16_supported(),   # Use FP16 if BF16 is not supported by hardware
        bf16=is_bfloat16_supported(),       # Use BF16 if supported by hardware (e.g., newer GPUs)
        logging_steps=1,                    # Log training metrics every step
        optim="adamw_8bit",                 # Optimizer with memory-efficient AdamW implementation (8-bit)
        weight_decay=0.01,                  # Weight decay regularization factor
        lr_scheduler_type="linear",         # Learning rate scheduler type (linear decay)
        seed=3407,                          # Random seed for reproducibility
        output_dir="outputs",               # Directory to save training outputs and checkpoints
        report_to="none"                    # Disable reporting to external tools like WandB or TensorBoard
        logging_dir="./logs"                # Directory for TensorBoard logs

    ),
)

# Step 7: Evaluate the model on the validation set
trainer.train()
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

#### 8. Optionally save the trained model and perform GGUF conversion
model.save_pretrained_gguf("ggufmodel", tokenizer, quantization_method="q4_k_m")

