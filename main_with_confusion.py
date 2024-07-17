import pandas as pd
import numpy as np
import torch
import transformers
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load the IMDB dataset
imdb = load_dataset("imdb")

# Create smaller datasets for faster training and testing
small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

# Load the DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define the preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

# Preprocess the training and test datasets
tokenized_train = small_train_dataset.map(preprocess_function, batched=True)
tokenized_test = small_test_dataset.map(preprocess_function, batched=True)

# Define the model architecture
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="finetuning-sentiment-model-3000-samples",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    save_total_limit=2,
    save_steps=500,
    load_best_model_at_end=True,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model to the Hugging Face model hub
trainer.push_to_hub("finetuning-sentiment-model-3000-samples")

# Load the trained model from the Hugging Face model hub
model = AutoModelForSequenceClassification.from_pretrained("finetuning-sentiment-model-3000-samples")

# Define a function to predict the sentiment of a text
def predict_sentiment(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor([input_ids]).cuda()
    outputs = model(input_ids)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=1)
    predicted_label = np.argmax(probabilities.cpu().detach().numpy())
    if predicted_label == 0:
        return "Negative"
    else:
        return "Positive"

# Test the model on a sample text
text = "This movie was terrible."
print(predict_sentiment(text))