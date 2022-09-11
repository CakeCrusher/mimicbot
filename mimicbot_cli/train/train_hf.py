import os
from typing import Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
import pyarrow as pa
import torch
# import dotenv
# dotenv.load_dotenv()
import numpy as np
from pathlib import Path
from mimicbot_cli import SUCCESS

def train(session_path: Path) -> Tuple[str, int]:
    print("Inside train_hf.py")
    return ("Yay!", SUCCESS)

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# config = {
#     "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY"),
#     "MODEL_NAME": os.getenv("MODEL_NAME"),
# }

# # tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
# # model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m").to(device)
# tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m", use_auth_token=config["HUGGINGFACE_API_KEY"])
# model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", use_auth_token=config["HUGGINGFACE_API_KEY"]).to(device)

# test_data = pd.read_csv("/content/test.csv")
# train_data = pd.read_csv("/content/train.csv")

# train_dataset = Dataset.from_pandas(train_data)
# test_dataset = Dataset.from_pandas(test_data)
# messages_datasets = DatasetDict({
#     "train": train_dataset,
#     "validation": test_dataset
# })
# print(messages_datasets)

# max_context_length = 64
# max_response_length = 64
# CONTEXT_LENGTH = 2
# def preprocess_function(row):
#     collated_context = tokenizer.eos_token.join([
#         row[f"context{i+1}"] for i in range(CONTEXT_LENGTH)
#     ])
#     response = row["response"]
#     tokenized_context = tokenizer(
#         collated_context, max_length=max_context_length, truncation=True, padding="max_length"
#     )
#     with tokenizer.as_target_tokenizer():
#         tokenized_response = tokenizer(
#             response, max_length=max_response_length, truncation=True, padding="max_length"
#         )
#     tokenized_context["labels"] = tokenized_response["input_ids"]
#     return tokenized_context
# print(preprocess_function(messages_datasets["train"][0]))

# tokenized_datasets = messages_datasets.map(preprocess_function)
# tokenized_datasets = tokenized_datasets.remove_columns(messages_datasets["train"].column_names)
# print(tokenized_datasets)

# rouge_score = load_metric("rouge")
# def compute_metrics(eval_pred):
#     preds, labels = eval_pred
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     scores = rouge_score.compute(predictions=decoded_preds, references=decoded_labels)
#     results = {k: np.round(v.mid.fmeasure*100, 4) for k, v in scores.items()}
#     return results

# data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# # train
# train_args = Seq2SeqTrainingArguments(
#     overwrite_output_dir=True,
#     output_dir="saves/models",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     weight_decay=0.01,
#     save_total_limit=2,
#     num_train_epochs=1,
#     predict_with_generate=True,
#     logging_steps=len(tokenized_datasets["train"]) // 4,
#     hub_strategy="every_save",
#     push_to_hub=True,
#     hub_model_id=config["MODEL_NAME"],
#     hub_private_repo=True,
#     hub_token=config["HUGGINGFACE_API_KEY"],
# )

# trainer = Seq2SeqTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=train_args,
#     data_collator=data_collator,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["validation"],
#     compute_metrics=compute_metrics
# )

# trainer.create_optimizer()

# trainer.train()
# trainer.push_to_hub()



# # overfit
# for batch in trainer.get_train_dataloader():
#     batch.to(device)
#     break
# for _ in range(10):
    
#     outputs = trainer.model(**batch)
#     loss = outputs.loss
#     loss.backward()
#     trainer.optimizer.step()
#     trainer.optimizer.zero_grad()

# with torch.no_grad():
#   outputs = trainer.model(**batch)
# preds = outputs.logits
# labels = batch["labels"]

# tokenizer.decode(labels[0])

# preds = torch.argmax(preds, dim=-1)
# tokenizer.decode(preds[0])

# # clear cache
# import gc
# gc.collect()
# torch.cuda.empty_cache()