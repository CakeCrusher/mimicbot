from mimicbot_cli import utils, Args, SUCCESS, GPU_ERROR, API_KEY_ERROR, CHANGE_VALUE
from huggingface_hub import get_full_repo_name
import pandas as pd
import torch
import os
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import DatasetDict, Dataset, load_metric
import numpy as np


def train(session_path: Path) -> Tuple[str, int]:
    config_parser = utils.callback_config()
    model_name = config_parser.get(
        "huggingface", "model_name")
    LARGE_LANGUAGE_MODEL = config_parser.get("huggingface", "large_language_model")
    HUGGINGFACE_API_KEY = config_parser.get("huggingface", "api_key")
    MODELS_PATH = session_path.parent.parent / "models"
    SESSION_PATH = session_path
    CACHE_DIR = MODELS_PATH / "cache"
    MODEL_TO = get_full_repo_name(model_name, token=HUGGINGFACE_API_KEY)

    trn_df = pd.read_csv(str(SESSION_PATH / "training_data" / "train.csv"))
    val_df = pd.read_csv(str(SESSION_PATH / "training_data" / "test.csv"))

    CONTEXT_LENGTH = len(trn_df.columns) - 1

    args = Args()


    args.output_dir = str(MODELS_PATH / MODEL_TO.replace("/","_"))
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_path = str(MODELS_PATH / MODEL_TO)
    args.save_to = MODEL_TO
    args.cache_dir = str(CACHE_DIR)
    args.no_cuda = not torch.cuda.is_available()
    
    init_model_res = utils.initialize_model(args, HUGGINGFACE_API_KEY, LARGE_LANGUAGE_MODEL, MODEL_TO)
    if (init_model_res[1] == SUCCESS):
        MODEL_FROM = init_model_res[0]
    else:
        return init_model_res

    tokenizer = AutoTokenizer.from_pretrained(MODEL_FROM, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
    model = AutoModelForCausalLM.from_pretrained(MODEL_FROM, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY).to(args.device)

    trn_dataset = Dataset.from_pandas(trn_df)
    val_dataset = Dataset.from_pandas(val_df)
    messages_datasets = DatasetDict({
        "train": trn_dataset,
        "validation": val_dataset
    })
    print(messages_datasets)

    max_context_length = 64
    max_response_length = 64

    def preprocess_function(row):
        collated_context = tokenizer.eos_token.join([
            row[f"context{i+1}"] for i in range(CONTEXT_LENGTH)
        ])
        response = row["response"]
        tokenized_context = tokenizer(
            collated_context, max_length=max_context_length, truncation=True, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            tokenized_response = tokenizer(
                response, max_length=max_response_length, truncation=True, padding="max_length"
            )
        tokenized_context["labels"] = tokenized_response["input_ids"]
        return tokenized_context
    print(preprocess_function(messages_datasets["train"][0]))

    tokenized_datasets = messages_datasets.map(preprocess_function)
    tokenized_datasets = tokenized_datasets.remove_columns(messages_datasets["train"].column_names)
    print(tokenized_datasets)


    rouge_score = load_metric("rouge")
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = rouge_score.compute(predictions=decoded_preds, references=decoded_labels)
        results = {k: np.round(v.mid.fmeasure*100, 4) for k, v in scores.items()}
        return results

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # train
    train_args = Seq2SeqTrainingArguments(
        overwrite_output_dir=True,
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_gpu_train_batch_size,
        per_device_eval_batch_size=args.per_gpu_eval_batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        logging_steps=len(tokenized_datasets["train"]) // 4,
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=MODEL_TO,
        hub_private_repo=True,
        hub_token=HUGGINGFACE_API_KEY,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics
    )

    trainer.create_optimizer()

    try:
        trainer.train()
    except RuntimeError:
        return (f"https://huggingface.co/{args.save_to}", GPU_ERROR)
    utils.save_to_repo(args, trainer.model, f"Epoch #{args.num_train_epochs}", HUGGINGFACE_API_KEY)

    return (f"https://huggingface.co/{args.save_to}", SUCCESS)