import glob
from lib2to3.pgen2 import token
import logging
from multiprocessing.connection import Client
import os
import pickle
import random
import re
import shutil
import datetime
from typing import Dict, List, Tuple
from urllib.error import HTTPError

import pandas as pd
import numpy as np
import torch
from pathlib import Path

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange
import gc
import typer

from requests.exceptions import HTTPError

from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_metric
from huggingface_hub import create_repo, HfApi, get_full_repo_name
import pdb
from mimicbot_cli import (
    config,
    utils,
    API_KEY_ERROR,
    CHANGE_VALUE,
    SUCCESS,
    GPU_ERROR,
    Args,
)


from torch.utils.tensorboard import SummaryWriter


def train(session_path: Path) -> Tuple[str, int]:
    # Configs
    rouge_score = load_metric("rouge")
    logger = logging.getLogger(__name__)
    config_parser = utils.callback_config()
    model_name = config_parser.get(
        "huggingface", "model_name")
    HUGGINGFACE_API_KEY = config_parser.get("huggingface", "api_key")
    MODELS_PATH = session_path.parent.parent / "models"
    SESSION_PATH = session_path
    CACHE_DIR = MODELS_PATH / "cache"
    MODEL_TO = get_full_repo_name(model_name, token=HUGGINGFACE_API_KEY)

    trn_df = pd.read_csv(str(SESSION_PATH / "training_data" / "train.csv"))
    val_df = pd.read_csv(str(SESSION_PATH / "training_data" / "test.csv"))

    args = Args()

    args.output_dir = str(MODELS_PATH)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_path = str(MODELS_PATH / MODEL_TO)
    args.save_to = MODEL_TO
    args.cache_dir = str(CACHE_DIR)
    args.no_cuda = not torch.cuda.is_available()

    try:
        config = AutoConfig.from_pretrained(
            args.config_name, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
        model = AutoModelWithLMHead.from_pretrained(
            MODEL_TO,
            from_tf=False,
            config=config,
            cache_dir=args.cache_dir,
            use_auth_token=HUGGINGFACE_API_KEY,
        )
        MODEL_FROM = MODEL_TO
    except OSError:
        MODEL_FROM = args.config_name
    except ValueError:
        return (f"https://huggingface.co/{args.save_to}", API_KEY_ERROR)

    typer.secho(f"({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Fine tuning model from: https://huggingface.co/{MODEL_FROM}\n", fg=typer.colors.BLUE)

    # initiate repository
    if MODEL_FROM != MODEL_TO:
        try:
            link_to_repo = create_repo(
                args.save_to, private=True, token=HUGGINGFACE_API_KEY)
            config = AutoConfig.from_pretrained(
                args.config_name, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
            tokenizer = AutoTokenizer.from_pretrained(
                args.config_name, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
            model = AutoModelWithLMHead.from_pretrained(
                args.config_name,
                from_tf=False,
                config=config,
                cache_dir=args.cache_dir,
                use_auth_token=HUGGINGFACE_API_KEY
            ).to(args.device)
            hf_api = HfApi()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            hf_api.upload_file(
                path_or_fileobj=str(Path(current_dir) /
                                    "huggingface/README.md"),
                path_in_repo="README.md",
                repo_id=args.save_to,
                token=HUGGINGFACE_API_KEY)
            # config.push_to_hub(
            #     args.model_path, commit_message="init config", use_auth_token=HUGGINGFACE_API_KEY)
            tokenizer.push_to_hub(
                args.model_path, commit_message="init tokenizer", use_auth_token=HUGGINGFACE_API_KEY)
            model.push_to_hub(
                args.model_path, commit_message="init model", use_auth_token=HUGGINGFACE_API_KEY)
            typer.secho(
                f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Huggingface repo initialized at: {link_to_repo}", fg=typer.colors.BLUE)
        except ValueError:
            return (f"https://huggingface.co/{args.save_to}", API_KEY_ERROR)
        except HTTPError:
            typer.secho(
                f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Will update repo at: {'https://huggingface.co/'+MODEL_TO}", fg=typer.colors.BLUE)
        except OSError:
            # cannot delete old model so if model is deleted you need to rename model
            return (f"https://huggingface.co/{args.save_to}", CHANGE_VALUE)

    args.model_name = MODEL_FROM
    args.tokenizer_name = MODEL_FROM

    def construct_conv(row, tokenizer, eos=True):
        def flatten(l): return [item for sublist in l for item in sublist]
        conv = list(
            reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
        conv = flatten(conv)
        return conv

    class ConversationDataset(Dataset):
        def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

            block_size = block_size - (tokenizer.max_len_single_sentence)

            directory = args.cache_dir
            cached_features_file = os.path.join(
                directory, args.model_type + "_cached_lm_" + str(block_size)
            )

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s",
                            cached_features_file)
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
            else:
                logger.info(
                    "Creating features from dataset file at %s", directory)

                self.examples = []
                for _, row in df.iterrows():
                    conv = construct_conv(row, tokenizer)
                    self.examples.append(conv)

                logger.info("Saving features into cached file %s",
                            cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, item):
            return torch.tensor(self.examples[item], dtype=torch.long)

    def encodeWithoutResponse(row, tokenizer):
        overfitEncoded = construct_conv(row, tokenizer)
        overfitStr = tokenizer.decode(overfitEncoded)
        splitByMessage = overfitStr.split(tokenizer.eos_token)
        overfitExcludingResponse = tokenizer.eos_token.join(
            splitByMessage[:-2]) + tokenizer.eos_token
        return tokenizer.encode(overfitExcludingResponse, return_tensors="pt")

    def decodeGeneratedOutput(input, output, tokenizer):
        return tokenizer.decode(output[:, input.shape[-1]:][0], skip_special_tokens=True)

    def makePreds(df, model, tokenizer):
        preds = []
        for i, row in df.iterrows():
            testInput = encodeWithoutResponse(row, tokenizer).to(args.device)
            chatHistoryIds = model.generate(
                testInput,
                max_length=200,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.7,
                temperature=0.8,
            )
            preds.append(decodeGeneratedOutput(
                testInput, chatHistoryIds, tokenizer))
        return preds

    def computeRouge(df, model, tokenizer):
        labels = list(df["response"])
        preds = makePreds(df, model, tokenizer)
        scores = rouge_score.compute(
            predictions=preds, references=labels
        )
        return ({k: np.round(v.mid.fmeasure*100, 4) for k, v in scores.items()}, labels, preds)

    benchmarkDf = val_df.iloc[:30]

    config = AutoConfig.from_pretrained(
        args.config_name, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
    model = AutoModelWithLMHead.from_pretrained(
        args.model_name,
        from_tf=False,
        config=config,
        cache_dir=args.cache_dir,
        use_auth_token=HUGGINGFACE_API_KEY
    ).to(args.device)

    benchmarkScore = computeRouge(benchmarkDf, model, tokenizer)
    typer.secho(
        f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Initial benchmark test: {str(benchmarkScore[0])}", fg=typer.colors.BLUE)

    def save_to_repo(args, model, tokenizer, message):
        typer.secho(
            f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Uploading model to: https://huggingface.co/{args.save_to}", fg=typer.colors.BLUE)
        model.push_to_hub(
            args.model_path, commit_message=f"model: {message}", use_auth_token=HUGGINGFACE_API_KEY)
        # tokenizer.push_to_hub(
        #     args.model_path, commit_message=f"tokenizer: {message}", use_auth_token=HUGGINGFACE_API_KEY)
        typer.secho(f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Uploading finished, view it at: https://huggingface.co/{args.save_to}", fg=typer.colors.BLUE)

    def load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False):
        return ConversationDataset(tokenizer, args, df_val if evaluate else df_trn)

    def set_seed(args):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)

    def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = glob.glob(os.path.join(
            args.output_dir, "{}-*".format(checkpoint_prefix)))

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append(
                    (os.path.getmtime(path), path))
            else:
                regex_match = re.match(
                    ".*{}-([0-9]+)".format(checkpoint_prefix), path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1]
                              for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
        if not args.save_total_limit:
            return
        if args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = _sorted_checkpoints(
            args, checkpoint_prefix, use_mtime)
        if len(checkpoints_sorted) <= args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(
            0, len(checkpoints_sorted) - args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info(
                "Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, saveToHub) -> Tuple[int, float]:
        """ Train the model """
        gc.collect()
        torch.cuda.empty_cache()
        if args.local_rank in [-1, 0]:
            if not (Path(args.output_dir) / "SummaryWriter-log").exists():
                os.makedirs(Path(args.output_dir) / "SummaryWriter-log")
            tb_writer = SummaryWriter(log_dir=str(
                Path(args.output_dir) / "SummaryWriter-log"))
        args.train_batch_size = args.per_gpu_train_batch_size * \
            max(1, args.n_gpu)

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

        train_sampler = RandomSampler(
            train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate, drop_last=True
        )
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (
                len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(
                train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        # Take care of distributed/parallel training
        model = model.module if hasattr(model, "module") else model
        model.resize_token_embeddings(len(tokenizer))
        # add_special_tokens_(model, tokenizer)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon, no_deprecation_warning=True)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Check if saved optimizer or scheduler states exist
        if (
            args.model_path
            and os.path.isfile(os.path.join(args.model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(
                os.path.join(args.model_path, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(
                os.path.join(args.model_path, "scheduler.pt")))

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[
                    args.local_rank], output_device=args.local_rank, find_unused_parameters=True
            )

        # # Train!
        # logger.info("***** Running training *****")
        # logger.info("  Num examples = %d", len(train_dataset))
        # logger.info("  Num Epochs = %d", args.num_train_epochs)
        # logger.info("  Instantaneous batch size per GPU = %d",
        #             args.per_gpu_train_batch_size)
        # logger.info(
        #     "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        #     args.train_batch_size
        #     * args.gradient_accumulation_steps
        #     * (torch.distributed.get_world_size()
        #        if args.local_rank != -1 else 1),
        # )
        # logger.info("  Gradient Accumulation steps = %d",
        #             args.gradient_accumulation_steps)
        # logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if args.model_path and os.path.exists(args.model_path):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split(
                    "-")[-1].split("/")[0]
                global_step = int(checkpoint_suffix)
                # epochs_trained = global_step // (
                #     len(train_dataloader) // args.gradient_accumulation_steps)
                # # Reset training when script ran
                epochs_trained = 0
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps)

                logger.info(
                    "  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d",
                            epochs_trained)
                logger.info(
                    "  Continuing training from global step %d", global_step)
                logger.info("  Will skip the first %d steps in the first epoch",
                            steps_trained_in_current_epoch)
            except ValueError:
                logger.info("  Starting fine-tuning.")
        tr_loss, logging_loss = 0.0, 0.0

        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
        )
        set_seed(args)  # Added here for reproducibility
        for _ in train_iterator:
            typer.secho(
                f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Iteration #{_ + 1}", fg=typer.colors.BLUE)
            epoch_iterator = tqdm(
                train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs, labels = (batch, batch)
                if inputs.shape[1] > 1024:
                    continue
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                model.train()
                outputs = model(inputs, labels=labels)
                # model outputs are always tuple in transformers (see doc)
                loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        if (
                            args.local_rank == -1 and args.evaluate_during_training
                        ):  # Only evaluate when single GPU otherwise metrics may not average well
                            results = evaluate(args, model, tokenizer)
                            for key, value in results.items():
                                tb_writer.add_scalar(
                                    "eval_{}".format(key), value, global_step)
                        tb_writer.add_scalar(
                            "lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        checkpoint_prefix = "checkpoint"
                        # Save model checkpoint
                        output_dir = os.path.join(
                            args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                        os.makedirs(output_dir, exist_ok=True)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(
                            output_dir, "training_args.bin"))
                        logger.info(
                            "Saving model checkpoint to %s", output_dir)

                        _rotate_checkpoints(args, checkpoint_prefix)

                        torch.save(optimizer.state_dict(), os.path.join(
                            output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(
                            output_dir, "scheduler.pt"))
                        logger.info(
                            "Saving optimizer and scheduler states to %s", output_dir)
                if args.max_steps > 0 and global_step > args.max_steps:
                    epoch_iterator.close()
                    break
            if args.max_steps > 0 and global_step > args.max_steps:
                train_iterator.close()
                break

            # run benchmark assessment
            typer.secho(
                f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Benchmark after iteration #{_+1}: {str(computeRouge(benchmarkDf, model, tokenizer)[0])}", fg=typer.colors.BLUE)
            # save
            # pdb.set_trace()
            if saveToHub:
                save_to_repo(args, model, tokenizer, f"Epoch #{_+1}")
            # pdb.set_trace()
        if args.local_rank in [-1, 0]:
            tb_writer.close()

        return global_step, tr_loss / global_step

    # Evaluation of some model

    def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, df_trn, df_val, prefix="") -> Dict:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = args.output_dir

        eval_dataset = load_and_cache_examples(
            args, tokenizer, df_trn, df_val, evaluate=True)
        os.makedirs(eval_output_dir, exist_ok=True)
        args.eval_batch_size = args.per_gpu_eval_batch_size * \
            max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly

        def collate(examples: List[torch.Tensor]):
            if tokenizer._pad_token is None:
                return pad_sequence(examples, batch_first=True)
            return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate, drop_last=True
        )

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        # logger.info("***** Running evaluation {} *****".format(prefix))
        # logger.info("  Num examples = %d", len(eval_dataset))
        # logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                outputs = model(inputs, labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(
            eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return result

    def main(df_trn, df_val, args, save_to_hub=False):
        if args.should_continue:
            sorted_checkpoints = _sorted_checkpoints(args)
            if len(sorted_checkpoints) == 0:
                raise ValueError(
                    "Used --should_continue but no checkpoint was found in --output_dir.")
            else:
                args.model_name = sorted_checkpoints[-1]

        if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
            and not args.should_continue
        ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir
                )
            )

        # Setup CUDA, GPU & distributed training
        device = torch.device("cpu" if args.no_cuda else "cuda")
        args.n_gpu = torch.cuda.device_count()
        args.device = device

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            args.local_rank,
            device,
            args.n_gpu,
            bool(args.local_rank != -1),
            args.fp16,
        )

        # Set seed
        set_seed(args)

        config = AutoConfig.from_pretrained(
            args.config_name, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, cache_dir=args.cache_dir, use_auth_token=HUGGINGFACE_API_KEY)
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name,
            from_tf=False,
            config=config,
            cache_dir=args.cache_dir,
            use_auth_token=HUGGINGFACE_API_KEY
        )
        model.to(args.device)

        # logger.info("Training/evaluation parameters %s", args)

        # Training
        if args.do_train:
            train_dataset = load_and_cache_examples(
                args, tokenizer, df_trn, df_val, evaluate=False)
            global_step, tr_loss = train(
                args, train_dataset, model, tokenizer, save_to_hub)
            logger.info(" global_step = %s, average loss = %s",
                        global_step, tr_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.do_train:
            # Create output directory if needed
            os.makedirs(args.output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(
                args.output_dir, "training_args.bin"))

            # Load a trained model and vocabulary that you have fine-tuned
            model = AutoModelWithLMHead.from_pretrained(
                args.output_dir, use_auth_token=HUGGINGFACE_API_KEY)
            tokenizer = AutoTokenizer.from_pretrained(
                args.output_dir, use_auth_token=HUGGINGFACE_API_KEY)
            model.to(args.device)

        # Evaluation
        results = {}
        if args.do_eval and args.local_rank in [-1, 0]:
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(
                    logging.WARN)  # Reduce logging
            # logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                global_step = checkpoint.split(
                    "-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split(
                    "/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                model = AutoModelWithLMHead.from_pretrained(
                    checkpoint, use_auth_token=HUGGINGFACE_API_KEY)
                model.to(args.device)
                result = evaluate(args, model, tokenizer,
                                  df_trn, df_val, prefix=prefix)
                result = dict((k + "_{}".format(global_step), v)
                              for k, v in result.items())
                results.update(result)
        typer.secho(
            f"\n({datetime.datetime.now().hour}:{datetime.datetime.now().minute}) Training finished\n", fg=typer.colors.BLUE)
        return results

    gc.collect()
    torch.cuda.empty_cache()

    try:
        main(trn_df, val_df, args, True)
    except RuntimeError:
        return("", GPU_ERROR)
    return (f"https://huggingface.co/{args.save_to}", SUCCESS)
