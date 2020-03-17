import glob
import json
import logging
import os
import pickle
import random
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW, get_linear_schedule_with_warmup,
    GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
)

# Silence logging because we don't wanna flood the Colab output window
logging.disable(logging.CRITICAL)

# Setting up argument parser

parser = ArgumentParser()

parser.add_argument('train_file_path',
                    help="The (text) data file to train the model on.")
parser.add_argument('--eval_file_path', default=None,
                    help="The (text) data file to evaluate the model on.")

parser.add_argument('--model_dir', default='model',
                    help="The model directory path.")

parser.add_argument('--pretrained_model_name', default='gpt2',
                    help="The pretrained model to use (if model dir not found).")
parser.add_argument('--new_tokens', nargs='+', type=str,
                    help="New tokens to initialize if any.")

parser.add_argument('--batch_size', type=int, default=1,
                    help="The number of examples in a minibatch.")
parser.add_argument('--n_epochs', type=int, default=1,
                    help="The number of epochs to train for.")
parser.add_argument('--n_accumulation', type=int, default=1,
                    help="The number of gradient accumulation steps.")
parser.add_argument('--msl', type=int, default=-1,
                    help="The maximum sequence length for the training data.")
parser.add_argument('--warmup_pct', type=float, default=0.0,
                    help="The percentage of total steps to warmup for.")
parser.add_argument('--lr', type=float, default=5e-5,
                    help="The learning rate.")                    

parser.add_argument('--recovery_every', type=int, default=20,
                    help="Create a recovery checkpoint every N steps.")
parser.add_argument('--ignore_recovery', action='store_true',
                    help="Don't continue training from a recovery checkpoint if one exists.")

args = parser.parse_args()

train_file_path = args.train_file_path
eval_file_path = args.eval_file_path
model_dir = args.model_dir
pretrained_model_name = args.pretrained_model_name
new_tokens = args.new_tokens
batch_size = args.batch_size
n_epochs = args.n_epochs
n_accumulation = args.n_accumulation
msl = args.msl
warmup_pct = args.warmup_pct
lr = args.lr
recovery_every = args.recovery_every
ignore_recovery = args.ignore_recovery

# Set PyTorch device (prioritize GPU)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seed

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Set model to load (from directory or from pretrained)

model_name = model_dir if os.path.isdir(model_dir) else pretrained_model_name

# Check if we're continuing from a checkpoint

checkpoint_dir = None
if model_dir and not ignore_recovery:
    previous_recoveries = sorted(
            glob.glob(os.path.join(model_dir, 'recovery-*-*')),
            key=lambda x: [int(y) for y in x.split('-')[-2:]],
            reverse=True
    )
    checkpoint_dir = previous_recoveries[0] if len(previous_recoveries) > 0 else None

if checkpoint_dir:
    print(f"Loading from checkpoint {checkpoint_dir}.")

# Load the tokenizer and model

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir if checkpoint_dir else model_name)

# If not continuing from checkpoint, add new uninitialized tokens to tokenizer and model

if not checkpoint_dir and new_tokens and len(new_tokens) > 0:
    print(f"Adding new token/s {', '.join(new_tokens)} to tokenizer and model vocabulary.")
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

# Set up the output directory for this model

if os.path.isdir(model_dir):
    print(f'Starting from model files in directory {model_dir}. Will be overwritten if modifications were made (e.g. new tokens).')
else:
    print(f"Creating new directory {model_dir} with initial pretrained model {pretrained_model_name}.")
    os.makedirs(model_dir)
    
# Save tokenizer once, since this won't change over epochs
tokenizer.save_pretrained(model_dir)

# Save base model we started from
model.save_pretrained(model_dir)

# Transfer model to training device

model.to(device)

# Prep the dataset

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_' + str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file):
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text)-block_size+1, block_size):
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))

            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])

class TextDirDataset(TextDataset):
    def __init__(self, tokenizer, dir_path='train', block_size=512):
        assert os.path.isdir(dir_path)
        directory_parent, directory = os.path.split(dir_path)
        cached_features_file = os.path.join(directory_parent, 'cached_lm_' + str(block_size) + '_' + directory)

        if os.path.exists(cached_features_file):
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            self.examples = []
            tokenized_text = []

            file_paths = sorted([x for x in glob.glob(f'{dir_path}/**', recursive=True)
                    if os.path.isfile(x)])

            for file_path in tqdm(file_paths, desc='Tokenizing'):
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text)-block_size+1, block_size):
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))

            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

if os.path.isfile(train_file_path):
    train_dataset = TextDataset(
                tokenizer,
                file_path=train_file_path,
                block_size=msl if msl > 0 else tokenizer.max_len_single_sentence
            )
else:
    train_dataset = TextDirDataset(
                tokenizer,
                dir_path=train_file_path,
                block_size=msl if msl > 0 else tokenizer.max_len_single_sentence
            )

do_eval = True

if eval_file_path == None:
    do_eval = False
elif os.path.isfile(eval_file_path):
    eval_dataset = TextDataset(
                tokenizer,
                file_path=eval_file_path,
                block_size=msl if msl > 0 else tokenizer.max_len_single_sentence
            )
elif os.path.isdir(eval_file_path):
    eval_dataset = TextDirDataset(
                tokenizer,
                dir_path=eval_file_path,
                block_size=msl if msl > 0 else tokenizer.max_len_single_sentence
            )
else:
    throw Exception("Invalid eval dataset option")

# Training loop

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=batch_size
        )

# Number of batches the dataset has been split into
n_batches = len(train_dataloader)

# Warmup percentage sanity check

if warmup_pct < 0.0:
    warmup_pct = 0.0
elif warmup_pct > 1.0:
    warmup_pct = 1.0

# For scheduler
t_total = int((n_batches * n_epochs) / n_accumulation)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total*warmup_pct), num_training_steps=t_total)

load_epoch = 0
load_step = 0
load_tr_loss = None

if checkpoint_dir:
    optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'pytorch_optimizer.bin')))
    scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'pytorch_scheduler.bin')))

    with open(os.path.join(checkpoint_dir, 'step.json'), 'r') as f:
        checkpoint_step = json.loads(f.read())

    load_epoch = checkpoint_step['epoch']
    load_step = checkpoint_step['step']
    if load_step > 0:
        load_tr_loss = checkpoint_step['tr_loss']

model.zero_grad()

for epoch in range(n_epochs):
    if load_epoch > 0:
        load_epoch -= 1
        continue

    print("Epoch {}".format(epoch))
    tr_loss = 0.0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        if load_step > 0:
            load_step -= 1
            continue

        if load_tr_loss is not None:
            tr_loss = load_tr_loss
            load_tr_loss = None

        # HF's LM implementation automatically shifts labels by one
        inputs, labels = (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Set train mode
        model.train()

        # Forward
        outputs = model(inputs, labels=labels)
        loss = outputs[0]

        # Backprop
        loss.backward()

        tr_loss += loss.item()

        if (step + 1) % n_accumulation == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if recovery_every > 0 and (epoch * n_batches + step + 1) % recovery_every == 0:
            # Prep epoch and step values to be saved
            save_epoch = epoch
            save_step = step + 1
            while save_step >= n_batches:
                save_epoch += 1
                save_step -= n_batches

            # Find previous recovery checkpoint/s, if any
            previous_recoveries = glob.glob(os.path.join(model_dir, 'recovery-*-*'))

            # Set up subdirectory for this epoch checkpoint
            save_dir = os.path.join(model_dir, 'recovery-{}-{}'.format(save_epoch, save_step))
            os.makedirs(save_dir, exist_ok=True)

            # Save the model weights
            model.save_pretrained(save_dir)

            # Save the optimizer and scheduler so we can continue where we left off
            torch.save(optimizer.state_dict(), os.path.join(save_dir, 'pytorch_optimizer.bin'))
            torch.save(scheduler.state_dict(), os.path.join(save_dir, 'pytorch_scheduler.bin'))

            # Save the step count and other details so we know where to continue
            with open(os.path.join(save_dir, 'step.json'), 'w') as f:
                f.write(json.dumps({
                        'epoch': save_epoch,
                        'step': save_step,
                        'tr_loss': tr_loss
                    }))

            # Don't save eval scores for recovery checkpoints
            # We only eval completed epochs

            # Delete previous recovery checkpoints to save space
            for previous in previous_recoveries:
                shutil.rmtree(previous)

    tr_loss /= n_batches

    # Evaluation
    # Perplexity for now; we'll add BLEU-4 later
    
    if do_eval:

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
                    eval_dataset,
                    sampler=eval_sampler,
                    batch_size=batch_size
                )

        n_batches_eval = len(eval_dataloader)
        eval_loss = 0.0

        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs, labels=labels)
                
                loss = outputs[0]
                eval_loss += loss.item()

        eval_loss /= n_batches_eval
        eval_ppl = torch.exp(torch.tensor(eval_loss)).item()

        print("Train Loss: {:.4f} Train LR: {:.8f} Eval Loss: {:.4f} Eval PPL: {:.4f}".format(
                tr_loss,
                scheduler.get_last_lr()[0],
                eval_loss,
                eval_ppl
            ))
    else:
        print("Train Loss: {:.4f} Train LR: {:.8f}".format(
                tr_loss,
                scheduler.get_last_lr()[0]
            ))

    # Prep epoch and step values to be saved
    save_epoch = epoch + 1
    save_step = 0

    # Set up subdirectory for this epoch checkpoint
    save_dir = os.path.join(model_dir, 'epoch-{}'.format(save_epoch))
    os.makedirs(save_dir, exist_ok=True)

    # Save the model weights
    model.save_pretrained(save_dir)

    # Save the optimizer and scheduler so we can continue where we left off
    # torch.save(optimizer.state_dict(), os.path.join(save_dir, 'pytorch_optimizer.bin'))
    # torch.save(scheduler.state_dict(), os.path.join(save_dir, 'pytorch_scheduler.bin'))

    # Save the step count and other details so we know where to continue
    with open(os.path.join(save_dir, 'step.json'), 'w') as f:
        f.write(json.dumps({
                'epoch': save_epoch,
                'step': save_step,
                'tr_loss': tr_loss
            }))

    eval_dict = {
        'tr_loss': tr_loss,
        'lr': scheduler.get_last_lr()[0]
    }

    if do_eval:
        eval_dict['eval_loss'] = eval_loss
        eval_dict['eval_ppl'] = eval_ppl

    # Save eval scores
    with open(os.path.join(save_dir, 'eval.json'), 'w') as f:
        f.write(json.dumps(eval_dict))

