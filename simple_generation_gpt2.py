import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

from transformers import (
    GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, tokenizer, length, context_raw, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    max_context = model.config.max_position_embeddings
    
    context = tokenizer.encode(context_raw)[-max_context:]
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)

    max_past = min(max_context+length,
                tokenizer.max_len_single_sentence)
    past = None
    next_token = None
    generated = context
    with torch.no_grad():
        for i in trange(length):
            if i == length - 1:
                outputs = model(generated[:, -max_past:])
            else:
                outputs = model(next_token if next_token else
                            generated, past=past)
            
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            past = [p[..., -max_past+1:, :] for p in outputs[1]]

    return generated

parser = ArgumentParser()

parser.add_argument('model_dir',
                    help="Model to be used for generation.")
parser.add_argument('checkpoint_dir',
                    help="Checkpoint to be used for generation.")
parser.add_argument('--prompt', default=None)
parser.add_argument('--length', type=int, default=20)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_k', type=int, default=0)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--stop_token', default=None)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

model_dir = args.model_dir
checkpoint_dir = args.checkpoint_dir
prompt = args.prompt
length = args.length
temperature = args.temperature
top_k = args.top_k
top_p = args.top_p
stop_token = args.stop_token
seed = args.seed

assert os.path.isdir(model_dir)
assert os.path.isdir(os.path.join(model_dir, checkpoint_dir))

# Prefer GPU

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set random seed

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(seed)

# Get tokenizer and model

tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(os.path.join(model_dir,
    checkpoint_dir))
model.to(device)
model.eval()

if length < 0 and model.config.max_position_embeddings > 0:
    length = model.config.max_position_embeddings
elif 0 < model.config.max_position_embeddings < length:
    length = model.config.max_position_embeddings  # No generation bigger than model size 
elif length < 0:
    length = int(10000) # Hardcoded to prevent infinite loop

# Check if prompt was given
# If no prompt, do unconditional generation

if prompt is None:
    print("No prompt given -- performing unconditional generation")
    prompt = "<|endoftext|>"

out = sample_sequence(
    model=model,
    tokenizer=tokenizer,
    context_raw=prompt,
    length=length,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    device=device
)
out = out[:, len(context_tokens):].tolist()
for o in out:
    text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
    text = text[: text.find(stop_token) if stop_token else None]

    print(f'Context: {prompt}')
    print(f'Question: {text}')

