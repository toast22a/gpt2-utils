import os
import random
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from transformers import (
    GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
)

# Setting up argument parser
# Arguments include:
# - model directory
# - config directory
# - tokenizer directory
# - pretrained model name
# - max context length
# - max generation length
# - temperature
# - repetition penalty

parser = ArgumentParser()

parser.add_argument('--model_dir', default='model',
                    help="The model directory path.")
parser.add_argument('--config_dir', default=None,
                    help="The config directory path if different from model_dir.")
parser.add_argument('--tokenizer_dir', default=None,
                    help="The tokenizer directory path if different from model_dir.")
parser.add_argument('--pretrained_model_name', default='gpt2',
                    help="The pretrained model to use (if model dir not found).")
parser.add_argument('--max_context', type=int, default=-1,
                    help="The maximum context length to carry over.")
parser.add_argument('--max_generation', type=int, default=-1,
                    help="The maximum length of each generation.")
parser.add_argument('--temperature', type=float, default=1,
                    help="The sampling temperature.")
parser.add_argument('--repetition_penalty', type=float, default=1.0,
                    help="The value for CTRL's repetition penalty.")
parser.add_argument('--forget_context_prob', type=float, default=0.0,
                    help="The chance that the carryover context will be cleared [0,1].")

args = parser.parse_args()

model_dir = args.model_dir
config_dir = args.config_dir
tokenizer_dir = args.tokenizer_dir
pretrained_model_name = args.pretrained_model_name
max_context = args.max_context
max_generation = args.max_generation
temperature = args.temperature
repetition_penalty = args.repetition_penalty
forget_context_prob = args.forget_context_prob

# Set PyTorch device (prioritize GPU)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Don't set random seed -- we want chat to be random every time

# Load the config/model/tokenizer

model_name = model_dir if os.path.isdir(model_dir) else pretrained_model_name
config_name = config_dir if config_dir and os.path.isdir(config_dir) else model_name
tokenizer_name = tokenizer_dir if tokenizer_dir and os.path.isdir(tokenizer_dir) else model_name

config = GPT2Config.from_pretrained(config_name)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

model.to(device)
model.eval()

# Sanity checks on MCL and MGL

max_context = max_context if 0 < max_context <= tokenizer.max_len_single_sentence \
        else tokenizer.max_len_single_sentence
max_generation = max_generation if 0 < max_generation <= config.max_position_embeddings \
        else config.max_position_embeddings

# Sanity check on forget probability

if forget_context_prob < 0.0:
    forget_context_prob = 0.0
elif forget_context_prob > 1.0:
    forget_context_prob = 1.0

forget_context_prob = int(forget_context_prob * 100)

# Get index of stop tokens

stop_tokens = ['<|user|>', '<|bot|>']

stop_token_indices = [tokenizer.encode(x)[0] for x in stop_tokens]

# Define sample function

def sample(context_raw, max_context, max_generation, temperature, repetition_penalty):
    context = tokenizer.encode(context_raw)[-max_context:]
    context = torch.tensor(context, dtype=torch.long, device='cpu')
    context = context.unsqueeze(0)

    max_past = min(max_context+max_generation, tokenizer.max_len_single_sentence)
    next_token = None
    past = None
    generated = context.clone().to(device)
    with torch.no_grad():
        for _ in range(max_generation):
            outputs = model(next_token if next_token else generated, past=past)

            if temperature == 0: # greedy
                next_token_logits = outputs[0][:, -1, :]
            else:
                next_token_logits = outputs[0][:, -1, :] / temperature

            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > 0.9
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float('Inf')

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for _ in set(generated[0].tolist()):
                next_token_logits[0, _] /= repetition_penalty

            if temperature == 0: # greedy
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() in stop_token_indices: break

            past = [p[..., -max_past:, :] for p in outputs[1]]

    generated = generated.to('cpu')
    out = generated[:, context.size()[1]:].tolist()
    out_decoded = tokenizer.decode(out[0])
    out_decoded = out_decoded[:min([x if x >= 0 else len(out_decoded) for x in
        [out_decoded.find(stop_token) for stop_token in stop_tokens]])]

    context = context[:, :].tolist()
    context_decoded = tokenizer.decode(context[0])

    return (context_decoded, out_decoded)

context_raw = ''
context_input = ''
while True:
    context_input = input('User: ')
    if context_input.rstrip() == '': break
    context_input = '<|user|>' + context_input + '<|bot|>'

    if random.randrange(0,100) < forget_context_prob:
        context_raw = ''

    context_raw += context_input
    context_raw, generated = sample(context_raw, max_context, max_generation,
            temperature, repetition_penalty)
    context_raw += generated
    for generated_line in generated.split('<|bot|>'):
        print('Bot: {}'.format(generated_line))
