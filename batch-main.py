import argparse
import random
import sys
from pathlib import Path

import torch
import transformers.utils as transformers_utils
import transformers.utils.import_utils as import_utils
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import_utils._torchao_available = False
transformers_utils.is_torchao_available = lambda *args, **kwargs: False

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen2-0.5b")

shanghai_wikipedia = """
Shanghai[a] is a direct-administered municipality and the most populous urban area in China. The city is located on the Chinese shoreline on the southern estuary of the Yangtze River, with the Huangpu River flowing through it. The population of the city proper is the second largest in the world after Chongqing, with around 24.87 million inhabitants in 2023, while the urban area is the most populous in China, with 29.87 million residents. As of 2022, the Greater Shanghai metropolitan area was estimated to produce a gross metropolitan product (nominal) of nearly 13 trillion RMB ($1.9 trillion).[13] Shanghai is one of the world's major centers for finance, business and economics, research, science and technology, manufacturing, transportation, tourism, and culture. The Port of Shanghai is the world's busiest container port.
""".strip()

shanghai_wikipedia += "Based on the previous information, "

prompts = [
    shanghai_wikipedia + "Where is Shanghai?",
    shanghai_wikipedia + "How much is the population of Shanghai?",
    shanghai_wikipedia + "What is the GDP of Shanghai?",
    shanghai_wikipedia + "What is the population of Shanghai?",
    shanghai_wikipedia + "What is the second largest city proper in China?",
    shanghai_wikipedia + "What is Shanghai known for?",
    shanghai_wikipedia + "What are the rivers in Shanghai?",
    shanghai_wikipedia + "Shanghai is the major center for what?",
    "What is the capital of France?",
    "Where is New York City?",
    "Where is Tokyo?",
    "What is the capital of China?",
    "Where is Pittsburgh?",
    "Where is Vancouver?",
    "Where is Toronto?",
    "Give me a short introduction to large language model.",
]

random.shuffle(prompts)

parser.add_argument("--solution", type=str, default="tiny_llm_torch")
parser.add_argument("--device", type=str, default="gpu")
parser.add_argument("--batch-size", type=int, default=5)
parser.add_argument("--prefill-step", type=int, default=128)
parser.add_argument("--enable-flash-attn", action="store_true")
parser.add_argument("--enable-thinking", action="store_true")
args = parser.parse_args()

if args.solution in {"tiny_llm_torch", "user"}:
    print("Using your tiny_llm_torch solution")
    from tiny_llm_torch import batch_generate, load_tokenizer, models
elif args.solution in {"tiny_llm_torch_ref", "torch_ref", "ref"}:
    print("Using tiny_llm_torch_ref solution")
    from tiny_llm_torch_ref import batch_generate, load_tokenizer, models
else:
    raise ValueError(f"Solution {args.solution} not supported")

if args.device != "gpu":
    raise ValueError("Only --device gpu is currently supported")
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available")

args.model = models.shortcut_name_to_full_name(args.model, week=2)
hf_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    local_files_only=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
hf_model.to("cuda")
hf_model.eval()

tokenizer = load_tokenizer(args.model)
print(
    f"Using week2 loader with flash_attn={args.enable_flash_attn} thinking={args.enable_thinking} for {args.model}"
)
tiny_llm_model = models.dispatch_model(
    args.model,
    hf_model,
    week=2,
    device="cuda",
    enable_flash_attn=args.enable_flash_attn,
)

encoded_prompts = []
for idx, prompt in enumerate(prompts):
    print(f"Prompt {idx}: {prompt}")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    encoded_prompts.append(prompt)

result = batch_generate(
    tiny_llm_model,
    tokenizer,
    encoded_prompts,
    batch_size=args.batch_size,
    prefill_step=args.prefill_step,
)
for prompt_idx, text in result:
    print(f"--- {prompt_idx} ---")
    print(f"Q: {prompts[prompt_idx]}")
    print(f"A: {text}")
