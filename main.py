import argparse
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
parser.add_argument("--model", type=str, default="qwen2-7b")
parser.add_argument(
    "--prompt",
    type=str,
    default="Give me a short introduction to large language model.",
)
parser.add_argument("--solution", type=str, default="tiny_llm_torch_ref")
parser.add_argument("--loader", type=str, default="week1")
parser.add_argument("--device", type=str, default="gpu")
parser.add_argument("--sampler-temp", type=float, default=0)
parser.add_argument("--sampler-top-p", type=float, default=None)
parser.add_argument("--sampler-top-k", type=int, default=None)
parser.add_argument("--enable-thinking", action="store_true")

args = parser.parse_args()

if args.solution in {"tiny_llm", "user"}:
    from tiny_llm import load_tokenizer, models, sampler, simple_generate
elif args.solution in {"tiny_llm_torch_ref", "torch_ref", "ref"}:
    from tiny_llm_torch_ref import load_tokenizer, models, sampler, simple_generate
else:
    raise ValueError(f"Solution {args.solution} not supported")
if args.loader != "week1":
    raise ValueError("Only week1 is currently supported")
if args.device != "gpu":
    raise ValueError("Only --device gpu is currently supported")
if not torch.cuda.is_available():
    raise ValueError("CUDA is not available")

args.model = models.shortcut_name_to_full_name(args.model)

hf_model = AutoModelForCausalLM.from_pretrained(
    args.model,
    local_files_only=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
hf_model.to("cuda")
hf_model.eval()

tokenizer = load_tokenizer(args.model)
tiny_llm_model = models.dispatch_model(args.model, hf_model, week=1, device="cuda")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": args.prompt},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=args.enable_thinking,
)
sampler_fn = sampler.make_sampler(
    args.sampler_temp,
    top_p=args.sampler_top_p,
    top_k=args.sampler_top_k,
)

print(f"Using {args.solution} week1 loader for {args.model} on cuda")
simple_generate(tiny_llm_model, tokenizer, prompt, sampler=sampler_fn)
