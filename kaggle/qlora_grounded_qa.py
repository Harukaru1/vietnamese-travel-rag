# %% [markdown]
# # QLoRA: grounded Vietnamese travel QA with Qwen2.5
#
# Fine-tunes **Qwen2.5-1.5B-Instruct** with QLoRA to do the job Gemini does in
# the RAG pipeline of [vietnamese-travel-rag](https://github.com/Harukaru1/vietnamese-travel-rag):
# read 5 numbered Wikipedia passages, answer in Vietnamese, cite the passage
# numbers, and abstain (`found: false`) when the passages do not contain the answer.
#
# **Setup on Kaggle**
# 1. Accelerator: *GPU T4 x2* (one GPU is used). Internet: *on* (to download the model).
# 2. Add the dataset made from `data/sft/` (files `train.jsonl`, `val.jsonl`, `eval_prompts.jsonl`).
# 3. Run all. Training takes roughly an hour; generation for the base and the
#    fine-tuned model takes roughly 20–30 minutes more.
# 4. Download from the *Output* tab: `predictions_base.jsonl`,
#    `predictions_qlora.jsonl`, `train_log.json` (and the `adapter/` folder if
#    you want to publish it on Hugging Face). Score them locally with
#    `python -m travel_rag.eval_finetuned`.
#
# QLoRA in one paragraph: the base weights are frozen and stored in 4-bit NF4
# (about 1 GB for 1.5B parameters instead of 3 GB in fp16); small low-rank
# matrices A (r x d) and B (d x r) are added to every linear layer and only they
# are trained, so the trainable part is well under 1% of the parameters and fits
# a free 16 GB T4 together with 2k-token prompts.

# %%
# !pip install -q peft bitsandbytes

# %%
import glob
import json
import math
import os
import random
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # "Qwen/Qwen2.5-3B-Instruct" also fits a T4 (slower)
MAX_LEN = 2048                          # longest eval prompt is ~1.9k tokens
EPOCHS = 2
LR = 2e-4
GRAD_ACCUM = 8                          # effective batch size 8
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
GEN_BATCH = 4
MAX_NEW_TOKENS = 192
SEED = 0
OUT = "/kaggle/working" if os.path.exists("/kaggle/working") else "outputs"
os.makedirs(OUT, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True) or glob.glob(f"data/sft/{name}")
    assert hits, f"{name} not found: add the data/sft dataset to the notebook"
    return hits[0]


def load(name):
    with open(find(name), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


train_rows, val_rows, eval_rows = load("train.jsonl"), load("val.jsonl"), load("eval_prompts.jsonl")
print(len(train_rows), "train", len(val_rows), "val", len(eval_rows), "eval prompts")

# %% [markdown]
# ## Model in 4-bit and LoRA adapters
# T4 GPUs have no bfloat16, so compute runs in float16.

# %%
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"  # for batched generation; training uses batch size 1
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0},
                                             torch_dtype=torch.float16)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# %% [markdown]
# ## Tokenisation: loss only on the answer
# The prompt tokens get label -100, so the model is trained to produce the
# JSON answer, not to reproduce the passages.


# %%
def prompt_text(row):
    return tok.apply_chat_template([{"role": "system", "content": row["system"]},
                                    {"role": "user", "content": row["prompt"]}],
                                   tokenize=False, add_generation_prompt=True)


def encode(row):
    p = tok(prompt_text(row), add_special_tokens=False)["input_ids"]
    t = tok(row["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = (p + t)[:MAX_LEN]
    labels = ([-100] * len(p) + t)[:MAX_LEN]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class Rows(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.items = [encode(r) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: [v] * (n - len(xs)) + xs  # noqa: E731  left padding
    return {"input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch])}


train_ds, val_ds = Rows(train_rows), Rows(val_rows)
print("truncated examples:", sum(len(x["input_ids"]) == MAX_LEN for x in train_ds.items))

# %% [markdown]
# ## Training

# %%
total_steps = math.ceil(len(train_ds) / GRAD_ACCUM) * EPOCHS
args = TrainingArguments(
    output_dir=f"{OUT}/checkpoints", num_train_epochs=EPOCHS, learning_rate=LR,
    per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=GRAD_ACCUM,
    lr_scheduler_type="cosine", warmup_steps=max(1, round(0.03 * total_steps)), weight_decay=0.0, fp16=True,
    logging_steps=10, eval_strategy="epoch", save_strategy="no", report_to="none",
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit", seed=SEED,
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate)
t0 = time.time()
trainer.train()
train_minutes = (time.time() - t0) / 60
model.save_pretrained(f"{OUT}/adapter")
tok.save_pretrained(f"{OUT}/adapter")
log = {"model": MODEL, "epochs": EPOCHS, "lr": LR, "effective_batch": GRAD_ACCUM, "lora_r": LORA_R,
       "lora_alpha": LORA_ALPHA, "max_len": MAX_LEN, "n_train": len(train_ds), "n_val": len(val_ds),
       "train_minutes": round(train_minutes, 1), "gpu": torch.cuda.get_device_name(0),
       "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
       "history": trainer.state.log_history}
with open(f"{OUT}/train_log.json", "w") as f:
    json.dump(log, f, indent=1)
print(f"trained in {train_minutes:.1f} min")

# %% [markdown]
# ## Generation: base model vs fine-tuned, same prompts, greedy decoding
# The base model is the same weights with the adapter switched off.


# %%
@torch.no_grad()
def generate(rows, path):
    model.eval()
    model.config.use_cache = True
    t0 = time.time()
    with open(path, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), GEN_BATCH):
            batch = rows[i:i + GEN_BATCH]
            enc = tok([prompt_text(r) for r in batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(0)
            out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            for r, seq in zip(batch, out):
                text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                f.write(json.dumps({"id": r["id"], "output": text}, ensure_ascii=False) + "\n")
            print(f"{path}: {i + len(batch)}/{len(rows)}", end="\r")
    print(f"\n{path} in {(time.time() - t0) / 60:.1f} min")


generate(eval_rows, f"{OUT}/predictions_qlora.jsonl")
with model.disable_adapter():
    generate(eval_rows, f"{OUT}/predictions_base.jsonl")

# %%
# quick look before downloading
for name in ("base", "qlora"):
    with open(f"{OUT}/predictions_{name}.jsonl", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]
    ok = 0
    for p in preds:
        try:
            json.loads(p["output"][p["output"].index("{"):p["output"].rindex("}") + 1])
            ok += 1
        except ValueError:
            pass
    print(name, f"valid JSON {ok}/{len(preds)}", "| example:", preds[0]["output"][:200])
