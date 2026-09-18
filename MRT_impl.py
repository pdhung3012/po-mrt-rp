# pip install -U bitsandbytes peft accelerate transformers datasets peft nltk
import os
import torch
import pandas as pd
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "../Qwen2.5-7B-Instruct/"
folder_output = "../weights_replicate/mrt/"

fp_file_tuning_train = "../results_replicate/train.csv"
fp_file_tuning_valid = "../results_replicate/valid.csv"
num_samples=10
df_train = pd.read_csv(fp_file_tuning_train, dtype=str, keep_default_na=False, na_filter=False).head(num_samples)
df_valid = pd.read_csv(fp_file_tuning_valid, dtype=str, keep_default_na=False, na_filter=False).head(num_samples)

for df in (df_train, df_valid):
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)

train_ds = Dataset.from_pandas(df_train[["prompt", "response"]], preserve_index=False)
valid_ds = Dataset.from_pandas(df_valid[["prompt", "response"]], preserve_index=False)

real_model_name = model_name.split("/")[-2]
fop_output_model = folder_output + real_model_name + "/"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

MAX_LEN = 1024

def format_prompt(p: str) -> str:
    return f"### Instruction:\n{p}\n\n### Response:\n"

def tokenize_prompt_response(example):
    prompt_text = format_prompt(example["prompt"])
    response_text = example["response"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    resp_ids   = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    # Truncate to MAX_LEN while prioritizing keeping response
    if len(prompt_ids) + len(resp_ids) > MAX_LEN:
        if len(resp_ids) >= MAX_LEN:
            resp_ids = resp_ids[:MAX_LEN]
            prompt_ids = []
        else:
            keep_prompt = MAX_LEN - len(resp_ids)
            prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = prompt_ids + resp_ids
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(prompt_ids)) + resp_ids  # CE loss only on response if you ever use it

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_tokenized = train_ds.map(tokenize_prompt_response, remove_columns=train_ds.column_names)
valid_tokenized = valid_ds.map(tokenize_prompt_response, remove_columns=valid_ds.column_names)

# 4-bit load
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

# training-friendly
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=fop_output_model,
    per_device_train_batch_size=1,      # MRT is expensive; start with 1
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=torch.cuda.is_available(),
    optim="paged_adamw_8bit",
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# -----------------------------
# MRT Trainer (expected 1-BLEU)
# -----------------------------
class MRTBleuTrainer(Trainer):
    """
    Minimum Risk Training:
      loss = E_q[ 1 - BLEU(y_hat, y_ref) ]
    where q is softmax over candidate log-probs: q_i ∝ exp(logp_i / tau)

    Backprop flows through logp_i, not through BLEU.
    """
    def __init__(
        self,
        *args,
        num_candidates: int = 4,         # K
        gen_max_new_tokens: int = 256,
        gen_temperature: float = 0.8,
        gen_top_p: float = 0.95,
        tau: float = 1.0,                # risk temperature for q softmax
        bleu_eps: float = 1e-6,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_candidates = int(num_candidates)
        self.gen_max_new_tokens = int(gen_max_new_tokens)
        self.gen_temperature = float(gen_temperature)
        self.gen_top_p = float(gen_top_p)
        self.tau = float(tau)
        self.bleu_eps = float(bleu_eps)
        self._smooth = SmoothingFunction().method1

    def _decode_ref_hyp(self, ref_ids: torch.Tensor, hyp_ids: torch.Tensor) -> tuple[list[str], list[str]]:
        ref_text = self.tokenizer.decode(ref_ids.tolist(), skip_special_tokens=True).strip()
        hyp_text = self.tokenizer.decode(hyp_ids.tolist(), skip_special_tokens=True).strip()
        # simple whitespace tokenization for BLEU
        return ref_text.split(), hyp_text.split()

    def _sentence_bleu_safe(self, ref_tokens: list[str], hyp_tokens: list[str]) -> float:
        if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
            return 0.0
        return float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=self._smooth))

    def _candidate_logprobs(self, model, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """
        full_ids: (K, L) = prompt + candidate_response
        returns: (K,) sum log p(candidate_response | prompt)
        """
        outputs = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids))
        logits = outputs.logits  # (K, L, V)

        logp = torch.log_softmax(logits, dim=-1)
        target = full_ids[:, 1:]                    # (K, L-1)
        token_logp = logp[:, :-1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (K, L-1)

        # response token positions start at token index = prompt_len
        # those correspond to token_logp positions starting at (prompt_len - 1)
        start = max(prompt_len - 1, 0)
        return token_logp[:, start:].sum(dim=1)     # (K,)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]      # (B, T)
        labels    = inputs["labels"]         # (B, T)

        device = input_ids.device
        batch_losses = []

        # Generate candidates without grads (cheaper); score with grads
        # (batch size is typically 1 here, but we support looping over B)
        for b in range(input_ids.size(0)):
            ids = input_ids[b]
            lab = labels[b]

            # find where response starts: first index with label != -100
            nz = torch.nonzero(lab != -100, as_tuple=False)
            if nz.numel() == 0:
                # no response tokens -> skip (rare)
                continue
            prompt_len = int(nz[0].item())

            prompt_ids = ids[:prompt_len].unsqueeze(0)  # (1, prompt_len)

            # reference response ids (labels != -100)
            ref_ids = lab[lab != -100]

            # ---- sample K candidate responses ----
            # temporarily allow cache during generation for speed
            old_cache = getattr(model.config, "use_cache", False)
            try:
                model.config.use_cache = True
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=torch.ones_like(prompt_ids),
                        do_sample=True,
                        temperature=self.gen_temperature,
                        top_p=self.gen_top_p,
                        num_return_sequences=self.num_candidates,
                        max_new_tokens=self.gen_max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            finally:
                model.config.use_cache = old_cache

            # gen: (K, prompt_len + gen_len)
            # make sure it includes prompt prefix; HF does by default
            K = gen.size(0)

            # ---- compute BLEU per candidate (no grads) ----
            bleu_scores = []
            for k in range(K):
                hyp_full = gen[k]
                hyp_resp = hyp_full[prompt_len:]  # generated continuation
                ref_tok, hyp_tok = self._decode_ref_hyp(ref_ids.detach().cpu(), hyp_resp.detach().cpu())
                print('ref tok {}\nhypt tok {}'.format(ref_tok,hyp_tok))
                bleu_scores.append(self._sentence_bleu_safe(ref_tok, hyp_tok))

            bleu_t = torch.tensor(bleu_scores, device=device, dtype=torch.float32)  # (K,)
            risk_t = 1.0 - bleu_t  # (K,)

            # ---- compute differentiable log-prob of each candidate ----
            gen = gen.to(device)
            cand_logp = self._candidate_logprobs(model, gen, prompt_len=prompt_len)  # (K,)

            # q_i ∝ exp(logp_i / tau)
            q = torch.softmax(cand_logp / max(self.tau, 1e-6), dim=0)  # (K,)

            # expected risk
            loss_b = (q * risk_t.to(q.dtype)).sum()
            batch_losses.append(loss_b)

        if not batch_losses:
            # fallback: if something weird happens, return 0
            loss = torch.zeros([], device=device, requires_grad=True)
        else:
            loss = torch.stack(batch_losses).mean()

        # optional logging
        if self.state.global_step % max(self.args.logging_steps, 1) == 0 and batch_losses:
            with torch.no_grad():
                avg_bleu = float((1.0 - torch.stack(batch_losses)).clamp(min=0).mean().cpu())
                print('avg bleu {}'.format(avg_bleu))
            self.log({"mrt_loss": float(loss.detach().cpu())})

        return (loss, None) if return_outputs else loss


trainer = MRTBleuTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,

    # MRT knobs
    num_candidates=2,        # K: try 2-8
    gen_max_new_tokens=2048,  # adjust to your response length
    gen_temperature=0.8,
    gen_top_p=0.95,
    tau=1.0,                 # softmax temperature over candidates
)

trainer.train()
trainer.save_model(fop_output_model)
