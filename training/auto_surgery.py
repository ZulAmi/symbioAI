"""
Automatic Model Surgery (LoRA/Adapter Fine-Tuning) for Self-Healing

This module implements targeted fine-tuning at the subnetwork/adapter level
using PEFT/LoRA for rapid remediation of failure modes detected in production.

Key features:
- Consumes curated failure datasets from monitoring.failure_monitor.
- Applies LoRA adapters to a base transformer via Hugging Face Transformers + PEFT.
- Supports evaluation via existing evaluation.benchmarks.
- Publishes updated artifacts to the ModelRegistry with version bumping.

Note: For demonstration, actual HF model loading is mocked configurable; you can
wire in a real model_name_or_path easily.
"""

from __future__ import annotations

import os
import json
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:  # Lazy optional imports to keep workspace healthy without heavy deps at import time
    import torch  # type: ignore
    from torch.utils.data import Dataset, DataLoader  # type: ignore
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore
except Exception:  # pragma: no cover - will import inside methods when needed
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    AutoModelForCausalLM = AutoTokenizer = get_linear_schedule_with_warmup = None  # type: ignore
    LoraConfig = get_peft_model = PeftModel = None  # type: ignore


@dataclass
class SurgeryConfig:
    base_model: str = "gpt2"  # or a local path
    lr: float = 2e-4
    batch_size: int = 8
    max_steps: int = 200
    warmup_steps: int = 20
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # e.g., ["c_attn", "c_proj"] for GPT2
    max_seq_len: int = 512
    gradient_accumulation: int = 1
    device: str = "auto"  # resolved at runtime to cuda/cpu
    output_dir: str = "artifacts/auto_surgery"
    save_every: int = 100
    seed: int = 42


class FailureDataset(Dataset):
    """A tiny dataset wrapper for failure samples."""

    def __init__(self, samples: List[Dict[str, Any]], tokenizer: Any, max_len: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Expect text inputs. If structured, you can serialize to a prompt.
        inp = s.get("input", {})
        target = s.get("output_target", {})
        prompt = inp.get("text") if isinstance(inp, dict) else json.dumps(inp)
        target_text = target.get("text") if isinstance(target, dict) else json.dumps(target)
        # Supervised fine-tuning format: prompt + expected continuation
        full = f"<|prompt|>\n{prompt}\n<|response|>\n{target_text}"[: 10_000]
        enc = self.tokenizer(
            full,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        # Labels = input_ids for causal LM
        enc["labels"] = enc["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in enc.items()}


class AutoModelSurgery:
    """Performs targeted LoRA adapter fine-tuning for self-healing."""

    def __init__(self, config: SurgeryConfig):
        self.cfg = config
        self.logger = logging.getLogger(__name__)
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def _load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        if AutoTokenizer is None:
            # Attempt lazy import here
            from transformers import AutoModelForCausalLM as _AM, AutoTokenizer as _AT, get_linear_schedule_with_warmup as _sched  # type: ignore
            from peft import LoraConfig as _LC, get_peft_model as _gpm  # type: ignore
            globals().update(
                AutoModelForCausalLM=_AM, AutoTokenizer=_AT, get_linear_schedule_with_warmup=_sched, LoraConfig=_LC, get_peft_model=_gpm
            )
        if torch is None:
            import torch as _torch  # type: ignore
            globals().update(torch=_torch)

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(self.cfg.base_model)

        peft_cfg = LoraConfig(
            r=self.cfg.lora_r,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=self.cfg.target_modules,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lora_model = get_peft_model(base_model, peft_cfg)
        device = self._resolve_device()
        lora_model.to(device)
        return lora_model, tokenizer

    def _resolve_device(self) -> str:
        if self.cfg.device == "auto":
            try:
                import torch as _torch  # type: ignore
                return "cuda" if _torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return self.cfg.device

    def train_on_failures(self, samples: List[Dict[str, Any]]) -> str:
        if not samples:
            raise ValueError("No samples provided for surgery training")

        model, tokenizer = self._load_model_and_tokenizer()
        ds = FailureDataset(samples, tokenizer, self.cfg.max_seq_len)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)
        total_steps = min(self.cfg.max_steps, max(self.cfg.warmup_steps + 1, len(dl) * 3))
        lr_sched = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, num_training_steps=total_steps
        )
        model.train()

        global_step = 0
        accum = self.cfg.gradient_accumulation
        device = self._resolve_device()
        scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))

        for epoch in range(9999):  # break when steps achieved
            for batch in dl:
                for k in batch:
                    batch[k] = batch[k].to(device)
                with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
                    out = model(**batch)
                    loss = out.loss / accum
                scaler.scale(loss).backward()
                if (global_step + 1) % accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    lr_sched.step()
                global_step += 1

                if global_step % 20 == 0:
                    self.logger.info(f"auto-surgery step={global_step} loss={loss.item() * accum:.4f}")

                if global_step % max(1, self.cfg.save_every) == 0 or global_step >= total_steps:
                    # Save adapter checkpoint
                    ckpt_dir = Path(self.cfg.output_dir) / f"lora_step_{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(ckpt_dir))
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        # Final artifact path
        final_dir = Path(self.cfg.output_dir) / f"lora_final_{int(time.time())}"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        return str(final_dir)

    def evaluate_adapter(self, adapter_dir: str, eval_fn) -> Dict[str, Any]:
        """Run a provided evaluation function that returns metrics dict."""
        try:
            metrics = eval_fn(adapter_dir)
            self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.warning(f"Evaluation failed: {e}")
            return {"status": "failed", "error": str(e)}

    def publish(self, adapter_dir: str) -> Dict[str, Any]:
        """
        Publish artifacts for consumption by the runtime. In production,
        this should version in the ModelRegistry and create a deploy spec.
        Here we return a descriptor.
        """
        desc = {
            "artifact_path": adapter_dir,
            "type": "lora_adapter",
            "base_model": self.cfg.base_model,
            "created_at": int(time.time()),
        }
        manifest = Path(adapter_dir) / "manifest.json"
        with open(manifest, "w") as f:
            json.dump(desc, f, indent=2)
        self.logger.info(f"Published adapter manifest: {manifest}")
        return desc
