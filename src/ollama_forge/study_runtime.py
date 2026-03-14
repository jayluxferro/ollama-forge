"""Transformer-backed runtime for generic ablation studies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _choose_device(requested: str, torch_module) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_from_name(name: str, torch_module):
    mapping = {
        "float32": torch_module.float32,
        "float": torch_module.float32,
        "float16": torch_module.float16,
        "half": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
    }
    return mapping.get(name.lower(), torch_module.float32)


def _extract_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers
    raise ValueError("Unsupported architecture: cannot locate transformer layers")


def _get_attention_module(layer: Any) -> Any:
    for attr in ("self_attn", "attn", "attention"):
        module = getattr(layer, attr, None)
        if module is not None:
            return module
    raise ValueError("Layer has no supported attention module")


def _get_ffn_module(layer: Any) -> Any:
    for attr in ("mlp", "feed_forward", "ffn"):
        module = getattr(layer, attr, None)
        if module is not None:
            return module
    raise ValueError("Layer has no supported FFN module")


def _get_output_projection(layer: Any, target: str) -> Any:
    if target == "attention":
        attention = _get_attention_module(layer)
        for attr in ("o_proj", "out_proj", "c_proj"):
            module = getattr(attention, attr, None)
            if module is not None and hasattr(module, "weight"):
                return module
        raise ValueError("Attention module has no supported output projection")
    if target == "mlp":
        ffn = _get_ffn_module(layer)
        for attr in ("down_proj", "out_proj", "proj"):
            module = getattr(ffn, attr, None)
            if module is not None and hasattr(module, "weight"):
                return module
        if hasattr(ffn, "weight"):
            return ffn
        raise ValueError("FFN module has no supported output projection")
    raise ValueError(f"Unsupported low-rank target {target!r}")


def _get_embedding_module(model: Any) -> Any:
    for path in (
        ("model", "embed_tokens"),
        ("transformer", "wte"),
    ):
        node = model
        ok = True
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                ok = False
                break
        if ok:
            return node
    getter = getattr(model, "get_input_embeddings", None)
    if callable(getter):
        embeddings = getter()
        if embeddings is not None:
            return embeddings
    raise ValueError("Unsupported architecture: cannot locate embedding module")


def _config_value(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        for name in names:
            value = getattr(text_config, name, None)
            if isinstance(value, int):
                return value
    return None


def _replace_first_output(output: Any, new_first: Any) -> Any:
    if isinstance(output, tuple):
        return (new_first, *output[1:])
    return new_first


def _zero_output_like(output: Any):
    if isinstance(output, tuple):
        first = output[0]
        return (first * 0, *output[1:])
    return output * 0


def _macro_f1(predictions: list[int], references: list[int]) -> float:
    labels = sorted(set(predictions) | set(references))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(int(p == label and r == label) for p, r in zip(predictions, references, strict=True))
        fp = sum(int(p == label and r != label) for p, r in zip(predictions, references, strict=True))
        fn = sum(int(p != label and r == label) for p, r in zip(predictions, references, strict=True))
        if tp == 0 and fp == 0 and fn == 0:
            scores.append(0.0)
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def _slice_dataset(dataset: Any, start: int, end: int) -> list[dict[str, Any]]:
    batch = dataset[start:end]
    if isinstance(batch, list):
        return batch
    if isinstance(batch, dict):
        keys = list(batch.keys())
        length = len(batch[keys[0]]) if keys else 0
        return [{key: batch[key][idx] for key in keys} for idx in range(length)]
    return list(batch)


def _last_token_index(attention_mask) -> int:
    return int(attention_mask.sum().item() - 1)


@dataclass
class StudyModelHandle:
    model: Any
    tokenizer: Any
    task: str
    architecture: str
    device: str
    num_layers: int
    num_heads: int
    hidden_size: int
    _restore_actions: list[Callable[[], None]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
        }

    def restore(self) -> None:
        while self._restore_actions:
            action = self._restore_actions.pop()
            action()

    def remove_layer(self, layer_index: int, strength: float = 1.0) -> None:
        layer = _extract_layers(self.model)[layer_index]
        strength = max(0.0, min(1.0, float(strength)))

        def hook(_module, args, output):
            hidden_states = args[0]
            if strength == 1.0:
                return _replace_first_output(output, hidden_states)
            current = output[0] if isinstance(output, tuple) else output
            blended = current * (1.0 - strength) + hidden_states * strength
            return _replace_first_output(output, blended)

        handle = layer.register_forward_hook(hook)
        self._restore_actions.append(handle.remove)

    def ablate_ffn(self, layer_index: int, strength: float = 1.0) -> None:
        ffn = _get_ffn_module(_extract_layers(self.model)[layer_index])
        strength = max(0.0, min(1.0, float(strength)))

        def hook(_module, _args, output):
            if strength == 1.0:
                return _zero_output_like(output)
            current = output[0] if isinstance(output, tuple) else output
            scaled = current * (1.0 - strength)
            return _replace_first_output(output, scaled)

        handle = ffn.register_forward_hook(hook)
        self._restore_actions.append(handle.remove)

    def ablate_embedding_range(self, start: int, end: int, strength: float = 1.0) -> None:
        embeddings = _get_embedding_module(self.model)
        strength = max(0.0, min(1.0, float(strength)))

        def hook(_module, _args, output):
            mutated = output.clone()
            mutated[..., start:end] = mutated[..., start:end] * (1.0 - strength)
            return mutated

        handle = embeddings.register_forward_hook(hook)
        self._restore_actions.append(handle.remove)

    def prune_head(self, layer_index: int, head_index: int, strength: float = 1.0) -> None:
        layer = _extract_layers(self.model)[layer_index]
        attention = _get_attention_module(layer)
        strength = max(0.0, min(1.0, float(strength)))
        head_dim = max(self.hidden_size // max(self.num_heads, 1), 1)
        start = head_index * head_dim
        end = start + head_dim
        touched = []
        for attr in ("q_proj", "k_proj", "v_proj"):
            module = getattr(attention, attr, None)
            if module is None or not hasattr(module, "weight"):
                continue
            touched.append((module.weight, slice(start, min(end, module.weight.shape[0])), None))
            if getattr(module, "bias", None) is not None:
                touched.append((module.bias, slice(start, min(end, module.bias.shape[0])), None))
        out_proj = (
            getattr(attention, "o_proj", None)
            or getattr(attention, "out_proj", None)
            or getattr(attention, "c_proj", None)
        )
        if out_proj is not None and hasattr(out_proj, "weight"):
            touched.append((out_proj.weight, None, slice(start, min(end, out_proj.weight.shape[1]))))
        if not touched:
            raise NotImplementedError("Head pruning is only implemented for attention modules with q/k/v/o projections")

        restore_actions: list[Callable[[], None]] = []
        for tensor, row_slice, col_slice in touched:
            data = tensor.data
            if row_slice is not None:
                backup = data[row_slice].detach().clone()
                data[row_slice] = data[row_slice] * (1.0 - strength)
                restore_actions.append(lambda t=tensor, b=backup, rs=row_slice: t.data[rs].copy_(b))
            elif col_slice is not None:
                backup = data[:, col_slice].detach().clone()
                data[:, col_slice] = data[:, col_slice] * (1.0 - strength)
                restore_actions.append(lambda t=tensor, b=backup, cs=col_slice: t.data[:, cs].copy_(b))
        self._restore_actions.append(lambda: [action() for action in reversed(restore_actions)])

    def apply_low_rank_adapter(
        self,
        layer_index: int,
        target: str = "mlp",
        rank: int = 1,
        strength: float = 1.0,
    ) -> None:
        import torch

        layer = _extract_layers(self.model)[layer_index]
        module = _get_output_projection(layer, target)
        weight = module.weight.data
        if weight.ndim != 2:
            raise ValueError("Low-rank adapter target must be a 2D weight matrix")
        rank = max(1, min(int(rank), min(weight.shape)))
        strength = max(0.0, float(strength))
        backup = weight.detach().clone()
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        low_rank = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
        module.weight.data = (weight.float() - strength * low_rank).to(weight.dtype)
        self._restore_actions.append(lambda m=module, b=backup: m.weight.data.copy_(b))

    def capture_layer_last_token(
        self,
        prompt: str,
        *,
        layer_index: int,
        max_length: int = 256,
    ):
        import torch

        layer = _extract_layers(self.model)[layer_index]
        captured = {}

        def hook(_module, _args, output):
            current = output[0] if isinstance(output, tuple) else output
            captured["value"] = current.detach()
            return output

        handle = layer.register_forward_hook(hook)
        try:
            encodings = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            encodings = {key: value.to(self.device) for key, value in encodings.items()}
            with torch.no_grad():
                self.model(**encodings)
            if "value" not in captured:
                raise ValueError("Layer hook did not capture an activation")
            idx = _last_token_index(encodings["attention_mask"][0])
            return captured["value"][0, idx, :].detach().cpu()
        finally:
            handle.remove()

    def patch_layer_last_token(
        self,
        layer_index: int,
        replacement,
        *,
        strength: float = 1.0,
    ) -> None:

        layer = _extract_layers(self.model)[layer_index]
        strength = max(0.0, min(1.0, float(strength)))
        replacement = replacement.detach().clone()

        def hook(_module, _args, output):
            current = output[0] if isinstance(output, tuple) else output
            patched = current.clone()
            token_idx = patched.shape[1] - 1
            repl = replacement.to(patched.device, dtype=patched.dtype)
            patched[:, token_idx, :] = patched[:, token_idx, :] * (1.0 - strength) + repl * strength
            return _replace_first_output(output, patched)

        handle = layer.register_forward_hook(hook)
        self._restore_actions.append(handle.remove)


def load_study_model(model_cfg) -> StudyModelHandle:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("study run requires torch and transformers") from exc

    device = _choose_device(model_cfg.device, torch)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no CUDA device is available")
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise ValueError("MPS requested but not available")

    dtype = _dtype_from_name(model_cfg.dtype, torch)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name, trust_remote_code=model_cfg.trust_remote_code)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = AutoModelForSequenceClassification if model_cfg.task == "classification" else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        model_cfg.name,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        num_labels=(model_cfg.num_labels if model_cfg.task == "classification" else None),
    )
    model.to(device)
    model.eval()

    config = model.config
    layers = _extract_layers(model)
    num_layers = len(layers)
    num_heads = _config_value(config, "num_attention_heads", "n_head") or 1
    hidden_size = _config_value(config, "hidden_size", "n_embd", "d_model") or 1
    return StudyModelHandle(
        model=model,
        tokenizer=tokenizer,
        task=model_cfg.task,
        architecture=type(model).__name__,
        device=device,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_size,
    )


def load_study_dataset(dataset_cfg):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("study run requires the datasets package") from exc

    path = Path(dataset_cfg.name)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in {".jsonl", ".json"}:
            return load_dataset("json", data_files=str(path), split="train")
        if suffix in {".txt", ".text"}:
            return load_dataset("text", data_files=str(path), split="train")
        if suffix == ".csv":
            return load_dataset("csv", data_files=str(path), split="train")
        raise ValueError(f"Unsupported local dataset file type: {path.suffix}")

    kwargs = {"path": dataset_cfg.name, "split": dataset_cfg.split}
    if dataset_cfg.subset:
        kwargs["name"] = dataset_cfg.subset
    return load_dataset(**kwargs)


class StudyEvaluator:
    def __init__(
        self,
        *,
        handle: StudyModelHandle,
        dataset: Any,
        metrics: list[str] | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        max_samples: int | None = None,
        text_column: str = "text",
        label_column: str = "label",
    ) -> None:
        self.handle = handle
        self.dataset = dataset
        self.metrics = metrics or (["perplexity"] if handle.task == "causal_lm" else ["accuracy", "f1"])
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_samples = max_samples
        self.text_column = text_column
        self.label_column = label_column

    def evaluate(self) -> dict[str, float]:
        if self.handle.task == "classification":
            return self._evaluate_classification()
        return self._evaluate_causal_lm()

    def _evaluate_causal_lm(self) -> dict[str, float]:
        from ollama_forge.study_metrics import effective_rank, mean_token_entropy

        dataset = self.dataset
        limit = min(len(dataset), self.max_samples) if self.max_samples is not None else len(dataset)
        total_loss = 0.0
        total_tokens = 0
        entropy_values: list[float] = []
        rank_values: list[float] = []
        for start in range(0, limit, self.batch_size):
            rows = _slice_dataset(dataset, start, min(start + self.batch_size, limit))
            texts = [str(row[self.text_column]) for row in rows]
            encodings = self.handle.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            encodings = {key: value.to(self.handle.device) for key, value in encodings.items()}
            outputs = self.handle.model(**encodings, labels=encodings["input_ids"])
            attention_mask = encodings["attention_mask"]
            num_tokens = int(attention_mask[:, 1:].sum().item())
            total_loss += float(outputs.loss.item()) * num_tokens
            total_tokens += num_tokens
            logits = outputs.logits.detach().cpu()
            if "avg_token_entropy" in self.metrics:
                entropy_values.append(mean_token_entropy(logits))
            if "effective_rank" in self.metrics:
                flat = logits.reshape(-1, logits.shape[-1])[: min(128, logits.numel() // logits.shape[-1])]
                rank_values.append(effective_rank(flat))
        avg_loss = total_loss / max(total_tokens, 1)
        results = {"perplexity": math.exp(avg_loss)} if "perplexity" in self.metrics else {}
        if entropy_values:
            results["avg_token_entropy"] = float(sum(entropy_values) / len(entropy_values))
        if rank_values:
            results["effective_rank"] = float(sum(rank_values) / len(rank_values))
        return results

    def _evaluate_classification(self) -> dict[str, float]:
        import torch

        from ollama_forge.study_metrics import classification_logit_margin

        dataset = self.dataset
        limit = min(len(dataset), self.max_samples) if self.max_samples is not None else len(dataset)
        predictions: list[int] = []
        references: list[int] = []
        margins: list[float] = []
        for start in range(0, limit, self.batch_size):
            rows = _slice_dataset(dataset, start, min(start + self.batch_size, limit))
            texts = [str(row[self.text_column]) for row in rows]
            labels = [int(row[self.label_column]) for row in rows]
            encodings = self.handle.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            encodings = {key: value.to(self.handle.device) for key, value in encodings.items()}
            with torch.no_grad():
                logits = self.handle.model(**encodings).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            references.extend(labels)
            if "logit_margin" in self.metrics:
                margins.append(classification_logit_margin(logits.detach().cpu()))
        results: dict[str, float] = {}
        if "accuracy" in self.metrics:
            correct = sum(int(p == r) for p, r in zip(predictions, references, strict=True))
            results["accuracy"] = correct / max(len(references), 1)
        if "f1" in self.metrics:
            results["f1"] = _macro_f1(predictions, references)
        if margins:
            results["logit_margin"] = float(sum(margins) / len(margins))
        return results
