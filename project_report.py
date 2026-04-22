"""
project_report.py

Generate a detailed, human-readable project report for the
Mini_Generative_Pretrained_Transformer repository.

Usage:
    python project_report.py
    python project_report.py --json
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def bytes_to_human(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def parse_step_from_checkpoint_name(name: str) -> int:
    # Expected format: ckpt_step_<number>.pt
    try:
        return int(name.split("_")[-1].split(".")[0])
    except (ValueError, IndexError):
        return -1


def read_requirements(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        lines.append(text)
    return lines


def get_python_and_torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    import sys

    info["python_version"] = sys.version.split(" ")[0]
    info["platform"] = sys.platform

    try:
        import torch  # type: ignore

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        else:
            info["cuda_device_count"] = 0
            info["cuda_devices"] = []
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)

    return info


def summarize_checkpoints(checkpoints_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "exists": checkpoints_dir.exists(),
        "count": 0,
        "latest": None,
        "total_size_bytes": 0,
        "files": [],
    }

    if not checkpoints_dir.exists():
        return result

    ckpt_files = sorted([p for p in checkpoints_dir.glob("ckpt_step_*.pt")], key=lambda p: parse_step_from_checkpoint_name(p.name))
    result["count"] = len(ckpt_files)

    entries = []
    total = 0
    for p in ckpt_files:
        size = p.stat().st_size
        total += size
        entries.append(
            {
                "name": p.name,
                "step": parse_step_from_checkpoint_name(p.name),
                "size_bytes": size,
                "size_human": bytes_to_human(size),
                "modified": dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            }
        )

    result["files"] = entries
    result["total_size_bytes"] = total

    if entries:
        result["latest"] = entries[-1]

    return result


def summarize_data_files(root: Path, train_name: str, val_name: str, tokenizer_name: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    data_paths = {
        "train_bin": root / train_name,
        "val_bin": root / val_name,
        "tokenizer_json": root / tokenizer_name,
    }

    for key, path in data_paths.items():
        exists = path.exists()
        entry = {
            "path": str(path.name),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else 0,
            "size_human": bytes_to_human(path.stat().st_size) if exists else "0 B",
        }
        if exists and path.suffix == ".bin":
            # uint16 tokens => 2 bytes per token
            entry["estimated_tokens"] = path.stat().st_size // 2
        out[key] = entry

    return out


def model_breakdown(config_module) -> Dict[str, Any]:
    # Local imports keep startup robust when only static info is needed.
    from model import GPTLanguageModel  # type: ignore

    model = GPTLanguageModel(config_module)

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    block0 = model.blocks[0]
    per_block = {
        "attn_q_proj": block0.attn.q_proj.weight.numel(),
        "attn_k_proj": block0.attn.k_proj.weight.numel(),
        "attn_v_proj": block0.attn.v_proj.weight.numel(),
        "attn_o_proj": block0.attn.o_proj.weight.numel(),
        "ffn_w1": block0.ffn.w1.weight.numel(),
        "ffn_w2": block0.ffn.w2.weight.numel(),
        "ffn_w_out": block0.ffn.w_out.weight.numel(),
        "norm1_scale": block0.norm1.scale.numel(),
        "norm2_scale": block0.norm2.scale.numel(),
    }

    per_block_attn = per_block["attn_q_proj"] + per_block["attn_k_proj"] + per_block["attn_v_proj"] + per_block["attn_o_proj"]
    per_block_ffn = per_block["ffn_w1"] + per_block["ffn_w2"] + per_block["ffn_w_out"]
    per_block_norms = per_block["norm1_scale"] + per_block["norm2_scale"]
    per_block_total = sum(per_block.values())

    breakdown = {
        "trainable_total": trainable_total,
        "total_parameters": total,
        "weight_tying": "token_embed.weight is shared with lm_head.weight",
        "high_level": {
            "token_embed_and_lm_head_tied": model.token_embed.weight.numel(),
            "all_transformer_blocks": per_block_total * len(model.blocks),
            "final_norm": model.norm_f.scale.numel(),
        },
        "per_block": per_block,
        "per_block_grouped": {
            "attention": per_block_attn,
            "ffn": per_block_ffn,
            "norms": per_block_norms,
            "total": per_block_total,
        },
        "all_blocks_grouped": {
            "attention": per_block_attn * len(model.blocks),
            "ffn": per_block_ffn * len(model.blocks),
            "norms": per_block_norms * len(model.blocks),
            "total": per_block_total * len(model.blocks),
        },
    }

    return breakdown


def gather_report(root: Path) -> Dict[str, Any]:
    config = importlib.import_module("config")

    project_scripts = [
        "prepare_data.py",
        "training.py",
        "generate.py",
        "model.py",
        "tokenizer.py",
        "dataset.py",
        "config.py",
    ]

    report: Dict[str, Any] = {
        "project": {
            "name": root.name,
            "root": str(root),
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        },
        "environment": get_python_and_torch_info(),
        "architecture": {
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_kv_heads": config.n_kv_heads,
            "dropout": config.dropout,
            "ffn_mult": config.ffn_mult,
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "attention_type": "Grouped-Query Attention (GQA)",
            "position_encoding": "RoPE",
            "normalization": "RMSNorm",
            "ffn_activation": "SwiGLU",
        },
        "training_config": {
            "device": config.device,
            "batch_size": config.batch_size,
            "max_iters": config.max_iters,
            "learning_rate": config.learning_rate,
            "eval_iters": config.eval_iters,
            "eval_interval": config.eval_interval,
            "checkpoint_interval": config.checkpoint_interval,
            "enable_profiling": getattr(config, "ENABLE_PROFILING", None),
            "profiling_window": getattr(config, "PROFILING_WINDOW", None),
        },
        "model_parameters": model_breakdown(config),
        "data_artifacts": summarize_data_files(root, config.TRAIN_BIN, config.VAL_BIN, config.TOKENIZER_PATH),
        "checkpoints": summarize_checkpoints(root / "checkpoints"),
        "dependencies": read_requirements(root / "requirements.txt"),
        "scripts": {
            name: (root / name).exists() for name in project_scripts
        },
        "how_to_run": {
            "prepare_data": "python prepare_data.py",
            "train": "python training.py",
            "generate": "python generate.py --prompt \"Once upon a time\" --max-tokens 100",
            "report": "python project_report.py",
        },
        "notes": [
            "Dataset binaries use uint16 token IDs (2 bytes/token).",
            "Dataset loading is memory-mapped in dataset.py via np.memmap.",
            "Tokenizer uses HuggingFace tokenizers (Rust backend).",
            "Embedding and LM head are weight-tied in model.py.",
        ],
    }

    return report


def print_human_report(report: Dict[str, Any]) -> None:
    def h1(title: str) -> None:
        print("\n" + "=" * 88)
        print(title)
        print("=" * 88)

    def h2(title: str) -> None:
        print("\n" + "-" * 40)
        print(title)
        print("-" * 40)

    h1("PROJECT OVERVIEW REPORT")
    print(f"Project Name        : {report['project']['name']}")
    print(f"Root Path           : {report['project']['root']}")
    print(f"Generated At        : {report['project']['generated_at']}")

    h2("Environment")
    env = report["environment"]
    print(f"Python Version      : {env.get('python_version', 'N/A')}")
    print(f"Platform            : {env.get('platform', 'N/A')}")
    print(f"Torch Version       : {env.get('torch_version', 'N/A')}")
    print(f"CUDA Available      : {env.get('cuda_available', 'N/A')}")
    if env.get("cuda_devices"):
        for idx, name in enumerate(env["cuda_devices"]):
            print(f"CUDA Device {idx:<8}: {name}")

    h2("Architecture")
    arch = report["architecture"]
    for k in [
        "n_embd",
        "n_layer",
        "n_head",
        "n_kv_heads",
        "dropout",
        "ffn_mult",
        "vocab_size",
        "block_size",
        "attention_type",
        "position_encoding",
        "normalization",
        "ffn_activation",
    ]:
        print(f"{k:<20}: {arch[k]}")

    h2("Training Config")
    tc = report["training_config"]
    for k, v in tc.items():
        print(f"{k:<20}: {v}")

    h2("Parameter Summary")
    mp = report["model_parameters"]
    print(f"Total Parameters    : {mp['total_parameters']:,}")
    print(f"Trainable Params    : {mp['trainable_total']:,}")
    print(f"Weight Tying        : {mp['weight_tying']}")

    print("\nHigh-level:")
    for k, v in mp["high_level"].items():
        print(f"  - {k:<30}: {v:,}")

    print("\nPer-block grouped:")
    for k, v in mp["per_block_grouped"].items():
        print(f"  - {k:<30}: {v:,}")

    print("\nPer-block detailed:")
    for k, v in mp["per_block"].items():
        print(f"  - {k:<30}: {v:,}")

    h2("Data Artifacts")
    da = report["data_artifacts"]
    for key in ["train_bin", "val_bin", "tokenizer_json"]:
        item = da[key]
        print(f"{key:<20}: exists={item['exists']} size={item['size_human']} ({item['size_bytes']:,} bytes)")
        if "estimated_tokens" in item:
            print(f"{'':<20}  estimated_tokens={item['estimated_tokens']:,}")

    h2("Checkpoints")
    ck = report["checkpoints"]
    print(f"Checkpoints Exist   : {ck['exists']}")
    print(f"Checkpoint Count    : {ck['count']}")
    print(f"Total Ckpt Size     : {bytes_to_human(ck['total_size_bytes'])}")
    if ck["latest"]:
        latest = ck["latest"]
        print(
            "Latest              : "
            f"{latest['name']} | step={latest['step']} | size={latest['size_human']} | modified={latest['modified']}"
        )

    h2("Dependencies (requirements.txt)")
    for dep in report["dependencies"]:
        print(f"  - {dep}")

    h2("Core Scripts Presence")
    for name, exists in report["scripts"].items():
        print(f"{name:<20}: {exists}")

    h2("How To Run")
    for k, cmd in report["how_to_run"].items():
        print(f"{k:<20}: {cmd}")

    h2("Key Notes")
    for note in report["notes"]:
        print(f"  - {note}")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an explainability report for this project.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of human report.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    report = gather_report(root)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human_report(report)


if __name__ == "__main__":
    main()
