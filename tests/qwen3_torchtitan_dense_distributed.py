# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from datasets import load_dataset
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from autoparallel.api import AutoParallel
from autoparallel.input_validation import ForwardInputs

TORCHTITAN_COMMIT = "52a292d2977690d407bd81781de932f7f7dc56c5"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
TOKENIZER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PINNED_C4_DATASET = "c4_pr482"
MODEL_FLAVOR = "0.6B"
SEQ_LEN = 4096
GLOBAL_BATCH_SIZE = 4
STEPS = 20
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQ_LEN


def _add_sibling_torchtitan_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    torchtitan_root = Path(
        os.environ.get("TORCHTITAN_ROOT", repo_root.parent / "torchtitan")
    ).resolve()
    if torchtitan_root.exists():
        sys.path.insert(0, str(torchtitan_root))
    return torchtitan_root


DEFAULT_TORCHTITAN_ROOT = _add_sibling_torchtitan_to_path()

from torchtitan.components.loss import CrossEntropyLoss  # noqa: E402
from torchtitan.config import (  # noqa: E402
    TORCH_DTYPE_MAP,
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims  # noqa: E402
from torchtitan.distributed.fsdp import (  # noqa: E402
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.hf_datasets import DatasetConfig  # noqa: E402
from torchtitan.hf_datasets.text_datasets import DATASETS  # noqa: E402
from torchtitan.models.qwen3.config_registry import qwen3_0_6b  # noqa: E402
from torchtitan.tools.logging import logger  # noqa: E402


def _load_pinned_c4(dataset_path: str):
    return load_dataset(
        dataset_path,
        name="en",
        split="train",
        streaming=True,
        revision=C4_REVISION,
    )


def _process_c4_text(sample):
    return sample["text"]


DATASETS[PINNED_C4_DATASET] = DatasetConfig(
    path="allenai/c4",
    loader=_load_pinned_c4,
    sample_processor=_process_c4_text,
)


def qwen3_0_6b_ce_baseline():
    config = qwen3_0_6b()
    config.loss = CrossEntropyLoss.Config()
    config.dataloader = replace(config.dataloader, dataset=PINNED_C4_DATASET)
    return config


def qwen3_0_6b_autoparallel():
    config = qwen3_0_6b_ce_baseline()
    config.model_spec = replace(
        config.model_spec,
        parallelize_fn=parallelize_autoparallel_qwen3,
    )
    return config


def parallelize_autoparallel_qwen3(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    if parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel Qwen3 does not support DDP yet")
    if parallel_dims.cp_enabled:
        raise ValueError("AutoParallel Qwen3 does not support CP yet")
    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel Qwen3 does not support PP yet")
    if parallel_dims.dp_shard != 2 or parallel_dims.tp != 2:
        raise ValueError(
            "This comparison requires the searched (dp=2, tp=2) mesh, got "
            f"(dp={parallel_dims.dp_shard}, tp={parallel_dims.tp})"
        )

    mesh_names = ["fsdp", "tp"]
    mesh = parallel_dims.get_mesh(mesh_names)

    def input_fn():
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (training.global_batch_size, training.seq_len),
            device=torch.device("cuda"),
        )
        positions = torch.arange(
            training.seq_len,
            dtype=torch.int32,
            device=torch.device("cuda"),
        ).repeat(training.global_batch_size, 1)
        return ForwardInputs(args=(tokens,), kwargs={"positions": positions})

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )
    input_sharding = (Shard(0), Replicate())
    output_sharding = (Shard(0), Replicate())

    with AutoParallel(
        model,
        input_fn,
        mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([input_sharding, input_sharding])
        autop.add_output_constraints([output_sharding])

        started = time.time()
        sharding_placement = autop.optimize_placement(verbose=False)
        logger.info(
            "AutoParallel searched the Qwen3 (dp=2, tp=2) placement in %.2fs",
            time.time() - started,
        )

        return autop.apply_placement(sharding_placement)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare TorchTitan Qwen3 0.6B training loss on one GPU against "
            "an AutoParallel-searched (dp=2, tp=2) placement."
        )
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    return parser.parse_args()


def _git_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _git_metadata(repo: Path) -> dict:
    origin_main = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=repo, text=True
    ).strip()
    merge_base = subprocess.check_output(
        ["git", "merge-base", "HEAD", "origin/main"], cwd=repo, text=True
    ).strip()
    changed_files = subprocess.check_output(
        ["git", "diff", "--name-status", f"{merge_base}..HEAD"],
        cwd=repo,
        text=True,
    ).splitlines()
    return {
        "commit": _git_revision(repo),
        "origin_main": origin_main,
        "merge_base": merge_base,
        "changed_files_from_merge_base": changed_files,
    }


def _validate_environment(args) -> tuple[Path, Path]:
    torchtitan_root = DEFAULT_TORCHTITAN_ROOT
    if _git_revision(torchtitan_root) != TORCHTITAN_COMMIT:
        raise RuntimeError(f"TorchTitan must be checked out at {TORCHTITAN_COMMIT}.")
    repo_root = Path(__file__).resolve().parents[1]
    dirty_files = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo_root, text=True
    ).splitlines()
    if dirty_files:
        raise RuntimeError(
            "Commit all AutoParallel changes before collecting reportable results: "
            f"{dirty_files}"
        )
    tokenizer_dir = torchtitan_root / "assets" / "hf" / "Qwen3-0.6B"
    required_tokenizer_files = (
        tokenizer_dir / "tokenizer.json",
        tokenizer_dir / "tokenizer_config.json",
    )
    missing = [str(path) for path in required_tokenizer_files if not path.exists()]
    if missing:
        raise RuntimeError(
            "Pinned Qwen3 tokenizer assets are missing: "
            f"{missing}. Download Qwen/Qwen3-0.6B revision {TOKENIZER_REVISION}."
        )
    work_dir = args.work_dir.resolve()
    output_dir = args.output_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return work_dir, output_dir


def _loss_compare_command(
    torchtitan_root: Path,
    work_dir: Path,
    output_dir: Path,
    steps: int,
) -> list[str]:
    module = Path(__file__).stem
    baseline_options = " ".join(
        (
            "--parallelism.data_parallel_shard_degree=1",
            "--parallelism.tensor_parallel_degree=1",
            f"--training.global_batch_size={GLOBAL_BATCH_SIZE}",
            f"--training.local_batch_size={GLOBAL_BATCH_SIZE}",
            f"--training.seq_len={SEQ_LEN}",
        )
    )
    test_options = " ".join(
        (
            "--parallelism.data_parallel_shard_degree=2",
            "--parallelism.tensor_parallel_degree=2",
            f"--training.global_batch_size={GLOBAL_BATCH_SIZE}",
            f"--training.local_batch_size={GLOBAL_BATCH_SIZE // 2}",
            f"--training.seq_len={SEQ_LEN}",
        )
    )
    return [
        sys.executable,
        str(torchtitan_root / "scripts" / "loss_compare.py"),
        ".",
        ".",
        f"--baseline-module={module}",
        "--baseline-config=qwen3_0_6b_ce_baseline",
        f"--test-module={module}",
        "--test-config=qwen3_0_6b_autoparallel",
        f"--baseline-options={baseline_options}",
        f"--test-options={test_options}",
        f"--steps={steps}",
        "--baseline-ngpus=1",
        "--test-ngpus=4",
        f"--job-dump-folder={work_dir / 'job'}",
        f"--output-folder={output_dir / 'loss_compare'}",
    ]


def _read_losses(job_dir: Path, scenario: str) -> dict[int, float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb_root = job_dir / f"tb_{scenario}"
    event_dirs = [path for path in tb_root.iterdir() if path.is_dir()]
    if len(event_dirs) != 1:
        raise RuntimeError(
            f"Expected one TensorBoard run under {tb_root}, found {event_dirs}."
        )
    accumulator = EventAccumulator(str(event_dirs[0]))
    accumulator.Reload()
    tag = "loss_metrics/global_avg_loss"
    if tag not in accumulator.Tags().get("scalars", []):
        raise RuntimeError(
            f"TensorBoard scalar {tag!r} is missing from {event_dirs[0]}."
        )
    return {event.step: event.value for event in accumulator.Scalars(tag)}


def _write_csv(
    path: Path,
    baseline_losses: dict[int, float],
    autoparallel_losses: dict[int, float],
) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "step",
                "tokens_seen",
                "torchtitan_1gpu_loss",
                "autoparallel_dp2_tp2_loss",
            )
        )
        for step in sorted(baseline_losses):
            writer.writerow(
                (
                    step,
                    step * TOKENS_PER_STEP,
                    repr(baseline_losses[step]),
                    repr(autoparallel_losses[step]),
                )
            )


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _write_plot(
    path: Path,
    baseline_losses: dict[int, float],
    autoparallel_losses: dict[int, float],
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1440, 900
    left, top, right, bottom = 130, 155, 70, 135
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    steps = sorted(baseline_losses)
    all_losses = [*baseline_losses.values(), *autoparallel_losses.values()]
    y_min, y_max = min(all_losses), max(all_losses)
    padding = max((y_max - y_min) * 0.08, 0.05)
    y_min -= padding
    y_max += padding

    def point(step: int, loss: float) -> tuple[float, float]:
        x_ratio = 0.0 if len(steps) == 1 else (step - steps[0]) / (steps[-1] - steps[0])
        y_ratio = (loss - y_min) / (y_max - y_min)
        return left + x_ratio * plot_width, top + (1.0 - y_ratio) * plot_height

    draw.text(
        (left, 36),
        "Qwen3 0.6B training loss",
        fill="#111827",
        font=_font(42, bold=True),
    )
    draw.text(
        (left, 92),
        "TorchTitan C4 | global batch 4 | sequence 4096 | deterministic seed 42",
        fill="#4B5563",
        font=_font(24),
    )

    grid_color = "#D1D5DB"
    axis_color = "#374151"
    for index in range(6):
        ratio = index / 5
        y = top + ratio * plot_height
        value = y_max - ratio * (y_max - y_min)
        draw.line((left, y, left + plot_width, y), fill=grid_color, width=1)
        label = f"{value:.3f}"
        draw.text(
            (left - 18 - draw.textlength(label, font=_font(20)), y - 12),
            label,
            fill=axis_color,
            font=_font(20),
        )
    x_ticks = sorted({steps[0], steps[-1], *steps[:: max(1, len(steps) // 5)]})
    for step in x_ticks:
        x, _ = point(step, y_min)
        draw.line((x, top, x, top + plot_height), fill=grid_color, width=1)
        label = str(step)
        draw.text(
            (x - draw.textlength(label, font=_font(20)) / 2, top + plot_height + 18),
            label,
            fill=axis_color,
            font=_font(20),
        )
    draw.line((left, top, left, top + plot_height), fill=axis_color, width=2)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill=axis_color,
        width=2,
    )

    curves = (
        ("TorchTitan 1 GPU", baseline_losses, "#2563EB"),
        ("AutoParallel DP=2, TP=2", autoparallel_losses, "#DC2626"),
    )
    for label, losses, color in curves:
        points = [point(step, losses[step]) for step in steps]
        draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)

    legend_x = left + plot_width - 420
    for index, (label, _losses, color) in enumerate(curves):
        y = top + 20 + index * 42
        draw.line((legend_x, y + 12, legend_x + 50, y + 12), fill=color, width=4)
        draw.text((legend_x + 66, y), label, fill="#111827", font=_font(22))

    x_label = f"Optimizer step ({TOKENS_PER_STEP:,} tokens per step)"
    draw.text(
        (
            left + (plot_width - draw.textlength(x_label, font=_font(22))) / 2,
            height - 78,
        ),
        x_label,
        fill=axis_color,
        font=_font(22),
    )
    draw.text(
        (left, top - 32),
        "Global average CE loss",
        fill=axis_color,
        font=_font(22),
    )
    note = "DP differs between runs; curves compare training behavior, not stepwise numeric parity."
    draw.text(
        (left, height - 38),
        note,
        fill="#6B7280",
        font=_font(18),
    )
    image.save(path, format="PNG", optimize=True)


def _environment() -> dict:
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": gpu_names,
    }


def _configuration(steps: int) -> dict:
    config = qwen3_0_6b_ce_baseline()
    training = asdict(config.training)
    training.update(
        {
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "steps": steps,
        }
    )
    training.pop("local_batch_size")
    return {
        "model": {
            "module": "torchtitan.models.qwen3",
            "config": "qwen3_0_6b",
            "flavor": MODEL_FLAVOR,
            "attention_backend": "sdpa",
        },
        "data": {
            "repo": "allenai/c4",
            "config": "en",
            "split": "train",
            "revision": C4_REVISION,
            "streaming": True,
            "sample_field": "text",
        },
        "tokenizer": {
            "repo": "Qwen/Qwen3-0.6B",
            "revision": TOKENIZER_REVISION,
        },
        "loss": "torchtitan.components.loss.CrossEntropyLoss",
        "optimizer": asdict(config.optimizer),
        "lr_scheduler": asdict(config.lr_scheduler),
        "training_common": training,
        "activation_checkpoint": asdict(config.activation_checkpoint),
        "compile": asdict(config.compile),
        "determinism": {
            "enabled": True,
            "seed": 42,
            "seed_checkpoint": True,
        },
        "baseline": {
            "gpus": 1,
            "data_parallel_shard_degree": 1,
            "tensor_parallel_degree": 1,
            "local_batch_size": GLOBAL_BATCH_SIZE,
        },
        "autoparallel": {
            "gpus": 4,
            "data_parallel_shard_degree": 2,
            "tensor_parallel_degree": 2,
            "local_batch_size": GLOBAL_BATCH_SIZE // 2,
            "mesh_dim_names": ["fsdp", "tp"],
            "input_placements": ["Shard(0)", "Replicate()"],
            "output_placements": ["Shard(0)", "Replicate()"],
        },
        "checkpoint": {
            "export_dtype": "bfloat16",
            "load_only": True,
        },
        "tokens_per_step": TOKENS_PER_STEP,
    }


def _write_report(
    path: Path,
    *,
    command: list[str],
    torchtitan_root: Path,
    baseline_losses: dict[int, float],
    autoparallel_losses: dict[int, float],
    output_dir: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report = {
        "autoparallel_codebase": _git_metadata(repo_root),
        "torchtitan_commit": _git_revision(torchtitan_root),
        "configuration": _configuration(len(baseline_losses)),
        "steps": len(baseline_losses),
        "baseline": {"gpus": 1, "dp": 1, "tp": 1, "losses": baseline_losses},
        "autoparallel": {
            "gpus": 4,
            "dp": 2,
            "tp": 2,
            "losses": autoparallel_losses,
        },
        "comparison": (
            "Behavioral curve comparison only: DP-dependent C4 sharding and packing "
            "differ, so no stepwise numerical-equivalence claim is made."
        ),
        "runs": 1,
        "variance": None,
        "result_source": "TensorBoard loss_metrics/global_avg_loss",
        "environment": _environment(),
        "command": command,
        "raw_logs": {
            "seed": str(output_dir / "loss_compare" / "seed_training.log"),
            "baseline": str(output_dir / "loss_compare" / "baseline_training.log"),
            "autoparallel": str(output_dir / "loss_compare" / "test_training.log"),
        },
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    args = parse_args()
    if args.steps <= 1:
        raise ValueError("--steps must be greater than one for a loss curve")
    work_dir, output_dir = _validate_environment(args)
    torchtitan_root = DEFAULT_TORCHTITAN_ROOT
    command = _loss_compare_command(torchtitan_root, work_dir, output_dir, args.steps)

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    python_path = [str(repo_root / "tests"), str(repo_root), str(torchtitan_root)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env["HF_HOME"] = str(work_dir / "hf_home")
    env["HF_DATASETS_CACHE"] = str(work_dir / "hf_datasets")
    env["TMPDIR"] = str(work_dir / "tmp")
    Path(env["TMPDIR"]).mkdir(parents=True)

    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=torchtitan_root, env=env, check=True)

    job_dir = work_dir / "job"
    baseline_losses = _read_losses(job_dir, "baseline")
    autoparallel_losses = _read_losses(job_dir, "test")
    expected_steps = set(baseline_losses)
    if len(expected_steps) != args.steps or set(autoparallel_losses) != expected_steps:
        raise RuntimeError(
            "Loss steps do not match: "
            f"baseline={sorted(baseline_losses)}, "
            f"autoparallel={sorted(autoparallel_losses)}"
        )
    if not all(
        math.isfinite(loss)
        for loss in (*baseline_losses.values(), *autoparallel_losses.values())
    ):
        raise RuntimeError("Found a non-finite loss value")

    _write_csv(
        output_dir / "qwen3_0_6b_loss_curve.csv", baseline_losses, autoparallel_losses
    )
    _write_plot(
        output_dir / "qwen3_0_6b_loss_curve.png", baseline_losses, autoparallel_losses
    )
    _write_report(
        output_dir / "report.json",
        command=command,
        torchtitan_root=torchtitan_root,
        baseline_losses=baseline_losses,
        autoparallel_losses=autoparallel_losses,
        output_dir=output_dir,
    )
    print(f"Artifacts written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
