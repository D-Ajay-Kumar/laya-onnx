import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# One-shot Laya CPU/ONNX bootstrap
#
# First run:
#   1. Downloads the Laya checkpoint from Hugging Face (if missing).
#   2. Reconstructs the exact Laya DecisionModel.
#   3. Exports it with PyTorch's modern ONNX exporter (dynamo=True).
#   4. Verifies the ONNX graph.
#   5. Runs benchmark.py completely offline.
#
# Later runs:
#   - Reuses the downloaded checkpoint and laya.onnx.
#   - Skips both download and export.
#   - Runs benchmark.py offline.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
REPO_ID = "convaiinnovations/laya"
ONNX_PATH = ROOT / "laya.onnx"
BENCHMARK_PATH = ROOT / "benchmark.py"

# Files proven necessary by the working export/debugging process.
HF_FILES = [
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/*",
    "tokenizer/*",
]


def fail(message):
    print(f"\nERROR: {message}")
    raise SystemExit(1)


def required_checkpoint_files_exist():
    return all(
        [
            (ROOT / "rl_agent_config.json").is_file(),
            (ROOT / "model.safetensors").is_file(),
            (ROOT / "encoder" / "config.json").is_file(),
            (ROOT / "tokenizer" / "tokenizer.json").is_file(),
        ]
    )


def download_checkpoint():
    print("\n[1/4] Checking Hugging Face checkpoint...")

    if required_checkpoint_files_exist():
        print("      Checkpoint already exists. Skipping download.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        fail("huggingface_hub is not installed. Run: pip install -r requirements.txt")

    print(f"      Downloading {REPO_ID} ...")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(ROOT),
        allow_patterns=HF_FILES,
    )

    if not required_checkpoint_files_exist():
        fail("Hugging Face download completed but required checkpoint files are missing.")

    print("      Checkpoint ready.")


def export_onnx():
    print("\n[2/4] Checking ONNX model...")

    if ONNX_PATH.is_file():
        print("      laya.onnx already exists. Skipping export.")
        return

    print("      Reconstructing Laya DecisionModel...")

    import torch
    from safetensors.torch import load_file
    from transformers.initialization import no_init_weights
    from laya.common import build_model

    config_path = ROOT / "rl_agent_config.json"
    encoder_dir = ROOT / "encoder"
    weights_path = ROOT / "model.safetensors"

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # This is the architecture/loading path that was proven to work in the
    # troubleshooting report; do not replace it with a generic HF exporter.
    with no_init_weights():
        model = build_model(
            cfg,
            encoder_dir=str(encoder_dir),
            pretrained=False,
        )

    weights = load_file(str(weights_path))
    model.load_state_dict(weights, strict=True)
    model.eval()
    model.cpu()

    # A representative multi-question batch is important. The working export
    # used batch=3 so act_logits is exported with a dynamic batch dimension.
    batch = 3
    seq_len = 64
    num_markers = 3

    input_ids = torch.ones(batch, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

    marker_pos = torch.tensor(
        [
            [10, 20, 30],
            [10, 20, 30],
            [10, 20, 30],
        ],
        dtype=torch.long,
    )

    marker_mask = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ],
        dtype=torch.bool,
    )

    qtype = torch.tensor([0, 0, 0], dtype=torch.long)

    print("      Testing PyTorch forward...")
    with torch.no_grad():
        logits, act_logits = model(
            input_ids,
            attention_mask,
            marker_pos,
            marker_mask,
            qtype,
        )

    print(f"      logits:     {tuple(logits.shape)}")
    print(f"      act_logits: {tuple(act_logits.shape)}")

    print("      Exporting with modern ONNX exporter (dynamo=True)...")
    torch.onnx.export(
        model,
        (
            input_ids,
            attention_mask,
            marker_pos,
            marker_mask,
            qtype,
        ),
        str(ONNX_PATH),
        input_names=[
            "input_ids",
            "attention_mask",
            "marker_pos",
            "marker_mask",
            "qtype",
        ],
        output_names=[
            "logits",
            "act_logits",
        ],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "marker_pos": {0: "batch", 1: "markers"},
            "marker_mask": {0: "batch", 1: "markers"},
            "qtype": {0: "batch"},
        },
        opset_version=18,
        dynamo=True,
    )

    if not ONNX_PATH.is_file():
        fail("ONNX export returned without creating laya.onnx.")

    print("      Export complete.")


def verify_onnx():
    print("\n[3/4] Verifying ONNX graph...")

    import onnxruntime as ort

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    inputs = {x.name: x.shape for x in session.get_inputs()}
    outputs = {x.name: x.shape for x in session.get_outputs()}

    expected_inputs = [
        "input_ids",
        "attention_mask",
        "marker_pos",
        "marker_mask",
        "qtype",
    ]
    expected_outputs = ["logits", "act_logits"]

    if list(inputs) != expected_inputs:
        fail(f"Unexpected ONNX inputs: {list(inputs)}")

    if list(outputs) != expected_outputs:
        fail(f"Unexpected ONNX outputs: {list(outputs)}")

    print("      Inputs:")
    for name, shape in inputs.items():
        print(f"        {name}: {shape}")

    print("      Outputs:")
    for name, shape in outputs.items():
        print(f"        {name}: {shape}")

    # The critical regression check from the troubleshooting process:
    # act_logits must NOT be hard-coded to [1, 2].
    act_shape = outputs["act_logits"]
    if len(act_shape) != 2 or act_shape[1] != 2:
        fail(f"Unexpected act_logits shape: {act_shape}")

    if act_shape[0] != "batch":
        fail(
            "act_logits batch dimension is static. "
            f"Expected ['batch', 2], got {act_shape}"
        )

    print("      ONNX graph verified.")


def run_benchmark():
    print("\n[4/4] Running benchmark.py in offline mode...")

    if not BENCHMARK_PATH.is_file():
        fail(f"benchmark.py was not found at: {BENCHMARK_PATH}")

    env = os.environ.copy()

    # No Hugging Face/network access is allowed during the actual benchmark.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    result = subprocess.run(
        [sys.executable, str(BENCHMARK_PATH)],
        cwd=str(ROOT),
        env=env,
    )

    if result.returncode != 0:
        fail(f"benchmark.py exited with code {result.returncode}")

    print("\nBenchmark completed successfully.")


def main():
    if sys.version_info < (3, 10):
        fail("Python 3.10+ is required.")

    if not BENCHMARK_PATH.is_file():
        fail("Keep benchmark.py in the same directory as this script.")

    download_checkpoint()
    export_onnx()
    verify_onnx()
    run_benchmark()


if __name__ == "__main__":
    main()
