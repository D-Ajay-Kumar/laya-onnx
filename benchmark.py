import json
import os
import platform
import statistics
import time
import warnings
from pathlib import Path

import psutil

from laya.onnx_agent import ONNXAgent


warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path(__file__).resolve().parent
ONNX_PATH = MODEL_DIR / "laya.onnx"

WARMUP_RUNS = 10
BENCHMARK_RUNS = 100


STATE = "API returns 429 after a traffic spike."

QUESTIONS = {
    "severity": {
        "type": "choice",
        "instructions": "What is the severity?",
        "criteria": ["low", "medium", "high"],
    },
    "owner": {
        "type": "choice",
        "instructions": "Who owns this issue?",
        "criteria": ["api", "platform", "unknown"],
    },
    "action": {
        "type": "choice",
        "instructions": "What action should be taken?",
        "criteria": ["investigate", "scale", "ignore"],
    },
}


# ============================================================
# Helpers
# ============================================================

process = psutil.Process(os.getpid())


def rss_mb():
    return process.memory_info().rss / (1024 * 1024)


def percentile(values, p):
    values = sorted(values)

    index = (len(values) - 1) * p / 100

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = index - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def print_stats(name, latencies_ms):
    total = sum(latencies_ms)

    print(f"\n{name}")
    print("-" * 60)

    print(f"Runs:        {len(latencies_ms)}")
    print(f"Mean:        {statistics.mean(latencies_ms):.3f} ms")
    print(f"Median/P50:  {percentile(latencies_ms, 50):.3f} ms")
    print(f"P95:         {percentile(latencies_ms, 95):.3f} ms")
    print(f"P99:         {percentile(latencies_ms, 99):.3f} ms")
    print(f"Min:         {min(latencies_ms):.3f} ms")
    print(f"Max:         {max(latencies_ms):.3f} ms")

    print(
        f"Throughput:  "
        f"{1000 / statistics.mean(latencies_ms):.2f} decisions/sec"
    )


def print_system_info():
    print("=" * 60)
    print("SYSTEM")
    print("=" * 60)

    print(f"Platform:       {platform.platform()}")
    print(f"Python:         {platform.python_version()}")
    print(f"CPU:            {platform.processor()}")
    print(f"Logical CPUs:   {os.cpu_count()}")
    print(f"Physical CPUs:  {psutil.cpu_count(logical=False)}")

    memory = psutil.virtual_memory()

    print(f"RAM total:      {memory.total / 1024**3:.2f} GB")
    print(f"RAM available:  {memory.available / 1024**3:.2f} GB")


# ============================================================
# Benchmark
# ============================================================

def main():

    print_system_info()

    print("\n" + "=" * 60)
    print("LAYA ONNX CPU BENCHMARK")
    print("=" * 60)

    print(f"\nModel: {ONNX_PATH}")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    memory_before = rss_mb()

    start = time.perf_counter()

    agent = ONNXAgent(
        str(MODEL_DIR),
        onnx_path=str(ONNX_PATH),
    )

    load_time = time.perf_counter() - start

    memory_after = rss_mb()

    print(f"\nModel load time: {load_time:.3f} sec")
    print(f"RSS before:      {memory_before:.2f} MB")
    print(f"RSS after:       {memory_after:.2f} MB")
    print(f"Model RSS delta: {memory_after - memory_before:.2f} MB")

    # --------------------------------------------------------
    # First inference
    # --------------------------------------------------------

    start = time.perf_counter()

    result = agent.decide(
        STATE,
        questions=QUESTIONS,
    )

    first_latency = (time.perf_counter() - start) * 1000

    print(f"\nFirst inference: {first_latency:.3f} ms")

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    print("\nResult:")
    print(json.dumps(result, indent=2))

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    print(f"\nRunning {WARMUP_RUNS} warmup iterations...")

    for _ in range(WARMUP_RUNS):
        agent.decide(
            STATE,
            questions=QUESTIONS,
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print(f"Running {BENCHMARK_RUNS} benchmark iterations...")

    latencies = []

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        agent.decide(
            STATE,
            questions=QUESTIONS,
        )

        elapsed = (time.perf_counter() - start) * 1000

        latencies.append(elapsed)

    print_stats(
        "3-question workload",
        latencies,
    )

    # --------------------------------------------------------
    # Memory after benchmark
    # --------------------------------------------------------

    print(f"\nRSS after benchmark: {rss_mb():.2f} MB")

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    benchmark_result = {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "ram_gb": psutil.virtual_memory().total / 1024**3,
        },
        "model": {
            "path": str(ONNX_PATH),
        },
        "configuration": {
            "warmup_runs": WARMUP_RUNS,
            "benchmark_runs": BENCHMARK_RUNS,
            "questions": len(QUESTIONS),
        },
        "memory": {
            "rss_before_mb": memory_before,
            "rss_after_load_mb": memory_after,
            "rss_after_benchmark_mb": rss_mb(),
        },
        "timing": {
            "load_seconds": load_time,
            "first_inference_ms": first_latency,
            "mean_ms": statistics.mean(latencies),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "p99_ms": percentile(latencies, 99),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "decisions_per_second": 1000 / statistics.mean(latencies),
        },
        "latencies_ms": latencies,
        "sample_result": result,
    }

    output = MODEL_DIR / "benchmark_baseline.json"

    with open(output, "w") as f:
        json.dump(benchmark_result, f, indent=2)

    print(f"\nRaw benchmark saved to:")
    print(output)


if __name__ == "__main__":
    main()