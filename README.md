# LLM_test

A collection of lightweight Python scripts designed to test and benchmark local Large Language Model (LLM) inference, with a specific focus on Qwen2.5.

## Overview

This repository provides tools for evaluating local inference performance:

- **Inference testing** – Validate model loading and basic response generation.
- **Latency benchmarking** – Measure time-to-first-token (TTFT) and overall generation speed.

## Project Structure

| File | Description |
|------|-------------|
| `test1.py` | Initial script for verifying model loading and inference. |
| `test_quen.py` | Dedicated script for Qwen2.5 integration testing. |
| `time_quen.py` | Benchmarking script for measuring streaming latency. |

## Quick Start

### 1. Prerequisites

Ensure you have:

- Python 3.10 or newer
- PyTorch installed and optimized for your hardware
  - NVIDIA GPU recommended for best performance

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/noop-rabbit/LLM_test.git
cd LLM_test
pip install -r requirements.txt
```

### 3. Usage

Run a basic inference test:

```bash
python test_quen.py
```

Run the latency benchmark:

```bash
python time_quen.py
```

## Purpose

This project is intended for:

- Verifying local LLM deployments
- Testing Qwen2.5 model integration
- Measuring inference latency and throughput
- Comparing hardware performance across systems

## Notes

- Results will vary depending on GPU, CPU, memory, and model size.
- For optimal performance, use a CUDA-enabled PyTorch installation.
- Streaming benchmarks are most meaningful when run on an otherwise idle system.

## License

This project is provided as-is for testing and benchmarking purposes.
