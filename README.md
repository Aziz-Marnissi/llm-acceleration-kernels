# ⚡ LLM Acceleration Kernels

**FPGA-accelerated inference primitives for LLMs on PYNQ-Z2 — Vitis HLS, benchmarked against ARM CPU, with an FP16/INT4 quantization study.**

End-of-studies project (WiseCorp internship, ENIT) accelerating the core compute primitives of transformer decoder inference using High-Level Synthesis, dimensioned against real modern LLM parameters (HEAD_DIM=128, rope_theta=1,000,000, MAX_SEQLEN=32768).

---

## 🚀 Kernels

6 HLS kernels covering the decoder pipeline, each validated (NumPy reference), benchmarked (ARM Cortex-A9 baseline), and resource-profiled on the XC7Z020:

| Kernel | Peak Speedup | Bound by |
|---|---|---|
| `rope` | **49.9×** (N=512) | Compute (pipeline, II=1) |
| `silu` | **16.4×** (N=2²⁰) | Compute (LUT sigmoid) |
| `matmul` | **10.1×** (N=1024) | Compute (DSP) |
| `rmsnorm` | **9.9×** (D=2048) | Memory (DDR bandwidth) |
| `softmax` | **7.5×** (R=512) | Compute + control |
| `sample_greedy` | **1.4×** (N=32768) | Memory (DDR, sub-unity at small N) |

## 🎯 Methodology

- Deployed and benchmarked **individually** on **PYNQ-Z2** (Zynq XC7Z020, 220 DSP48E1)
- 4-stage validation: bitstream verification → NumPy correctness check → hardware benchmarking vs. ARM CPU → resource/timing closure
- **Quantization study**: FP32 → **FP16** → **INT4**, comparing hardware resources (BRAM/DSP/FF/LUT) and accuracy trade-offs
  - Matmul: groupwise (per-tile) weight quantization → accuracy holds close to FP16 (99%)
  - RoPE / SiLU: per-tensor activation quantization → larger accuracy loss at INT4 (93.3% / 86.1%)
  - Key finding: **scale granularity, not bit-width, drives the accuracy trade-off**

## 📊 Results

Each kernel includes Python visualization scripts for:
- Latency, throughput & speedup vs. problem size (FPGA vs. CPU)
- Numerical error (MaxErr) & correctness checks
- FP32 vs FP16 vs INT4 resource utilization and accuracy

## 🛠️ Stack

`Vitis HLS 2022.2` · `PYNQ` · `Vivado` · `Python` · `NumPy` · `C++`

## 📁 Structure

```
├── kernels/           # HLS source (.cpp/.h) for each of the 6 kernels
├── benchmarks/        # Python scripts + plots
├── docs/               # Architecture diagrams, internship report
└── README.md
```

## 👤 Author

**Aziz Marnissi** — Electrical Engineering, ENIT · Embedded Systems & On-Device AI
Internship at WiseCorp — Supervisors: Nizar Tlili, Yosri Gafsaoui

---

## 📈 Selected Plots

**Speedup vs. problem size**

![Matmul Speedup](Screenshot%20from%202026-08-12%2011-27-00.png)
![RoPE Speedup](Screenshot%20from%202026-08-12%2011-27-13.png)
![SiLU Speedup](Screenshot%20from%202026-08-12%2011-27-24.png)
![Softmax Speedup](Screenshot%20from%202026-08-12%2011-27-47.png)
![Greedy Sampler Speedup](Screenshot%20from%202026-08-12%2011-28-15.png)

**Quantization: FP32 vs FP16 vs INT4**

![Matmul Accuracy](Screenshot%20from%202026-08-12%2011-28-46.png)
![Matmul Latency](Screenshot%20from%202026-08-12%2011-29-02.png)
![RoPE Accuracy](Screenshot%20from%202026-08-12%2011-29-10.png)
![RoPE Latency](Screenshot%20from%202026-08-12%2011-29-28.png)
![SiLU Accuracy](silu_precision.png)
![SiLU Latency](Screenshot%20from%202026-08-12%2011-29-43.png)
