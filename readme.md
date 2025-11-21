# EdgeLLM: Full-Stack Small Language Model Framework

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Models: 13+](https://img.shields.io/badge/Models-13%2B%20SLMs-purple)](#supported-models)
[![Hardware: Edge](https://img.shields.io/badge/Hardware-Edge%20Deployment-green)](#hardware-requirements)

*Enterprise-grade Small Language Models on consumer hardware*

**Three Powerful Modules**: RAG + LoRA + Structured Querying

</div>

---

## 🎯 What is EdgeLLM?

**EdgeLLM** is a complete, production-ready framework for deploying **Small Language Models (0.5B-14B)** on edge devices and consumer hardware (≥6GB VRAM).

### Three Integrated Modules:

#### 📚 Module 1: Conversational RAG
**Train + Deploy domain-specific chatbots**
- ✅ 13 SLM families (Qwen, Llama, DeepSeek, Gemma, Mistral, SmolLM)
- ✅ QLoRA 4-bit fine-tuning
- ✅ FAISS vector retrieval
- ✅ Live feedback system

#### 🔧 Module 2: LoRA Training Pipeline
**Efficient fine-tuning on consumer GPUs**
- ✅ 4-bit quantized training (6GB VRAM minimum)
- ✅ Multi-model support (13 training scripts)
- ✅ Custom dataset preparation
- ✅ Hyperparameter templates

#### 🔍 Module 3: Schema-Action Query System
**SQL-like queries without databases using SLMs**
- ✅ Natural language → Structured queries
- ✅ Multi-table auto-JOIN
- ✅ 3B model (vs 20B+ competitors)
- ✅ Direct CSV/Excel querying

---

## 🏗️ Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                    EdgeLLM Framework                          │
│          Full-Stack Small Language Model Suite                │
├──────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Module 1:   │  │  Module 2:   │  │   Module 3:      │   │
│  │  RAG System  │  │  LoRA Train  │  │  Query Pipeline  │   │
│  ├──────────────┤  ├──────────────┤  ├──────────────────┤   │
│  │              │  │              │  │                  │   │
│  │ • Retrieval  │  │ • 13 Models  │  │ • NL2Query      │   │
│  │ • Inference  │  │ • 4-bit QLoRA│  │ • Auto-JOIN     │   │
│  │ • Feedback   │  │ • Custom Data│  │ • CSV/Excel     │   │
│  │ • Live Update│  │ • Templates  │  │ • 3B SLM        │   │
│  │              │  │              │  │                  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐│
│  │         Unified Deployment Interface (Streamlit)          ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

---

## 🌟 Why EdgeLLM?

### The Small Language Model Revolution

**Large LMs (GPT-4, Claude) vs Small LMs (0.5B-14B)**

| Aspect | Large LMs | EdgeLLM (Small LMs) |
|--------|-----------|---------------------|
| **Cost** | $10-100 per 1M tokens | $0 (self-hosted) |
| **Latency** | 500-2000ms | 50-200ms |
| **Privacy** | Cloud-based | 100% local |
| **Hardware** | API only | Consumer GPU |
| **Customization** | Limited | Full fine-tuning |
| **Scalability** | Pay per use | Unlimited |

### Our Value Proposition

**Not just smaller models** — A complete development stack:
1. **Train** your own SLM (Module 2: LoRA)
2. **Enhance** with knowledge retrieval (Module 1: RAG)
3. **Query** structured data (Module 3: Schema-Action)
4. **Deploy** with production UI (Streamlit)

---

## 📊 Supported Small Language Models

| Model Family | Parameters | Script | Training VRAM | Use Case |
|-------------|-----------|--------|---------------|----------|
| **Qwen** | 0.5B-14B | `train_qwen_lora*.py` | 6GB-16GB | General purpose |
| **DeepSeek** | 1.5B-14B | `train_deepseek_lora*.py` | 6GB-20GB | Reasoning tasks |
| **Llama** | 1B-8B | `train_llama_lora*.py` | 6GB-16GB | Instruction following |
| **Gemma** | 4B | `train_gemma_lora.py` | 8GB | Balanced performance |
| **Mistral** | 7B | `train_mistral_lora.py` | 12GB | Advanced reasoning |
| **SmolLM** | 1.7B | `train_smollm_lora.py` | 6GB | Ultra-efficient |

**Key Features:**
- ✅ All support 4-bit QLoRA training
- ✅ FAISS RAG integration ready
- ✅ Multi-language capable
- ✅ Edge deployment optimized
