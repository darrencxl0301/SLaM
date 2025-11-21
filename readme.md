# EdgeLLM: Full-Stack Small Language Model Framework

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Models: 13+](https://img.shields.io/badge/Models-13%2B%20SLMs-purple)](#supported-models)
[![Hardware: Edge](https://img.shields.io/badge/Hardware-Edge%20Deployment-green)](#hardware-requirements)

*Enterprise-grade Small Language Models on consumer hardware*

**Three Powerful Components**: RAG + LoRA Training + Structured Querying

[📖 Documentation](#documentation) | [🚀 Quick Start](#quick-start) | [🎬 Demos](#demos) | [🤝 Collaboration](#collaboration-opportunities)

</div>

---

## 🎯 What is EdgeLLM?

**EdgeLLM** is a complete, production-ready framework for deploying **Small Language Models (0.5B-14B parameters)** on edge devices and consumer hardware (≥6GB VRAM).

### Three Integrated Components:

#### 📚 Component 1: Conversational RAG System
**Train + Deploy domain-specific chatbots**
- ✅ 13 SLM families (Qwen, Llama, DeepSeek, Gemma, Mistral, SmolLM)
- ✅ QLoRA 4-bit fine-tuning pipeline
- ✅ FAISS vector retrieval for accurate context
- ✅ Live feedback system with instant knowledge updates

#### 🔧 Component 2: LoRA Training Pipeline
**Efficient fine-tuning on consumer GPUs**
- ✅ 4-bit quantized training (6GB VRAM minimum)
- ✅ 13 pre-configured training scripts for different models
- ✅ Custom dataset preparation utilities
- ✅ Hyperparameter templates and best practices

#### 🔍 Component 3: Schema-Action Query System
**SQL-like queries without databases using Small LMs**
- ✅ Natural language → Structured data queries
- ✅ Multi-table auto-JOIN with intelligent planning
- ✅ 3B model (vs 20B+ traditional Text-to-SQL)
- ✅ Direct CSV/Excel querying (zero database setup)

---

## 🏗️ Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                    EdgeLLM Framework                         │
│          Full-Stack Small Language Model Suite               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │ Component 1: │  │ Component 2: │  │  Component 3:    │    │
│  │  RAG System  │  │  LoRA Train  │  │ Query Pipeline   │    │
│  ├──────────────┤  ├──────────────┤  ├──────────────────┤    │
│  │              │  │              │  │                  │    │
│  │ • Retrieval  │  │ • 13 Models  │  │ • NL → Query     │    │
│  │ • Inference  │  │ • 4-bit LoRA │  │ • Auto-JOIN      │    │
│  │ • Feedback   │  │ • Custom Data│  │ • CSV/Excel      │    │
│  │ • Live Update│  │ • Templates  │  │ • 3B SLM         │    │
│  │              │  │              │  │                  │    │
│  └──────────────┘  └──────────────┘  └──────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │         Unified Deployment Interface (Streamlit)         ││
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
| **Privacy** | Cloud-based | 100% local |
| **Hardware** | API only | Consumer GPU (6GB+) |
| **Customization** | Limited | Full fine-tuning control |
| **Scalability** | Pay per use | Unlimited |
| **Data Control** | Vendor-dependent | Complete ownership |

### Our Value Proposition

**Not just smaller models** — A complete development stack:
1. **Train** your own SLM with efficient LoRA (Component 2)
2. **Enhance** with knowledge retrieval (Component 1: RAG)
3. **Query** structured data without SQL (Component 3: Schema-Action)
4. **Deploy** with production-ready UI (Streamlit)

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/darrencxl0301/EdgeLLM.git
cd EdgeLLM
pip install -r requirements.txt
```

---

### Option 1: Unified Web Interface (Recommended)

**Launch both components in one interface:**
```bash
streamlit run deployment/streamlit_app.py
```

Access at `http://localhost:8501`

**Features:**
- ✅ Component 1: RAG chatbot with live feedback
- ✅ Component 3: Natural language data queries
- ✅ Real-time knowledge base updates
- ✅ User feedback collection system

---

### Option 2: Component 1 - Conversational RAG (CLI)
```bash
cd conversational_rag

# Run LV customer service demo (Chinese)
python inference_rag.py \
    --rag_dataset examples/lv_customer_service/data/knowledge.jsonl \
    --lora_path examples/lv_customer_service/models/lv_lora \
    --mode interactive
```

---

### Option 3: Component 3 - Schema-Action Queries (CLI)
```bash
cd schema_action

# Run retail analytics demo
python schema_action.py \
    --data_directory examples/retail_analytics/data/ \
    --metadata_path examples/retail_analytics/metadata.json \
    --join_config_path examples/retail_analytics/join_config.json
```

**Sample queries:**
- "Show me top 5 customers by total revenue"
- "How many orders were placed in 2017?"
- "What is the email address for customer John Smith?"

---

## 📊 Supported Small Language Models

| Model Family | Parameters | Training Script | Inference Speed |
|-------------|-----------|-----------------|-----------------|
| **Qwen** | 0.5B-14B | `train_qwen_lora*.py` | Fast |
| **DeepSeek** | 1.5B-14B | `train_deepseek_lora*.py` | Fast |
| **Llama** | 1B-8B | `train_llama_lora*.py` | Fast |
| **Gemma** | 4B | `train_gemma_lora.py` | Medium |
| **Mistral** | 7B | `train_mistral_lora.py` | Medium |
| **SmolLM** | 1.7B | `train_smollm_lora.py` | Very Fast |


**All models support:**
- ✅ 4-bit QLoRA training
- ✅ FAISS RAG integration
- ✅ Multi-language deployment
- ✅ Edge device optimization

---

## 🎯 Component 3: Why Schema-Action Matters

### Current Text-to-SQL Limitations:

| Challenge | Traditional Systems | EdgeLLM Solution |
|-----------|-------------------|------------------|
| **Model Size** | 20B+ params (GPT-4, Claude) | 3B params (Llama-3.2-3B) |
| **Infrastructure** | PostgreSQL/MySQL setup | Direct CSV/Excel access |
| **Multi-Table** | Manual JOIN config | Auto-JOIN via metadata |
| **Cost** | $0.50+ per 1K queries | $0 (self-hosted) |
| **Data Privacy** | Cloud-based | 100% local |

### Our Innovation:

✅ **10x smaller model** achieves comparable accuracy  
✅ **Zero database setup** - point to files and query  
✅ **Intelligent JOIN planning** - automatic table relationships  
✅ **Domain-agnostic** - works across industries without retraining

---

## 🔧 Hardware Requirements

### Component 1: Conversational RAG

| Task | Min VRAM | Recommended | Speed |
|------|----------|-------------|-------|
| **LoRA Training (4-bit)** | 6GB | 12GB | ~500 samples/hr |
| **Inference** | 5GB | 6GB | ~2-5 tokens/sec |
| **RAG Indexing** | 2GB RAM | 4GB RAM | ~500 pairs/sec |

### Component 3: Schema-Action Query

| Task | Min VRAM | Recommended | Notes |
|------|----------|-------------|-------|
| **Query Planning (3B)** | 6GB | 8GB | Llama-3.2-3B |
| **Data Processing** | 4GB RAM | 8GB RAM | CPU only |

**Tested Hardware:**
- ✅ RTX 3090 (24GB)    - All components
- ✅ RTX 2000 ada (8GB) - Inference only

---

## 🎨 Industry Use Cases

| Industry | Component 1 (RAG) | Component 3 (Query) |
|----------|-------------------|---------------------|
| 🛍️ **E-commerce** | Product Q&A chatbot | Sales analytics |
| 🏢 **HR** | Policy assistant | Employee data lookups |
| ⚖️ **Legal** | Compliance Q&A | Case database search |
| 🏥 **Healthcare** | Patient info assistant* | Medical records queries* |
| 🏦 **Finance** | Banking FAQ bot | Transaction analysis |
| 🏨 **Hospitality** | Concierge chatbot | Booking analytics |
| 🎓 **Education** | Course assistant | Student performance queries |
| 🏭 **Manufacturing** | SOP assistant | Inventory analytics |

*Requires regulatory compliance review

---

## 🌟 Key Differentiators

### vs. Cloud LLM APIs (GPT-4, Claude):
- ✅ **100x cheaper** - No per-token costs
- ✅ **Complete privacy** - Data never leaves your infrastructure
- ✅ **Unlimited scaling** - No rate limits
- ✅ **Full customization** - Fine-tune on your exact domain

### vs. Traditional Text-to-SQL:
- ✅ **10x smaller model** - 3B vs 20B+ parameters
- ✅ **Zero database setup** - Direct file access
- ✅ **Auto-JOIN** - Intelligent table relationships
- ✅ **Production-ready** - Includes UI and feedback system

### vs. Other RAG Frameworks:
- ✅ **Multi-model support** - 13 SLM families
- ✅ **Live feedback loop** - Instant knowledge updates
- ✅ **Complete stack** - Training + RAG + Deployment
- ✅ **Edge-optimized** - Consumer GPU deployment

---

## 🤝 Collaboration Opportunities

### 💼 For Enterprises & Researchers

We welcome collaboration with organizations interested in:
- 🎯 Custom domain-specific SLM deployment
- 🔬 Research partnerships on edge AI
- 📊 Industry-specific implementations
- 🎓 Educational use cases

### 💡 Symbolic Technical Consultation Fee

**If you have any concerns or need technical guidance**, we offer collaboration with a **symbolic technical consultation fee**.

**Contact for collaboration inquiries:**

**Dr. Lim Tong Ming**  
Director, Centre for Business, Economics and Intelligent Ventures (CBEIV)  
Tunku Abdul Rahman University of Management and Technology (TARUMT)

📧 Email: limtm@tarc.edu.my  
📱 Phone: +60 18-776 2865

**We provide:**
- ✅ Technical consultation and guidance
- ✅ Custom model training support
- ✅ Domain-specific implementation assistance
- ✅ Research collaboration opportunities
- ✅ Training workshops and seminars

---

## 🚀 Getting Started Commands

### Quick Demo (All-in-One Interface)
```bash
streamlit run deployment/streamlit_app.py
```

### Component 1: Train Your Own Model
```bash
cd conversational_rag/lora_trainer
python train_qwen_lora.py \
    --data_path your_training_data.jsonl \
    --output_dir ./models/your_domain_lora
```

### Component 1: Run RAG Inference
```bash
cd conversational_rag
python inference_rag.py \
    --rag_dataset your_knowledge.jsonl \
    --lora_path ./models/your_domain_lora \
    --mode interactive
```

### Component 3: Query Your Data
```bash
cd schema_action
python schema_action.py \
    --data_directory ./your_data/ \
    --metadata_path ./your_metadata.json \
    --join_config_path ./your_join_config.json
```

---

## 🤝 Contributing

Contributions welcome! 

**Priority areas:**
- 🌐 More language support (currently EN/CN)
- 📊 Additional data connectors (PostgreSQL, MongoDB)
- 🎨 UI/UX improvements
- 📖 Documentation translations
- 🧪 Test coverage
- 🔧 New model integrations

---

## 📝 Citation
```bibtex
@software{edgellm2025,
  title={EdgeLLM: Full-Stack Small Language Model Framework},
  author={Darren & Lim},
  year={2025},
  institution={Tunku Abdul Rahman University of Management and Technology (TARUMT)},
  organization={Centre for Business, Economics and Intelligent Ventures (CBEIV)},
  url={https://github.com/darrencxl0301/EdgeLLM}
}
```

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE)

This project is open-source and free to use for:
- ✅ Academic research
- ✅ Commercial applications
- ✅ Educational purposes
- ✅ Personal projects

---

## 📮 Contact & Support

### Project Members
**Darren Chai Xin Lun**  
Data Science Honours Student  
Tunku Abdul Rahman University of Management and Technology (TARUMT)

### Academic Supervisor & Collaboration Inquiries
**Dr. Lim Tong Ming**  
Director, Centre for Business, Economics and Intelligent Ventures (CBEIV)  
Tunku Abdul Rahman University of Management and Technology (TARUMT)

📧 Email: limtm@tarc.edu.my  
📱 Phone: +60 18-776 2865  
🏢 Office: CBEIV, TARUMT Kuala Lumpur Campus

### Institution
**Centre for Business, Economics and Intelligent Ventures (CBEIV)**  
Tunku Abdul Rahman University of Management and Technology  
Jalan Genting Kelang, Setapak  
53300 Kuala Lumpur, Malaysia

🌐 Website: [https://cbiev.tarc.edu.my](https://cbiev.tarc.edu.my)

---

## 🙏 Acknowledgments

### Technology Stack
- Built with [Transformers](https://huggingface.co/transformers/) by Hugging Face
- Powered by [Qwen](https://github.com/QwenLM/Qwen), [Llama](https://llama.meta.com/), [DeepSeek](https://github.com/deepseek-ai), and more
- UI with [Streamlit](https://streamlit.io/)
- Vector search with [FAISS](https://github.com/facebookresearch/faiss)
- Training with [PEFT](https://github.com/huggingface/peft) and [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

### Special Thanks
- **TARUMT Faculty of Computing and Information Technology** for research support and infrastructure
- **Centre for Business, Economics and Intelligent Ventures (CBEIV)** for guidance and resources
- **Open-source AI community** for foundational models, tools, and collaboration
- **All contributors** who have helped improve this framework

---

## 🎯 Roadmap

### Current Version (v0.1.0)
- ✅ 13 SLM training scripts
- ✅ RAG system with FAISS
- ✅ Schema-Action query pipeline
- ✅ Streamlit deployment interface
- ✅ Live feedback system

### Upcoming (v0.2.0)
- 🔄 PostgreSQL/MongoDB connectors
- 🔄 Multi-language UI (EN/CN/MS)
- 🔄 Automated hyperparameter tuning
- 🔄 Model compression techniques
- 🔄 REST API interface

### Future (v1.0.0)
- 📋 Docker containerization
- 📋 Kubernetes deployment
- 📋 Advanced monitoring dashboard
- 📋 Multi-tenant support

---

<div align="center">

⭐ **Star this repo if EdgeLLM helps your business or research!**

[Report Bug](https://github.com/darrencxl0301/EdgeLLM/issues) · [Request Feature](https://github.com/darrencxl0301/EdgeLLM/issues) · [Discussions](https://github.com/darrencxl0301/EdgeLLM/discussions)

**Made with ❤️ for democratizing AI**

*Developed at Tunku Abdul Rahman University of Management and Technology (TARUMT)*  
*Centre for Business, Economics and Intelligent Ventures (CBEIV)*

---

**For technical consultation and collaboration:**  
📧 limtm@tarc.edu.my | 📱 +60 18-776 2865

</div>
