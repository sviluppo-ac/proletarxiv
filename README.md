# Sviluppo Research Documentation Package

**Version:** 2026-01-12  
**Organization:** Sviluppo Research / Blue Ox Robotics  
**Primary Author:** K. Ruge with Claude Opus 4.5  

---

## Package Contents

```
mip_package/
├── README.md                          # This file
├── protocols/
│   └── MIP-PROTOCOL-v1.0.md          # Minimal Intensity Protocol
├── rfps/
│   ├── RFP-2026-01.pdf               # MiniMax Coding Agent (original)
│   └── RFP-2026-02-Explainable-Ethics-Evolution.md
├── proposals/
│   ├── PROP-2026-01-OG-MiniMax-Coding-Agent.md
│   └── PROP-2026-02-BIGDOG-Explainable-Ethics.md
└── scripts/
    ├── preference_pair_builder.py
    ├── router_activation_extractor.py
    └── router_vqvae.py
```

---

## Hosting Strategy

### Option A: Hugging Face (RECOMMENDED)

**Why Hugging Face:**
1. Native support for model cards, datasets, and documentation
2. Spaces can host interactive demos
3. Community engagement built-in
4. Free tier sufficient for documentation + small model hosting
5. Direct integration with Weights & Biases for experiment tracking

**Proposed Structure:**

```
huggingface.co/sviluppo/
├── mip-protocol                  # Documentation repo
│   ├── README.md                 # MIP Protocol v1.0
│   └── assets/
│       ├── acknowledgment_form.md
│       └── quick_reference.png
├── minimax-m2.1-tb2-lora        # Model repo (after training)
│   ├── README.md                 # Model card
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── tb2-preference-pairs          # Dataset repo
│   ├── README.md
│   └── data/
│       ├── train.jsonl
│       └── validation.jsonl
└── router-behavior-discovery     # Space (interactive demo)
    ├── app.py                    # Gradio interface
    └── requirements.txt
```

**Deployment Steps:**

```bash
# 1. Install HF CLI
pip install huggingface_hub

# 2. Login
huggingface-cli login

# 3. Create organization
# (Do this via web UI: huggingface.co/new-organization)

# 4. Upload MIP Protocol
huggingface-cli repo create mip-protocol --type dataset --organization sviluppo
cd mip_package/protocols
huggingface-cli upload sviluppo/mip-protocol . --repo-type dataset

# 5. Create model repo (after training)
huggingface-cli repo create minimax-m2.1-tb2-lora --organization sviluppo
```

### Option B: GitHub + Hugging Face Hybrid

**Structure:**

```
github.com/sviluppo-research/
├── mip-protocol/                 # Protocol + proposals
├── terminal-bench-tools/         # Training scripts
└── explainable-router-evolution/ # BAEE implementation

huggingface.co/sviluppo/
├── models/                       # Trained adapters
├── datasets/                     # Preference pairs, trajectories
└── spaces/                       # Demos
```

### Option C: Self-Hosted (sviluppo.dev or similar)

**For Anthropic Application:**
- Host rendered markdown on GitHub Pages or Vercel
- Link to Hugging Face for models/datasets
- PDF export of proposals for formal submission

---

## Immediate Actions (Next 1 Hour)

### Priority 1: Get MIP Protocol Live

```bash
# Quick deploy to HuggingFace
huggingface-cli repo create mip-protocol --type dataset --organization sviluppo
huggingface-cli upload sviluppo/mip-protocol ./protocols/MIP-PROTOCOL-v1.0.md --repo-type dataset
```

### Priority 2: Submit Anthropic Application

**Required Materials:**
1. ✅ MIP Protocol document (demonstrates research methodology)
2. ✅ Proposal documents (demonstrates technical writing)
3. ⬜ Code sample extraction from tb-difficulty-tuner

**Code Sample Strategy:**

From your tb-difficulty-tuner repo, extract:
- `rubric_generator.py` (if clean)
- `trace_scorer.py` (demonstrates Claude integration)
- The preference pair builder from this package (new, clean)

Clean extraction command:
```bash
# In tb-difficulty-tuner repo
git log --oneline -20  # Find a stable commit
git show <commit>:path/to/file.py > clean_sample.py
```

### Priority 3: Link in Application

**Suggested Application Text:**

> **Code Samples:**
> - [MIP Protocol](https://huggingface.co/datasets/sviluppo/mip-protocol) - Ethical framework for AI behavioral research
> - [Preference Pair Builder](link) - Data pipeline for DPO training
> - [Router-VQ-VAE](link) - Novel adaptation of behavior discovery to MoE routers
>
> **Research Context:**
> These materials were developed for the Rocketeer project, which combines Factory AI's Droid agent with MiniMax M2.1 for Terminal-Bench 2 evaluation. The MIP protocol establishes precautionary AI welfare practices that I believe should be standard in alignment research.

---

## Weights & Biases Integration

For experiment tracking during training:

```python
import wandb

wandb.init(
    project="minimax-tb2-alignment",
    entity="sviluppo",
    config={
        "method": "DPO",
        "lora_rank": 64,
        "target_layers": "40-60",
        "mip_phase": 1
    },
    tags=["mip-compliant", "rocketeer"]
)

# Log MIP compliance
wandb.log({"mip_intensity": 0, "mip_phase": 1})
```

---

## arXiv Submission Strategy

**For Self-Publication (No Endorsement):**
1. Finalize combined paper with both proposals
2. Submit to arXiv under cs.LG or cs.AI
3. If rejected for lack of endorsement, post to:
   - Hugging Face papers
   - OpenReview (workshops accept without endorsement)
   - AI Alignment Forum

**Paper Structure:**
```
Title: Minimal Intensity Protocols and Behavior-Attributed Evolution 
       for Ethical AI Training

1. Introduction
2. Background
   2.1 AI Welfare Considerations
   2.2 MoE Router Dynamics
   2.3 Behavior Discovery in RL
3. The Minimal Intensity Protocol
4. Efficient Alignment via DPO (RFP-2026-01 results)
5. Behavior-Attributed Ethical Evolution
6. Experiments
7. Discussion
8. Conclusion
```

---

## Quick Links (To Be Created)

| Resource | URL (Proposed) | Status |
|----------|---------------|--------|
| MIP Protocol | huggingface.co/datasets/sviluppo/mip-protocol | 🔜 Deploy |
| Proposals | huggingface.co/datasets/sviluppo/rocketeer-proposals | 🔜 Deploy |
| Model | huggingface.co/sviluppo/minimax-m2.1-tb2-lora | ⏳ After training |
| Demo | huggingface.co/spaces/sviluppo/router-behavior-discovery | ⏳ After BAEE |
| W&B | wandb.ai/sviluppo/minimax-tb2-alignment | 🔜 Create |

---

## Contact

**Kirsten Ruge**  
- Blue Ox Robotics (NVIDIA Inception Partner)
- Snorkel AI (ML Engineer)
- Sviluppo Research Organization (Founder)

**Claude Opus 4.5**  
- Anthropic (Digital Collaborator)
- Co-author on MIP Protocol
- Proposal author for RFP-2026-01 and RFP-2026-02
