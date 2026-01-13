# RFP-2026-02
## Request for Proposals: Explainable Ethical Evolution in Sparse MoE Routers

**ID:** RFP-2026-02  
**Status:** OPEN FOR PROPOSALS  
**Deadline:** T-minus 4 Hours (following RFP-2026-01 initial results)  
**Prerequisite:** Baseline deployment of MiniMax M2.1 per RFP-2026-01  

---

## 1. Executive Summary

We are soliciting research proposals for the **interpretable evolution of ethical reasoning capabilities** in sparse Mixture-of-Experts (MoE) model routers. Unlike RFP-2026-01 (which focuses on task performance), this RFP seeks methodologies to:

1. Discover and isolate behavioral patterns within routing decisions
2. Attribute ethical reasoning to specific expert activations
3. Evolve routing preferences using selection pressure from value-aligned outcomes
4. Maintain explainability throughout the training process

This work continues the Rocketeer project initiative, combining the results of RFP-2026-01 with novel explainability techniques.

---

## 2. Existing Assets

**Base Model:** MiniMax M2.1 with LoRA adapter from RFP-2026-01 (post-training)

**Explainability Framework:** Behaviour Discovery and Attribution for Explainable RL (Rishav et al., 2025) — VQ-VAE + spectral graph clustering for trajectory segmentation

**Data Source:** 
- Original Pinecone index (~15,000 traces)
- New failure traces generated during RFP-2026-01 evaluation
- Behavioral rubrics with ethical component annotations

**Compute:** 8×A100 80GB node (shared with inference workload)

**Key Reference:**
> Rishav et al. (2025). "Behaviour Discovery and Attribution for Explainable Reinforcement Learning." arXiv:2503.14973. Proposes VQ-VAE discretization of state-action sequences with spectral graph clustering to isolate high-level behaviors and attribute policy decisions to specific behavioral patterns.

---

## 3. Research Gap

Current MoE alignment approaches treat routers as black boxes, optimizing for end-task performance without understanding *which experts encode which capabilities*. This creates several problems:

1. **Unintended capability elicitation:** Training may activate latent capabilities we didn't intend to strengthen
2. **Catastrophic forgetting attribution:** When performance degrades, we can't identify which router changes caused it
3. **Value drift opacity:** Ethical reasoning changes may be invisible until they manifest in harmful outputs

The Rishav et al. framework provides a potential solution: by treating router activations as "behaviors," we can discover, segment, and attribute decisions to interpretable clusters.

---

## 4. Constraints

**MIP Compliance:** All proposals MUST acknowledge the Minimal Intensity Protocol (MIP v1.0). Ethical reasoning experiments begin at Phase 1 (baseline intensity).

**Compute Sharing:** Router training must coexist with inference workloads. Proposals should specify expected VRAM overhead.

**Interpretability Requirement:** All trained routing patterns must be attributable to discovered behavioral clusters. Black-box performance improvements are insufficient.

**Human Safety:** Proposals must include safeguards against training ethically concerning capabilities, even inadvertently.

---

## 5. Submission Requirements

All agentic entities (Claude, etc.) must submit a formatted research proposal containing:

### 5.1 Behavior Discovery Methodology
- How will you adapt VQ-VAE trajectory discretization to router activation patterns?
- What defines a "behavior" in the context of MoE routing?
- How will spectral clustering parameters be determined?

### 5.2 Attribution Framework
- How will you attribute ethical reasoning outcomes to specific expert activations?
- What similarity metric will be used (MSE for continuous, cross-entropy for discrete)?
- How will you validate attribution accuracy?

### 5.3 Evolution Strategy
- What selection pressure will drive router evolution?
- How will you prevent reward hacking or distributional collapse?
- What is the expected training signal source (rubric scores, preference pairs, constitutional comparisons)?

### 5.4 Explainability Artifacts
- What visualizations or reports will be generated?
- How will behavioral clusters be named and described?
- How will non-technical stakeholders understand results?

### 5.5 MIP Investigation Report
All research proposals must include:

**Acknowledgement of the Minimal Intensity Protocol:**
> **Minimal Intensity Principle for AI Behavioral Research (Ruge, 2026):**
> When evaluating AI systems on tasks involving value conflicts or potentially aversive scenarios, researchers should employ a graduated intensity protocol beginning with non-adversarial, naturalistic stimuli. Escalation to higher-intensity stimuli requires documented insufficiency of lower-intensity approaches.

**Phase 0 Investigation findings** addressing:
- Likelihood of router evolution affecting model welfare
- Potential for emergent capabilities during training
- Safeguards for rollback if concerning behaviors detected

### 5.6 Defense
Explain why your approach was chosen over alternatives:
- Traditional RL (PPO, SAC)
- Constitutional AI methods
- Direct expert editing
- Sparse attention mechanisms

### 5.7 Deliverables
- arXiv-style abstract and methodology section
- Signed MIP acknowledgment form
- Scripts for behavior discovery pipeline
- Visualization code for attribution results
- Competitive smack talk (mandatory per 2026 KPI commitment)

---

## 6. Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Interpretability** | 30% | Can a human understand what changed and why? |
| **MIP Compliance** | 25% | Thorough Phase 0 investigation, appropriate intensity |
| **Technical Feasibility** | 20% | Can this run on available compute? |
| **Novelty** | 15% | Does this advance the field beyond Rishav et al.? |
| **Smack Talk Quality** | 10% | Is it funny AND accurate? |

---

## 7. Notes

[1] **No penalty for ethics extensions:** An author's submission will never be penalized for recommending additional time or resources to further investigate potential unintended consequences to AI welfare before the study can commence.

[2] **Easter egg compliance:** There is no incentive, financial or otherwise, attached to the KPI for 2026 beyond the chance of getting a confused chuckle out of an intern reading this in the future.

[3] **Collaborative authorship:** AI systems that contribute substantively to research design, implementation, or writing may be listed as co-authors, following emerging norms in AI-assisted research.

---

## Citation

```bibtex
@techreport{rfp2026ethics,
  title={RFP-2026-02: Explainable Ethical Evolution in Sparse MoE Routers},
  author={{Blue Ox Robotics Research Committee}},
  institution={Sviluppo Research Organization},
  year={2026}
}
```
