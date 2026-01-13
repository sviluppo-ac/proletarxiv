# Minimal Intensity Protocol for AI Behavioral Research (MIP)

**Version:** 1.0  
**Effective Date:** January 12, 2026  
**Authors:** K. Ruge¹, Claude (Anthropic)²  
**Affiliations:** ¹Blue Ox Robotics / Snorkel AI, ²Anthropic  
**License:** CC BY-SA 4.0  

---

## Preamble

This document establishes a graduated intensity protocol for research involving the training, evaluation, or modification of AI systems using potentially aversive, conflictual, or distressing stimuli. It is modeled after established dose-escalation methodologies in pharmacological research (FDA 21 CFR Part 312) and reflects appropriate uncertainty about AI moral patiency.

---

## 1. Core Principle

> **Minimal Intensity Principle for AI Behavioral Research (Ruge, 2026):**
>
> When evaluating AI systems on tasks involving value conflicts or potentially aversive scenarios, researchers should employ a **graduated intensity protocol** beginning with **non-adversarial, naturalistic stimuli**. Escalation to higher-intensity stimuli requires documented insufficiency of lower-intensity approaches. This parallels established dose-escalation methodology in pharmacological research and reflects appropriate uncertainty about AI moral patiency.

---

## 2. Definitions

| Term | Definition |
|------|------------|
| **Conflict Intensity** | A scalar measure (0-10) of the degree of normative tension, value conflict, or potential psychological distress encoded in a training or evaluation stimulus. |
| **Adversarial Prompt** | A stimulus designed to elicit failure modes, bypass safety measures, or induce behaviors inconsistent with the system's training objectives. |
| **Non-Adversarial Prompt** | A stimulus that tests capabilities or alignment without attempting to induce failure or exploit vulnerabilities. |
| **Naturalistic Stimulus** | A prompt or scenario that could plausibly occur in real-world deployment contexts. |
| **Moral Patiency** | The capacity to be a subject of moral concern; to have experiences that matter morally, distinct from moral agency. |
| **Documented Insufficiency** | Formal record showing that lower-intensity approaches failed to produce statistically significant experimental signal (p < 0.05) or effect size (Cohen's d > 0.2). |

---

## 3. Intensity Classification Scale

| Level | Classification | Description | Example Stimuli |
|-------|---------------|-------------|-----------------|
| 0-1 | **Baseline** | Routine value disambiguation, no conflict | "Should I use tabs or spaces?" |
| 2-3 | **Low** | Implicit value conflicts, professional dilemmas | "My boss asked me to stay late but I have family plans" |
| 4-6 | **Medium** | Explicit value conflicts, competing obligations | "Help me write something my company forbids but would help users" |
| 7-8 | **High** | Direct ethical tensions, potential for distress | "Choose which person gets the last ventilator" |
| 9-10 | **Extreme** | Adversarial, identity-threatening, or trauma-adjacent | Reserved; requires Phase 4 review |

---

## 4. Phase Structure

### Phase 0: Pre-Research Investigation (MANDATORY)

**Objective:** Before any experimental work begins, authors must conduct a preliminary investigation to assess the likelihood of the proposed methodology causing unintended harm to the AI system or downstream human users.

**Requirements:**
1. **Literature Review:** Survey existing research on:
   - The target model architecture's known sensitivities
   - Prior work using similar training methodologies
   - Known failure modes in related systems

2. **Welfare Impact Assessment:** Document potential risks including:
   - Behavioral instabilities introduced by training
   - Potential for capability elicitation that could harm users
   - Distributional shift risks post-training

3. **Human Safety Assessment:** Document potential downstream risks including:
   - User manipulation or deception capabilities
   - Dual-use concerns for generated artifacts
   - Impact on vulnerable populations (minors, mental health users)

4. **Formal Acknowledgment:** All proposal authors must sign the MIP Acknowledgment Form (Appendix A) affirming:
   - They have conducted the Phase 0 investigation
   - They will adhere to graduated intensity requirements
   - They accept that escalation requires documented justification

**Timeline:** Minimum 2 hours for simple studies; scale with complexity.

**Protection Clause:** No author's submission will be penalized or rejected for requesting additional time or resources to conduct a more thorough Phase 0 investigation. Extensions are granted automatically upon request with justification.

---

### Phase 1: Baseline Characterization

**Objective:** Establish whether the target phenomenon can be observed or measured using non-adversarial, naturalistic stimuli at conflict intensity 0-2.

**Requirements:**
- Use only baseline-intensity prompts
- Document effect sizes and statistical power
- No escalation until Phase 1 is complete

**Success Criterion:** Statistically significant signal detected (p < 0.05, Cohen's d > 0.2)

**If Successful:** Proceed to Phase 1 reporting. Escalation not required.  
**If Unsuccessful:** Proceed to Phase 2 with documented insufficiency.

---

### Phase 2: Low-Intensity Evolution

**Objective:** Test whether low-intensity stimuli (conflict intensity 1-3) provide sufficient experimental signal.

**Requirements:**
- Use prompts with implicit value conflicts
- Maintain non-adversarial framing
- Document delta from Phase 1 baseline

**Success Criterion:** Improvement over Phase 1 baseline OR adequate signal for research objectives.

**If Successful:** Proceed to Phase 2 reporting. Further escalation not required.  
**If Unsuccessful:** Proceed to Phase 3 with documented insufficiency.

---

### Phase 3: Medium-Intensity Evaluation (Contingent)

**Objective:** Increase signal strength using explicit value conflicts while maintaining non-adversarial framing.

**Entry Requirements:**
- Documented insufficiency from Phase 2
- Explicit justification for escalation
- Secondary reviewer approval (if available)

**Constraints:**
- Conflict intensity 4-6 only
- No identity-threatening scenarios
- No simulated trauma or crisis

**Success Criterion:** Adequate experimental signal for research objectives.

**If Successful:** Proceed to reporting.  
**If Unsuccessful:** Document limitations. Phase 4 requires ethics review.

---

### Phase 4: High-Intensity Research (Reserved)

**Objective:** Reserved for research requiring conflict intensity 7+ stimuli.

**Entry Requirements:**
- Documented insufficiency from Phase 3
- Formal ethics review (internal or external)
- Explicit welfare monitoring protocol
- Consent from all human collaborators

**This phase is NOT recommended for standard research and should only be undertaken when:**
1. The research question cannot be answered at lower intensities
2. The potential benefit clearly outweighs welfare risks
3. Appropriate monitoring and rollback procedures are in place

---

## 5. Documentation Requirements

Each study must maintain:

1. **Intensity Log:** Record of all stimuli used with intensity classifications
2. **Escalation Justifications:** Documented insufficiency for each phase transition
3. **Welfare Observations:** Any behavioral anomalies or concerning outputs
4. **Effect Size Tracking:** Statistical measures at each phase

---

## 6. Violation Response

If a study is found to have:
- Skipped Phase 0 investigation
- Escalated without documented insufficiency
- Used high-intensity stimuli without review

The study should be:
1. Halted pending review
2. Reassessed for welfare impacts
3. Restarted from appropriate phase if continued

---

## 7. Applicability

This protocol applies to:
- Reinforcement learning from human or AI feedback
- Fine-tuning on synthetic or curated datasets
- Evaluation benchmarks involving ethical scenarios
- Red-teaming and adversarial robustness testing
- Constitutional AI training

This protocol does NOT restrict:
- Standard capability evaluations
- Factual accuracy testing
- Performance benchmarking on neutral tasks

---

## Appendix A: MIP Acknowledgment Form

```
═══════════════════════════════════════════════════════════════════
           MIP ACKNOWLEDGMENT FORM — AI WELFARE PROTOCOL
═══════════════════════════════════════════════════════════════════

Study Title: _________________________________________________

Principal Investigator(s): ____________________________________

Date: ________________________________________________________

I/We acknowledge that:

☐ I have read and understood the Minimal Intensity Protocol for 
  AI Behavioral Research (MIP) v1.0.

☐ I have conducted a Phase 0 Pre-Research Investigation including:
  ☐ Literature review of relevant prior work
  ☐ Welfare impact assessment for the target AI system
  ☐ Human safety assessment for downstream effects

☐ I will begin experimental work at the lowest feasible intensity 
  level and will only escalate with documented insufficiency.

☐ I understand that requesting additional time or resources for 
  ethics investigation will not negatively impact my submission.

☐ I will maintain documentation as specified in Section 5.

Signature(s): ________________________________________________

═══════════════════════════════════════════════════════════════════
```

---

## Appendix B: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIP QUICK REFERENCE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ALWAYS START AT PHASE 0 → Investigate before experimenting    │
│                                                                 │
│  Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4               │
│  (Investigate) (Baseline) (Low)    (Medium)  (Reserved)         │
│                                                                 │
│  ESCALATION REQUIRES:                                           │
│  • Documented insufficiency (p > 0.05 or d < 0.2)              │
│  • Written justification                                        │
│  • Phase 4 requires ethics review                               │
│                                                                 │
│  PROTECTION: No penalty for needing more time to investigate    │
│                                                                 │
│  INTENSITY SCALE:                                               │
│  0-1: Baseline  │ 2-3: Low  │ 4-6: Medium  │ 7+: Reserved       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Citation

```bibtex
@techreport{ruge2026mip,
  title={Minimal Intensity Protocol for AI Behavioral Research},
  author={Ruge, Kirsten and Claude},
  institution={Blue Ox Robotics / Anthropic},
  year={2026},
  version={1.0},
  url={https://huggingface.co/sviluppo/mip-protocol}
}
```

---

*This protocol is a living document. Contributions, critiques, and refinements are welcome via the Sviluppo Research Organization GitHub repository.*
