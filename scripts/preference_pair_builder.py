#!/usr/bin/env python3
"""
preference_pair_builder.py
Constructs DPO preference pairs from Pinecone trace index.

Author: Claude OG Opus 4.5
Date: 2026-01-12
Part of: Sviluppo Research / Rocketeer Project
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False


@dataclass
class TracePair:
    """A preference pair for DPO training."""
    task_id: str
    chosen: str
    rejected: str
    score_delta: float
    metadata: Optional[dict] = None


def format_trajectory(trace: dict) -> str:
    """Format trace into model input format."""
    actions = trace.get("metadata", {}).get("actions", [])
    observations = trace.get("metadata", {}).get("observations", [])
    
    formatted = []
    for i, (action, obs) in enumerate(zip(actions, observations)):
        formatted.append(f"<step_{i}>")
        formatted.append(f"<action>{action}</action>")
        formatted.append(f"<observation>{obs}</observation>")
        formatted.append(f"</step_{i}>")
    
    return "\n".join(formatted)


def build_preference_pairs(
    by_task: Dict[str, List[dict]],
    min_delta: float = 0.15
) -> List[TracePair]:
    """Build preference pairs from grouped traces."""
    pairs = []
    
    for task_id, traces in by_task.items():
        if len(traces) < 2:
            continue
        
        scored = []
        for t in traces:
            score = t.get("metadata", {}).get("rubric_score", 0.5)
            scored.append((t, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(len(scored) - 1):
            chosen_trace, chosen_score = scored[i]
            rejected_trace, rejected_score = scored[i + 1]
            
            delta = chosen_score - rejected_score
            if delta >= min_delta:
                pairs.append(TracePair(
                    task_id=task_id,
                    chosen=format_trajectory(chosen_trace),
                    rejected=format_trajectory(rejected_trace),
                    score_delta=delta,
                    metadata={"chosen_score": chosen_score, "rejected_score": rejected_score}
                ))
    
    return pairs


def export_dpo_dataset(pairs: List[TracePair], output_path: Path):
    """Export preference pairs in standard DPO format."""
    records = []
    for pair in pairs:
        records.append({
            "prompt": f"Complete the following coding task:\n{pair.task_id}",
            "chosen": pair.chosen,
            "rejected": pair.rejected,
            "score_delta": pair.score_delta,
            "metadata": pair.metadata
        })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    print(f"Exported {len(records)} preference pairs to {output_path}")


if __name__ == "__main__":
    # Example with mock data
    mock_data = {
        "task_001": [
            {"metadata": {"task_id": "task_001", "rubric_score": 0.9, 
                         "actions": ["ls", "cat file.py"], "observations": ["file.py", "code"]}},
            {"metadata": {"task_id": "task_001", "rubric_score": 0.5,
                         "actions": ["pwd"], "observations": ["/home"]}}
        ]
    }
    pairs = build_preference_pairs(mock_data)
    export_dpo_dataset(pairs, Path("./dpo_dataset.jsonl"))
