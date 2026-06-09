"""TraceContext: minimal trace carrier. Extended in Phase F."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageRecord:
    name: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceContext:
    """Lightweight trace context passed through the pipeline."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    stages: List[StageRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_stage(self, name: str, **data: Any) -> None:
        self.stages.append(StageRecord(name=name, data=data))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "stages": [{"name": s.name, "data": s.data} for s in self.stages],
            "metadata": self.metadata,
        }
