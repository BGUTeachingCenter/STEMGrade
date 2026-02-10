# grader/payload_io.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

Key = Tuple[int, str]

@dataclass(frozen=True)
class LoadedPayload:
    key: Key
    qid: str
    max_points: float
    reference: Dict[str, Any]
    student: Dict[str, Any]
    ai_input: Dict[str, Any]
    path: Path

def load_payload_manifest(manifest_path: Path) -> List[LoadedPayload]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_dir = manifest_path.parent / "payloads"

    items: List[LoadedPayload] = []
    for it in manifest["items"]:
        p = payload_dir / it["payload_file"]
        data = json.loads(p.read_text(encoding="utf-8"))

        key = data["key"]
        k: Key = (int(key["qnum"]), str(key.get("part") or ""))

        items.append(
            LoadedPayload(
                key=k,
                qid=str(data["qid"]),
                max_points=float(data.get("max_points") or 0.0),
                reference=data.get("reference", {}),
                student=data.get("student", {}),
                ai_input=data.get("ai_input", {}),
                path=p,
            )
        )
    return items
