import json
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger

def log_steps(root_dir: Path, run_data: Dict[str, Any]) -> None:
    log_path = root_dir / "pipeline_log.json"
    runs = {"runs": []}
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                runs = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Existing pipeline_log.json is corrupted. Starting a fresh log array.")
    runs["runs"].append(run_data)
    tmp_path = log_path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)
        tmp_path.replace(log_path)
    except Exception as e:
        logger.error(f"Failed to write JSON log to {log_path}: {e}")

def locate_files(root_dir: Path, suffix: str) -> List[Path]:
    if not root_dir.exists() or not root_dir.is_dir():
        logger.error(f"Directory not found: {root_dir}")
        return []
    files = list(root_dir.rglob(f"*{suffix}"))
    return files