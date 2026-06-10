import os
import sys
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from picasso import io
from src.config import LocalizeConfig

#Converts user config to picasso cli flags 
def picasso_config_mapper(filepath: Path, config: LocalizeConfig) -> list:
    cmd = [sys.executable, "-m", "picasso", "localize"]
    cmd.extend(["-a", config.fit_method])
    cmd.extend(["-g", str(config.gradient_threshold)])
    cmd.extend(["-b", str(config.box_side_length)])
    cmd.extend(["-bl", str(config.camera_baseline)])
    cmd.extend(["-s", str(config.camera_sensitivity)])
    cmd.extend(["-ga", str(config.camera_gain)])
    cmd.extend(["-qe", str(config.quantum_efficiency)])
    cmd.extend(["-d", str(config.drift_segmentation)])
    if config.roi:
        # The -r argument expects four separate values, not a single comma-separated string.
        cmd.extend([
            "-r", str(config.roi[0]), str(config.roi[1]), str(config.roi[2]), str(config.roi[3])
        ])
    cmd.append(str(filepath))
    return cmd

#Localisation CLI and File IO handeling
def localise_data(filepath: Path, config: LocalizeConfig, root_dir: Path) -> dict:
    cmd = picasso_config_mapper(filepath, config)
    #For Environment safekeeping and keeping all the wirings safe
    env = os.environ.copy()
    picasso_path = Path(__file__).resolve().parent.parent.parent / "picasso"
    env["PYTHONPATH"] = str(picasso_path) + os.pathsep + env.get("PYTHONPATH", "")
    #dictate cmd and capture console output for logging
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root_dir), env=env)
    out_path = filepath.with_name(f"{filepath.stem}_locs.hdf5")
#sanity check to ensure both cmd ran and output files created
    if result.returncode == 0 and out_path.exists():
        locs, _ = io.load_locs(str(out_path)) #make the hdf5 files with locs using picassos io
        loc_count = len(locs)#count locs     
        return {"file": filepath.name, "status": "success", "localizations": loc_count, "out_path": str(out_path)}
    else:
        error_message = f"Failed with return code {result.returncode}."
        if result.stderr:
            error_message += f"\nStderr:\n{result.stderr.strip()}"
        return {"file": filepath.name, "status": "failed", "error": error_message}

#Do batch processing in ssd, not in hdd, hdd throttles, so copying data to ssd recommended to get high speed processing
def execute_localisation_batch(root_dir: Path, ext: str, config: LocalizeConfig) -> Dict[str, Any]:
    files = list(root_dir.rglob(f"*{ext}"))
    logger.info(f"Found {len(files)} files to localize.") #log the files found to be localised
    #Final Count
    results = {"total_files": len(files), "success_count": 0, "failed_count": 0, "total_localizations": 0, "files": []}

    if not files:
        return results
    
#Parallel Processing of localisations
    if hasattr(config, "drive_type") and config.drive_type.lower() == "hdd":
        # HDDs struggle with heavy concurrent I/O (Disk Thrashing).
        # Set max_workers to 1 for sequential reads. 
        max_workers = 1
    else:
        # For SSDs, use all workers for faster processing.
        max_workers = max(1, os.cpu_count() - 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(localise_data, f, config, root_dir) for f in files]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results["files"].append(res)
            if res.get("status") == "success":
                results["success_count"] += 1
                results["total_localizations"] += res["localizations"]
                logger.info(f"Localized {res['file']} -> {res['localizations']} spots.")
            else:
                results["failed_count"] += 1
                logger.error(f"Failed to localize {res['file']}: {res.get('error')}")
    return results