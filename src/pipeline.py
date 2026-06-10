import time
from datetime import datetime
from loguru import logger

from src.config import PipelineConfig
from src.core import localize, undrift, align, render
from src.utils.file_io import log_steps

def run_full_pipeline(config: PipelineConfig) -> None:
    """
    The master orchestrator. Executes enabled pipeline stages sequentially,
    passing configuration objects and tracking execution time/status.
    """
    logger.info(f"Initializing pipeline for directory: {config.root_dir}")
    
    # Initialize run tracking
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_summary = {
        "run_id": run_id,
        "start_time": datetime.now().isoformat(),
        "config_snapshot": config.model_dump(mode='json'),
        "steps": [],
        "status": "in_progress",
        "total_duration_seconds": 0.0
    }
    
    pipeline_start = time.time()
    
    try:
        # --- Stage 1: Localize ---
        if config.run.localize:
            logger.info("=== Starting Stage: Localize ===")
            step_start = time.time()
            
            # The core module returns a structured dictionary instead of printing text
            loc_results = localize.execute_localisation_batch(config.root_dir, config.ext, config.localize)
            
            run_summary["steps"].append({
                "stage": "localize",
                "duration_seconds": round(time.time() - step_start, 2),
                "status": "success",
                "details": loc_results
            })
        else:
            logger.info("Skipping Stage: Localize (Disabled in config)")

        # --- Stage 2: Undrift ---
        if config.run.undrift:
            logger.info("=== Starting Stage: Undrift ===")
            step_start = time.time()
            
            undrift_results = undrift.run_batch(config.root_dir, config.undrift)
            
            run_summary["steps"].append({
                "stage": "undrift",
                "duration_seconds": round(time.time() - step_start, 2),
                "status": "success",
                "details": undrift_results
            })
        else:
            logger.info("Skipping Stage: Undrift (Disabled in config)")

        # --- Stage 3: Align ---
        if config.run.align:
            logger.info("=== Starting Stage: Align ===")
            step_start = time.time()
            
            align_results = align.run_alignment(config.root_dir, config.align, config.undrift)
            
            run_summary["steps"].append({
                "stage": "align",
                "duration_seconds": round(time.time() - step_start, 2),
                "status": "success",
                "details": align_results
            })
        else:
            logger.info("Skipping Stage: Align (Disabled in config)")

        # --- Stage 4: Render ---
        if config.run.render:
            logger.info("=== Starting Stage: Render ===")
            step_start = time.time()
            
            render_results = render.render_channels(config.root_dir, config.render)
            
            run_summary["steps"].append({
                "stage": "render",
                "duration_seconds": round(time.time() - step_start, 2),
                "status": "success",
                "details": render_results
            })
        else:
            logger.info("Skipping Stage: Render (Disabled in config)")

        # Mark pipeline as successfully completed
        run_summary["status"] = "success"

    except Exception as e:
        # Catch any standard Python exception thrown by the core modules
        logger.exception(f"Pipeline failed critically: {e}")
        run_summary["status"] = "failed"
        run_summary["error"] = str(e)
        
    finally:
        # Ensure logging happens whether the pipeline succeeds or crashes
        run_summary["end_time"] = datetime.now().isoformat()
        run_summary["total_duration_seconds"] = round(time.time() - pipeline_start, 2)
        
        logger.info(f"Writing pipeline run log to {config.root_dir}")
        log_steps(config.root_dir, run_summary)
        
        if run_summary["status"] == "success":
            logger.success("Pipeline completed successfully!")
        else:
            logger.error("Pipeline finished with errors. Check the logs.")