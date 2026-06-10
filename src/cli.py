import sys
import typer
from pathlib import Path
from loguru import logger
from src.config import load_config
from src.pipeline import run_full_pipeline
from src.utils.logging import start_logging

app = typer.Typer(
    name="PICASSO AUTO CLI",
    help="PICASSO AUTO CLI HELP",
    add_completion=False,
    no_args_is_help=True,
)

@app.command(name="run-all")
def run_all(
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the TOML configuration file.")
):
    try:
        config = load_config(config_path)
        start_logging(log_dir=config.root_dir)
        config.run.localize = True
        config.run.undrift = True
        config.run.align = True
        config.run.render = True
        run_full_pipeline(config)
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        sys.exit(1)

@app.command(name="localize")
def run_localize(
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the TOML configuration file.")
):
   
    try:
        config = load_config(config_path)
        start_logging(log_dir=config.root_dir)
        # Isolate this step
        config.run.localize = True
        config.run.undrift = False
        config.run.align = False
        config.run.render = False
        run_full_pipeline(config)
    except Exception as e:
        logger.error(f"Failed to start localization: {e}")
        sys.exit(1)

@app.command(name="undrift")
def run_undrift(
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the TOML configuration file.")
):
    try:
        config = load_config(config_path)
        start_logging(log_dir=config.root_dir)
        config.run.localize = False
        config.run.undrift = True
        config.run.align = False
        config.run.render = False
        
        run_full_pipeline(config)
    except Exception as e:
        logger.error(f"Failed to start undrifting: {e}")
        sys.exit(1)

@app.command(name="align")
def run_align(
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the TOML configuration file.")
):
    """Run ONLY the alignment step."""
    try:
        config = load_config(config_path)
        start_logging(log_dir=config.root_dir)
        config.run.localize = False
        config.run.undrift = False
        config.run.align = True
        config.run.render = False
        
        run_full_pipeline(config)
    except Exception as e:
        logger.error(f"Failed to start alignment: {e}")
        sys.exit(1)

@app.command(name="render")
def run_render(
    config_path: Path = typer.Option("config.toml", "--config", "-c", help="Path to the TOML configuration file.")
):
    try:
        config = load_config(config_path)
        start_logging(log_dir=config.root_dir)
        config.run.localize = False
        config.run.undrift = False
        config.run.align = False
        config.run.render = True
        
        run_full_pipeline(config)
    except Exception as e:
        logger.error(f"Failed to start rendering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()