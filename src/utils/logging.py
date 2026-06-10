import sys
from pathlib import Path
from loguru import logger

def start_logging(log_dir: Path | str = "logs", console_level: str = "INFO"):
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=console_level
    )
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path / "pipeline_{time:YYYY-MM-DD}.log",
        rotation="10 MB",      
        retention="1000 month",   
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"          
    )