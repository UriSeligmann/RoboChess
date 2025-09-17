import logging

class LoggingManager:
    """Handles all logging operations with configurable levels."""
    
    def __init__(self, name: str = "CVDebugger", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(name)s][%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        self.logger.error(message)
