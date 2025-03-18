import logging
import os
import datetime

# Ensure logs directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a unique log file for each run (timestamped)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_dir, f"run_{timestamp}.log")

# Configure logger
logger = logging.getLogger("MyGlobalLogger")
logger.setLevel(logging.INFO)

# File handler (logs to a file)
file_handler = logging.FileHandler(log_filename, mode="w")
file_handler.setLevel(logging.INFO)

# Console handler (optional, logs to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Log format
# formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
formatter = logging.Formatter("%(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Prevent duplicate log entries in Jupyter
logger.propagate = False
