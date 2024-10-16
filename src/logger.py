import sys, logging

logging.basicConfig(level=logging.INFO, encoding="utf-8")
logger = logging.getLogger()
formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger.handlers[0].setFormatter(formatter)

def stream_print(s):
    sys.stdout.write(s)
    sys.stdout.flush()