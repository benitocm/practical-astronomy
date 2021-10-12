import logging
import sys
from logging.handlers import TimedRotatingFileHandler

#https://www.toptal.com/python/in-depth-python-logging

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(lineno)d — %(message)s")

LOG_FILE = "my_app.log"

def get_console_handler():
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(FORMATTER)
	return console_handler

def get_file_handler():
	file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
	file_handler.setFormatter(FORMATTER)
	return file_handler

def get_logger(logger_name):
	logger = logging.getLogger(logger_name)

	#logger.setLevel(logging.INFO) 
	logger.setLevel(logging.WARNING) 
    #logger.setLevel(logging.INFO) 

	#logger.addHandler(get_console_handler())
	logger.addHandler(get_file_handler())

	# with this pattern, it's rarely necessary to propagate the error up to parent
	logger.propagate = False

	return logger


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.error("Test")
