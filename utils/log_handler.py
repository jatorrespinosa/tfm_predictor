import logging
import logging.handlers
import os

level = logging.INFO
formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(name)s: %(message)s')

root_path = os.path.dirname(os.path.abspath(__file__)) + '\log\\root\\'
os.makedirs(root_path, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(level)

root_file = logging.handlers.TimedRotatingFileHandler(
                f'{root_path} root',
                when='midnight', interval=1
            )
root_file.setLevel(level)
root_file.setFormatter(formatter)


root_logger.addHandler(root_file)

stream = logging.StreamHandler()
stream.setLevel(level)
stream.setFormatter(formatter)

root_logger.addHandler(stream)