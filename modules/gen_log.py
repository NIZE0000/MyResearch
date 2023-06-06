import os
import logging
import argparse

def setup_logger(log_file, log_data):
    # Create a custom logger
    logger = logging.getLogger(log_file)

    if not logger.hasHandlers():
        # Configure the logger
        logger.setLevel(logging.DEBUG)

        if not os.path.exists(log_file):
            fh = logging.FileHandler(log_file)
            print('create log file')

        # Create a file handler and set the log level
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Create a formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] : %(message)s')
        fh.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(fh)

        # Log some messages
        logger.info(log_data)

    # RemoveHandler 
    for handler in logger.handlers:
        logger.removeHandler(handler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='logs/default.log', help='Path to the log file')
    parser.add_argument('--message', default='no message', help='Message to log')
    args = parser.parse_args()

    # check the log file exists
    if not os.path.exists(args.log_file):
        print(f'Log file would create at: {args.log_file}')

    # Set up the logger
    setup_logger(args.log_file, args.message)


