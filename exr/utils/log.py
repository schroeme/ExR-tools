import logging

def configure_logger(name,log_file_name='ExR-Tools_logs.log'):
    r"""
    Configures and returns a logger with both stream and file handlers.

    This function sets up a logger to send log messages to the console and to a log file. The console will display
    messages with a level of INFO and higher, while the file will contain messages with a level of DEBUG and higher.

    :param name: Name of the logger to configure. Typically, this is the name of the module calling the logger.
    :type name: str
    :param log_file_name: Name of the log file where logs will be saved. Defaults to 'ExR-Tools_logs.log'.
    :type log_file_name: str
    :return: Configured logger object.
    :rtype: logging.Logger

    :raises OSError: If there is an issue with opening or writing to the log file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    fhandler = logging.FileHandler(log_file_name,mode="a")
    fhandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    fhandler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(fhandler)

    return logger