import os
from exr.utils.log import configure_logger
logger = configure_logger('ExSeq-Toolbox')


def chmod(path):
    r"""Sets permissions so that users and the owner can read, write and execute files at the given path.

    :param str path: path in which privileges should be granted
    """
    if os.name != "nt":  # Skip for windows OS
        os.system("chmod 766 {}".format(path))

