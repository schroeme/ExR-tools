import os
from pathlib import Path
from exr.utils.log import configure_logger

logger = configure_logger('ExR-Tools')


def chmod(path: Path) -> None:
    """
    Sets permissions so that users and the owner can read, write and execute files at the given path.

    :param path: Path in which privileges should be granted.
    :type path: pathlib.Path
    """
    if os.name != "nt":  # Skip for Windows OS
        try:
            path.chmod(0o766)  # octal notation for permissions
        except Exception as e:
            logger.error(f"Failed to change permissions for {path}. Error: {e}")
            raise