from datetime import datetime


def getTimestampedFilename() -> str:
    """
    Generates a filename based on the current date and time.

    Args:
        extension (str): The file extension (default is "json").

    Returns:
        str: A string representing the filename with a timestamp.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}"


def formatComputationTime(seconds):
    """Convert computation time in seconds to a human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} h {minutes} m {seconds} s"
