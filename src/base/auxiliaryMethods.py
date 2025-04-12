
def formatComputationTime(seconds):
    """Convert computation time in seconds to a human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours} h {minutes} m {seconds} s"