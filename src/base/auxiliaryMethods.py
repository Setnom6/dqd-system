import os
import re
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


def getLatestSimulationFile(dataPath: str) -> str:
    """
    Get the latest simulation file for a given simulation name in the specified folder.

    Args:
        dataFolder (str): The folder containing the simulation files.
        simulationName (str): The name of the simulation.

    Returns:
        str: The path to the latest simulation file.

    Raises:
        FileNotFoundError: If no simulation files are found.
    """
    # Define the filename pattern
    dataFolder, simulationName = os.path.split(dataPath)
    simulationPattern = re.compile(rf"^{simulationName}(\d{{8}}_\d{{6}})\.npz$")

    # List all files in the folder that match the pattern
    files = [f for f in os.listdir(dataFolder) if simulationPattern.match(f)]
    if not files:
        raise FileNotFoundError(f"No simulation data found matching the pattern '{simulationName}_<timestamp>.npz'.")

    # Sort files by timestamp in descending order
    files.sort(key=lambda f: datetime.strptime(simulationPattern.match(f).group(1), "%Y%m%d_%H%M%S"), reverse=True)

    # Return the most recent file
    return os.path.join(dataFolder, files[0])
