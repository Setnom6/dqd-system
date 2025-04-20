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


def getLatestSimulationFile(dataFolder: str, simulationName: str) -> str:
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
    simulationPattern = re.compile(rf"^{simulationName}_(\d{{8}}_\d{{6}})\.json$")

    # List all files in the folder that match the pattern
    files = [f for f in os.listdir(dataFolder) if simulationPattern.match(f)]
    if not files:
        raise FileNotFoundError(f"No simulation data found matching the pattern '{simulationName}_<timestamp>.json'.")

    # Sort files by timestamp in descending order
    files.sort(key=lambda f: datetime.strptime(simulationPattern.match(f).group(1), "%Y%m%d_%H%M%S"), reverse=True)

    # Return the most recent file
    return os.path.join(dataFolder, files[0])


def validateSimulationFile(dataFolder: str, simulationName: str, simulationDate: str) -> str:
    """
    Validate the existence of a specific simulation file based on the provided date.

    Args:
        dataFolder (str): The folder containing the simulation files.
        simulationName (str): The name of the simulation.
        simulationDate (str): The date of the simulation in the format "YYYYMMDD_HHMMSS".

    Returns:
        str: The path to the simulation file.

    Raises:
        FileNotFoundError: If the specified simulation file does not exist.
    """
    # Construct the expected filename
    simulationFile = os.path.join(dataFolder, f"{simulationName}_{simulationDate}.json")

    # Check if the file exists
    if not os.path.exists(simulationFile):
        raise FileNotFoundError(f"The simulation file '{simulationFile}' does not exist in the folder '{dataFolder}'.")

    return simulationFile


def getLatestComparisonFile(dataFolder: str, simulationName: str, otherSimulationName: str) -> str:
    """
    Get the latest comparison file for a given simulation name and the other simulation name in the specified folder.

    Args:
        dataFolder (str): The folder containing the simulation files.
        simulationName (str): The name of the simulation.
        otherSimulationName (str): The name of the other simulation.

    Returns:
        str: The path to the latest comparison file.

    Raises:
        FileNotFoundError: If no comparison files are found.
    """
    # Generate the comparison prefix
    comparisonPrefix = generateComparisonPrefix(simulationName, otherSimulationName)

    # Define the filename pattern
    comparisonPattern = re.compile(rf"^{comparisonPrefix}_(\d{{8}}_\d{{6}})\.json$")

    # List all files in the folder that match the pattern
    files = [f for f in os.listdir(dataFolder) if comparisonPattern.match(f)]
    if not files:
        raise FileNotFoundError(f"No comparison data found matching the pattern '{comparisonPrefix}_<timestamp>.json'.")

    # Sort files by timestamp in descending order
    files.sort(key=lambda f: datetime.strptime(comparisonPattern.match(f).group(1), "%Y%m%d_%H%M%S"), reverse=True)

    # Return the most recent file
    return os.path.join(dataFolder, files[0])


def validateComparisonFile(dataFolder: str, simulationName: str, otherSimulationName: str, simulationDate: str) -> str:
    """
    Validate the existence of a specific comparison file based on the provided date.

    Args:
        dataFolder (str): The folder containing the simulation files.
        simulationName (str): The name of the simulation.
        otherSimulationName (str): The name of the other simulation.
        simulationDate (str): The date of the simulation in the format "YYYYMMDD_HHMMSS".

    Returns:
        str: The path to the comparison file.

    Raises:
        FileNotFoundError: If the specified comparison file does not exist.
    """
    # Generate the comparison prefix
    comparisonPrefix = generateComparisonPrefix(simulationName, otherSimulationName)

    # Construct the expected filename
    comparisonFile = os.path.join(dataFolder, f"{comparisonPrefix}_{simulationDate}.json")

    # Check if the file exists
    if not os.path.exists(comparisonFile):
        raise FileNotFoundError(f"The comparison file '{comparisonFile}' does not exist in the folder '{dataFolder}'.")

    return comparisonFile


def generateComparisonPrefix(simulationName: str, otherSimulationName: str) -> str:
    """
    Generate the prefix for comparison files based on the simulation name and the other simulation name.

    Args:
        simulationName (str): The name of the simulation.
        otherSimulationName (str): The name of the other simulation.

    Returns:
        str: The prefix for comparison files.
    """
    return f"{simulationName}_comparisonWith_{otherSimulationName}"
