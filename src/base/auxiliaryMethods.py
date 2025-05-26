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


def formatNumberListSmart(values):
    def needsScientific(val):
        return abs(val) > 99.99 or abs(val) < 0.001

    # Excluir ceros o valores muy pequeños en comparación con el máximo absoluto
    nonZeroValues = [abs(v) for v in values if abs(v) > 1e-12]
    maxAbs = max(nonZeroValues) if nonZeroValues else 1.0

    def isNegligible(val):
        return abs(val) < 1e-6 * maxAbs  # Por ejemplo: < 1ppm del valor mayor

    filteredValues = [v for v in values if not isNegligible(v)]
    useScientific = any(needsScientific(v) for v in filteredValues)

    formattedValues = []
    if useScientific:
        for v in values:
            if abs(v) < 1e-12:
                formattedValues.append("0")
            else:
                for decimals in range(1, 4):
                    formatted = f"{v:.{decimals}e}"
                    significand = float(formatted.split('e')[0])
                    if round(significand, decimals) == round(significand, 4):
                        break
                formattedValues.append(formatted)
    else:
        for v in values:
            formattedVal = f"{v:.3f}"
            if '.' in formattedVal:
                formattedVal = formattedVal.rstrip('0').rstrip('.')
            formattedValues.append(formattedVal)

    return formattedValues
