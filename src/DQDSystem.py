import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np

from src.base.AttributeInterpreter import AttributeInterpreter
from src.base.DoubleQuantumDot import DoubleQuantumDot
from src.base.PlotsManager import PlotsManager
from src.base.SimulationManager import SimulationManager
from src.base.auxiliaryMethods import getTimestampedFilename, getLatestSimulationFile, validateSimulationFile, \
    getLatestComparisonFile, validateComparisonFile


class DQDSystem:

    @classmethod
    def createDQDSystemFromPrecomputedData(cls, iterationParameters: List[Dict[str, Any]],
                                           folder: str = ".", date="",
                                           otherSystemDict: Dict[str, Any] = None) -> "DQDSystem":
        """
        Creates a DQDSystem instance from precomputed data.

        Args:
            iterationParameters (List[Dict[str, Any]]): Parameters to iterate over during the simulation.
            folder (str): The folder where the precomputed data is stored.

        Returns:

            DQDSystem: An instance of the DQDSystem class.
        """

        # Create a DQDSystem instance with empty fixed parameters
        dqdSystem = cls({}, iterationParameters, folder)

        # Load the data from the specified folder
        dqdSystem.loadData(date, otherSystemDict)

        return dqdSystem

    def __init__(self, fixedParameters: Dict[str, Any], iterationParameters: List[Dict[str, Any]],
                 folder: str = ".") -> None:
        """
        Initializes the DQDSystem with fixed parameters and iteration parameters.

        Args:
            fixedParameters (Dict[str, Any]): Fixed parameters for the Double Quantum Dot system.
            iterationParameters (List[Dict[str, Any]]): Parameters to iterate over during the simulation.
        """

        self.attributeInterpreter = AttributeInterpreter(fixedParameters, iterationParameters)
        self.dqdObject = self.createDqdObject()
        self.dependentArrays = None
        self.simulationName = self.attributeInterpreter.getSimulationName()
        self.folder = os.path.join(folder, "results", self.simulationName)
        self.title = ""
        self.plotOptions = {}
        self.otherToCompare = {}

        # Create the folder if it doesn't exist
        os.makedirs(self.folder, exist_ok=True)

    def createDqdObject(self) -> DoubleQuantumDot:
        """
        Creates and initializes the DoubleQuantumDot object with fixed parameters.

        Returns:
            DoubleQuantumDot: The initialized Double Quantum Dot object.
        """
        adjustedParams = self.attributeInterpreter.fixedParameters
        return DoubleQuantumDot(adjustedParams)

    def updateDQDParameters(self, *indices: int) -> None:
        """
        Updates the parameters of the DoubleQuantumDot object based on the provided indices.

        Args:
            indices (int): Indices corresponding to the current point in the simulation grid.
        """
        # Retrieve updater functions and their corresponding attribute names
        updaterFunctions = self.attributeInterpreter.getUpdateFunctions(*indices)

        # Apply each updater function to update the DQD parameters
        for updater, attributeName in updaterFunctions:
            currentValue = self.dqdObject.getAttributeValue(attributeName)
            updatedParams = updater(currentValue)
            self.dqdObject.setParameters(updatedParams)

        self.dependentArrays = []

    def updateAttributeInterpreterParameters(self, fixedParameters: Dict[str, Any],
                                             iterationParameters: List[Dict[str, Any]]) -> None:
        """
        Updates the parameters of the attributeInterpreter and resets dependentArrays.

        Args:
            fixedParameters (Dict[str, Any]): The new fixed parameters for the attributeInterpreter.
            iterationParameters (List[Dict[str, Any]]): The new iteration parameters for the attributeInterpreter.
        """
        self.attributeInterpreter = AttributeInterpreter(fixedParameters, iterationParameters)
        self.dependentArrays = []

    def simulatePoint(self) -> Tuple[float, ...]:
        """
        Simulates a single point and returns the results.

        Returns:
            Tuple[float, ...]: The results of the simulation for the current point.
        """
        return self.dqdObject.getCurrent()

    def runSimulation(self) -> List[np.ndarray]:
        """
        Runs a simulation of arbitrary dimensionality.

        Returns:
            List[np.ndarray]: A list of arrays, one for each value returned by simulatePoint.
        """
        # Get the lengths of all iteration arrays
        lenArrays = self.attributeInterpreter.lenIterationArrays()

        # Initialize the SimulationManager with the appropriate parameters
        simulationManager = SimulationManager(
            lenArrays,
            self.updateDQDParameters,
            self.simulatePoint
        )

        # Run the simulation and return the results
        return simulationManager.runSimulation()

    def fillTitle(self, titleOptions: Dict[str, Any]) -> str:
        """
        Fills the title string with the appropriate values from the DQD object.

        Args:
            titleOptions (Dict[str, Any]): A dictionary containing the title template and placeholders.

        Returns:
            str: The filled title string.
        """
        title_str = titleOptions["title"]
        placeholders = titleOptions["placeholders"]

        # Replace placeholders with the corresponding values from the DQD object
        values = []
        for placeholder in placeholders:
            name, axis, side = self.attributeInterpreter.parseAttributeString(placeholder)

            # Retrieve the value from the DQD object
            value = self.dqdObject.getAttributeValue(name)

            # Handle axis and side if applicable
            if axis is not None and side is not None:
                if axis == 3:
                    value = np.linalg.norm(value[side])
                else:
                    value = value[side, axis]
            elif axis is not None:
                if axis == 3:
                    value = np.linalg.norm(value)
                else:
                    value = value[axis]
            elif side is not None:
                value = value[side]

            values.append(str(value))

        # Format the title string with the retrieved values
        return title_str.format(*values)

    def computeDependentArrays(self) -> None:
        # Check if the simulation has already been run
        if self.dependentArrays is None:
            # Run the simulation and store the results
            self.dependentArrays = self.runSimulation()

    def simulateAndPlot(
            self,
            title: List[str] = None,
            options: Dict[str, Any] = None,
            saveData: bool = False,
            saveFigure: bool = False
    ) -> None:
        """
        Runs the simulation (if not already done), plots the results, and optionally saves the data and figure.

        Args:
            title (Dict[str, Any]): A dictionary containing title information for the plot.
            options (Dict[str, Any]): A dictionary containing plotting options.
            saveData (bool): Whether to save the simulation data to a file.
            saveFigure (bool): Whether to save the plot figure to a file.
        """

        if title is None:
            title = self.title

        title_str = self.fillTitle(self.attributeInterpreter.getTitle(title))

        if options is None:
            options = self.plotOptions

        self.computeDependentArrays()

        # Retrieve the independent arrays from AttributeInterpreter
        independentArrays = self.attributeInterpreter.getIndependentArrays()

        # Prepare plotting information
        plottingInfo = {
            "labels": [self.attributeInterpreter.getLabels(), self.attributeInterpreter.getDependentLabels()],
            "title": title_str,
            "options": options
        }

        # Plot the simulation results
        plotsManager = PlotsManager(independentArrays, self.dependentArrays, plottingInfo)
        plotsManager.plotSimulation()

        # Generate a common base filename
        baseFilename = getTimestampedFilename()

        if saveData:
            self.saveData(title, options, independentArrays, baseFilename)

        if saveFigure:
            self.saveFigure(plotsManager, baseFilename)

    def compareSimulationsAndPlot(self,
                                  otherSystemDict: Dict[str, Any] = None,
                                  title: List[str] = None,
                                  options: Dict[str, Any] = None,
                                  saveData: bool = False,
                                  saveFigure: bool = False):

        if title is None:
            title = self.title

        if options is None:
            options = self.plotOptions

        if otherSystemDict is None:
            otherSystemDict = self.otherToCompare

        fixedParametersForOtherSystem = otherSystemDict["fixedParameters"]
        iterationParametersForOtherSystem = otherSystemDict["iterationParameters"]

        title_str = self.fillTitle(self.attributeInterpreter.getTitle(title))
        otherDQDSystem = DQDSystem(fixedParametersForOtherSystem,
                                   iterationParametersForOtherSystem)

        # If self.dependentArrays is already calculated, skip the subtraction
        if not self.dependentArrays:
            # Compute dependent arrays for both systems
            otherDQDSystem.computeDependentArrays()
            self.computeDependentArrays()

            # Check if the shapes of dependent arrays are compatible
            if len(self.dependentArrays) != len(otherDQDSystem.dependentArrays):
                raise ValueError("The number of dependent arrays in both systems must be the same.")

            for idx, (selfArray, otherArray) in enumerate(zip(self.dependentArrays, otherDQDSystem.dependentArrays)):
                if selfArray.shape != otherArray.shape:
                    raise ValueError(f"Shape mismatch in dependent array at index {idx}: "
                                     f"{selfArray.shape} vs {otherArray.shape}")

            # Subtract dependent arrays and store the result in self.dependentArrays
            self.dependentArrays = [
                selfArray - otherArray
                for selfArray, otherArray in zip(self.dependentArrays, otherDQDSystem.dependentArrays)
            ]

        # Update labels for independent arrays
        independentLabels = [
            f"{self.attributeInterpreter.getLabels()[idx]}-{otherDQDSystem.attributeInterpreter.getLabels()[idx]}"
            for idx in range(len(self.attributeInterpreter.getLabels()))
        ]

        # Dependent labels remain the same
        dependentLabels = self.attributeInterpreter.getDependentLabels()

        # Retrieve the independent arrays from AttributeInterpreter
        independentArrays = self.attributeInterpreter.getIndependentArrays()

        # Divergent map
        options["colormap"] = 'RdBu_r'

        # Prepare plotting information
        plottingInfo = {
            "labels": [independentLabels, dependentLabels],
            "title": title_str,
            "options": options
        }

        # Plot the simulation results
        plotsManager = PlotsManager(independentArrays, self.dependentArrays, plottingInfo)
        plotsManager.plotSimulation()

        # Generate a common base filename
        baseFilename = f"comparisonWith_{otherDQDSystem.simulationName}_" + getTimestampedFilename()

        if saveData:
            self.saveData(title, options, independentArrays, baseFilename)

        if saveFigure:
            self.saveFigure(plotsManager, baseFilename)

    def saveData(self, title: List[str], options: Dict[str, Any], independentArrays, baseFilename: str) -> None:
        dataFolder = os.path.join(self.folder, "data")
        os.makedirs(dataFolder, exist_ok=True)
        dataFilename = os.path.join(dataFolder, f"{self.simulationName}_{baseFilename}.json")
        print(dataFilename)

        dataToSave = {
            "dqdObject": self.dqdObject.toDict(),
            "independentArrays": [array.tolist() for array in independentArrays],
            "dependentArrays": [array.tolist() for array in self.dependentArrays],
            "title": title,
            "options": options
        }

        with open(dataFilename, "w") as dataFile:
            json.dump(dataToSave, dataFile, indent=4)

    def saveFigure(self, plotsManager: PlotsManager, baseFilename: str) -> None:
        plotsFolder = os.path.join(self.folder, "plots")
        os.makedirs(plotsFolder, exist_ok=True)
        figureFilename = os.path.join(plotsFolder, f"{self.simulationName}_{baseFilename}.pdf")
        plotsManager.saveFig(figureFilename)

    def loadData(self, simulationDate: str = "", otherToCompare: Dict[str, Any] = None) -> None:
        """
        Load simulation data from a JSON file. If the date is not provided, it will load the latest simulation data
        for the given simulation name. If otherToCompare is provided, it will look for comparison files.

        Args:
            simulationDate (str, optional): The date of the simulation to load in the format "YYYYMMDD_HHMMSS". Defaults to "".
            otherToCompare (Dict[str, Any], optional): A dictionary containing "fixedParameters" and "iterationParameters".
                                                    If provided, it will look for comparison files. Defaults to None.
        """

        # Get the folder containing the results
        dataFolder = os.path.join(self.folder, "data")

        # Ensure the data folder exists
        if not os.path.exists(dataFolder):
            raise FileNotFoundError(f"The data folder '{dataFolder}' does not exist.")

        # Determine the simulation file to load
        if otherToCompare is None:
            # Workflow for standard simulation files
            if not simulationDate:
                simulationFile = getLatestSimulationFile(dataFolder, self.simulationName)
            else:
                simulationFile = validateSimulationFile(dataFolder, self.simulationName, simulationDate)
        else:
            # Workflow for comparison files
            otherDQDSystem = DQDSystem(otherToCompare["fixedParameters"],
                                   otherToCompare["iterationParameters"])
            otherSimulationName = otherDQDSystem.simulationName
            if not simulationDate:
                simulationFile = getLatestComparisonFile(dataFolder, self.simulationName, otherSimulationName)
            else:
                simulationFile = validateComparisonFile(dataFolder, self.simulationName, otherSimulationName, simulationDate)

        # Load the simulation data from the JSON file
        try:
            with open(simulationFile, "r") as dataFile:
                data = json.load(dataFile)
        except Exception as e:
            raise RuntimeError(f"Failed to load simulation data from '{simulationFile}': {e}")

        # Update the DQD object with the loaded parameters
        self.dqdObject.fromDict(data["dqdObject"])
        self.dependentArrays = [np.array(array) for array in data["dependentArrays"]]
        self.attributeInterpreter.iterationArrays = [np.array(array) for array in data["independentArrays"]]
        self.title = data["title"]
        self.plotOptions = data["options"]

        # If otherToCompare is provided, store it
        if otherToCompare is not None:
            self.otherToCompare = otherToCompare
