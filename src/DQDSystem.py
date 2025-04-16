import numpy as np
from typing import Dict, Any, List, Tuple
from src.base.AttributeInterpreter import AttributeInterpreter
from src.base.DoubleQuantumDot import DoubleQuantumDot
from src.base.SimulationManager import SimulationManager
from src.base.PlotsManager import PlotsManager
import json
import os
from src.base.auxiliaryMethods import getTimestampedFilename


class DQDSystem:
    def __init__(self, fixedParameters: Dict[str, Any], iterationParameters: List[Dict[str, Any]], folder: str = ".") -> None:
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
                value = value[side, axis]
            elif axis is not None:
                value = value[axis]
            elif side is not None:
                value = value[side]

            values.append(value)

        # Format the title string with the retrieved values
        return title_str.format(*values)
    

    def simulateAndPlot(
        self,
        title: List[str],
        options: Dict[str, Any],
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

        title_str = self.fillTitle(self.attributeInterpreter.getTitle(title))

        # Check if the simulation has already been run
        if self.dependentArrays is None:
            # Run the simulation and store the results
            self.dependentArrays = self.runSimulation()

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

    def saveData(self, title: List[str], options: Dict[str, Any], independentArrays, baseFilename: str) -> None:
        dataFolder = os.path.join(self.folder, "data")
        os.makedirs(dataFolder, exist_ok=True)
        dataFilename = os.path.join(dataFolder, f"{baseFilename}.json")

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
        figureFilename = os.path.join(plotsFolder, f"{baseFilename}.pdf")
        plotsManager.saveFig(figureFilename)


