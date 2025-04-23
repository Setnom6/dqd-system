import os
from typing import Dict, Any, List, Tuple

import numpy as np

from src.base.AttributeInterpreter import AttributeInterpreter
from src.base.DoubleQuantumDot import DoubleQuantumDot
from src.base.PlotsManager import PlotsManager
from src.base.SimulationManager import SimulationManager
from src.base.auxiliaryMethods import getTimestampedFilename, getLatestSimulationFile


class DQDSystem:
    """
    Represents a Double Quantum Dot (DQD) system for simulation and analysis.

    This class manages the initialization, simulation, plotting, and comparison of DQD systems.

    Attributes:
        dqdObject (DoubleQuantumDot): The Double Quantum Dot object representing the system.
        independentArrays (List[np.ndarray]): Arrays representing the independent variables for the simulation.
        dependentArrays (List[np.ndarray]): Arrays representing the dependent variables resulting from the simulation.
        simulationName (str): The name of the simulation, derived from the iteration parameters.
        folder (str): The folder where simulation results are stored.
        plottingInfo (Dict[str, Any]): Information for plotting the simulation results.
        otherDQD (DQDSystem): Another DQDSystem instance for comparison.
    """

    def __init__(self, fixedParameters: Dict[str, Any] = None, iterationParameters: List[Dict[str, Any]] = None,
                 folder: str = ".") -> None:
        """
        Initializes the DQDSystem with fixed parameters and iteration parameters.

        Args:
            fixedParameters (Dict[str, Any]): Fixed parameters for the Double Quantum Dot system.
            iterationParameters (List[Dict[str, Any]]): Parameters to iterate over during the simulation.
            folder (str): The folder where simulation results will be stored.
        """
        if iterationParameters is not None:
            self.attributeInterpreter = AttributeInterpreter(fixedParameters, iterationParameters)
            self.independentArrays = self.attributeInterpreter.getIndependentArrays()
            self.simulationName = self.attributeInterpreter.getSimulationName()
        else:
            self.attributeInterpreter = None
            self.independentArrays = None
            self.simulationName = ""

        self.dqdObject = self._createDqdObject()
        self.dependentArrays = None
        self.folder = os.path.join(folder, "results", self.simulationName)
        self.plottingInfo = {}
        self.otherDQD = None

        # Create the folder if it doesn't exist
        if self.simulationName:
            os.makedirs(self.folder, exist_ok=True)

    def _createDqdObject(self) -> DoubleQuantumDot:
        """
        Creates and initializes the DoubleQuantumDot object with fixed parameters.

        Returns:
            DoubleQuantumDot: The initialized Double Quantum Dot object.
        """
        if self.attributeInterpreter is not None:
            return DoubleQuantumDot(self.attributeInterpreter.fixedParameters)
        return DoubleQuantumDot()

    def _updateDQDParameters(self, *indices: int) -> None:
        """
        Updates the parameters of the DoubleQuantumDot object based on the provided indices.

        Args:
            indices (int): Indices corresponding to the current point in the simulation grid.
        """
        updaterFunctions = self.attributeInterpreter.getUpdateFunctions(*indices)
        for updater, attributeName in updaterFunctions:
            currentValue = self.dqdObject.getAttributeValue(attributeName)
            updatedParams = updater(currentValue)
            self.dqdObject.setParameters(updatedParams)
        self.dependentArrays = []

    def _simulatePoint(self) -> Tuple[float, ...]:
        """
        Simulates a single point and returns the results.

        Returns:
            Tuple[float, ...]: The results of the simulation for the current point.
        """
        return self.dqdObject.getCurrent()

    def runSimulation(self) -> None:
        """
        Runs a simulation of arbitrary dimensionality and stores the results in dependentArrays.
        """
        if self.attributeInterpreter is None:
            if self.dependentArrays is not None:
                raise RuntimeError("Simulation already computed.")
            raise RuntimeError("DQDSystem is not properly initialized.")

        lenArrays = self.attributeInterpreter.lenIterationArrays()
        simulationManager = SimulationManager(lenArrays, self._updateDQDParameters, self._simulatePoint)
        self.dependentArrays = simulationManager.runSimulation()

    def plotSimulation(self, title: List[str] = None, options: Dict[str, Any] = None,
                       saveData: bool = False, saveFigure: bool = False) -> None:
        """
        Plots the simulation results and optionally saves the data and figure.

        Args:
            title (List[str]): A list of strings specifying the title components.
            options (Dict[str, Any]): A dictionary containing plotting options.
            saveData (bool): Whether to save the simulation data to a file.
            saveFigure (bool): Whether to save the plot figure to a file.
        """
        if self.dependentArrays is None:
            raise RuntimeError("runSimulation must be called before plotting results.")

        if self.attributeInterpreter is not None:
            title_str = self._fillTitle(self.attributeInterpreter.getTitle(title or []))
            self.plottingInfo = {
                "labels": [self.attributeInterpreter.getLabels(), self.attributeInterpreter.getDependentLabels()],
                "title": title_str,
                "options": options or {}
            }

        plotsManager = PlotsManager(self.independentArrays, self.dependentArrays, self.plottingInfo)
        plotsManager.plotSimulation()

        baseFilename = getTimestampedFilename()
        if saveData:
            self._saveData(baseFilename)
        if saveFigure:
            self._saveFigure(plotsManager, baseFilename)

    def compareSimulationsAndPlot(self, otherSystemDict: 'DQDSystem' = None, title: List[str] = None,
                                  options: Dict[str, Any] = None, saveData: bool = False,
                                  saveFigure: bool = False) -> None:
        """
        Compares the current DQDSystem with another and plots the results.

        Args:
            otherSystemDict (DQDSystem): The other DQDSystem to compare with.
            title (List[str]): A list of strings specifying the title components.
            options (Dict[str, Any]): A dictionary containing plotting options.
            saveData (bool): Whether to save the comparison data to a file.
            saveFigure (bool): Whether to save the comparison plot to a file.
        """
        if self.dependentArrays is None:
            raise RuntimeError("runSimulation must be called before plotting results.")

        if otherSystemDict is None:
            if self.otherDQD is None:
                raise ValueError("The DQDSystem to compare with must be defined.")
        else:
            self.otherDQD = otherSystemDict

        if self.otherDQD.dependentArrays is None:
            self.otherDQD.runSimulation()

        subtractedDependentArrays = [
            selfArray - otherArray
            for selfArray, otherArray in zip(self.dependentArrays, self.otherDQD.dependentArrays)
        ]

        if self.attributeInterpreter is not None:
            title_str = self._fillTitle(self.attributeInterpreter.getTitle(title or []))
            independentLabels = [
                f"{self.attributeInterpreter.getLabels()[idx]}-{self.otherDQD.attributeInterpreter.getLabels()[idx]}"
                for idx in range(len(self.attributeInterpreter.getLabels()))
            ]
            dependentLabels = self.attributeInterpreter.getDependentLabels()
            options = options or {}
            options["colormap"] = 'RdBu_r'

            self.plottingInfo = {
                "labels": [independentLabels, dependentLabels],
                "title": title_str,
                "options": options
            }

        plotsManager = PlotsManager(self.independentArrays, subtractedDependentArrays, self.plottingInfo)
        plotsManager.plotSimulation()

        baseFilename = f"comparisonWith_{self.otherDQD.simulationName}_" + getTimestampedFilename()
        if saveData:
            self._saveData(baseFilename)
        if saveFigure:
            self._saveFigure(plotsManager, baseFilename)

    def _getDataPathWithoutExtension(self, baseFilename: str) -> str:
        """
        Constructs the path for saving data without the file extension.

        Args:
            baseFilename (str): The base filename.

        Returns:
            str: The full path without the file extension.
        """
        return os.path.join(self.folder, "data", f"{self.simulationName}_{baseFilename}")

    def _saveData(self, baseFilename: str) -> None:
        """
        Saves the simulation data to a compressed file.

        Args:
            baseFilename (str): The base filename for the data file.
        """
        filePath = self._getDataPathWithoutExtension(baseFilename)
        os.makedirs(os.path.dirname(filePath), exist_ok=True)

        otherDQDDict = {}
        if self.otherDQD is not None:
            otherDQDDict = {
                "dqdObject": self.otherDQD.dqdObject.toDict(),
                "independentArrays": np.array(self.otherDQD.independentArrays, dtype=float),
                "dependentArrays": np.array(self.otherDQD.dependentArrays, dtype=float),
                "plottingInfo": self.otherDQD.plottingInfo,
            }

        dataToSave = {
            "dqdObject": self.dqdObject.toDict(),
            "independentArrays": np.array(self.independentArrays, dtype=float),
            "dependentArrays": np.array(self.dependentArrays, dtype=float),
            "plottingInfo": self.plottingInfo,
            "otherDQD": otherDQDDict
        }

        np.savez_compressed(filePath, **dataToSave)

    def _saveFigure(self, plotsManager: PlotsManager, baseFilename: str) -> None:
        """
        Saves the plot figure to a file.

        Args:
            plotsManager (PlotsManager): The PlotsManager instance used for plotting.
            baseFilename (str): The base filename for the figure file.
        """
        plotsFolder = os.path.join(self.folder, "plots")
        os.makedirs(plotsFolder, exist_ok=True)
        figureFilename = os.path.join(plotsFolder, f"{self.simulationName}_{baseFilename}.pdf")
        plotsManager.saveFig(figureFilename)

    @staticmethod
    def loadData(iterationParameters: List[Dict[str, Any]], simulationDate: str = "", folder: str = ".") -> 'DQDSystem':
        """
        Loads a DQDSystem instance from precomputed data.

        Args:
            iterationParameters (List[Dict[str, Any]]): Parameters to iterate over during the simulation.
            simulationDate (str): The date of the simulation to load in the format "YYYYMMDD_HHMMSS".
            folder (str): The folder where the precomputed data is stored.

        Returns:
            DQDSystem: The loaded DQDSystem instance.
        """
        dqdSystemAuxiliary = DQDSystem({}, iterationParameters=iterationParameters, folder=folder)
        dataPath = dqdSystemAuxiliary._getDataPathWithoutExtension("")
        simulationFile = getLatestSimulationFile(dataPath) if not simulationDate else dataPath + simulationDate + ".npz"

        if not os.path.exists(simulationFile):
            raise FileNotFoundError(f"The data '{simulationFile}' does not exist.")

        data = np.load(simulationFile, allow_pickle=True)
        result = {
            "dqdObject": data["dqdObject"].item(),
            "independentArrays": data["independentArrays"],
            "dependentArrays": data["dependentArrays"],
            "plottingInfo": data["plottingInfo"].item(),
            "otherDQD": data["otherDQD"].item(),
        }

        dqdSystemToReturn = DQDSystem()
        dqdSystemToReturn.dqdObject.fromDict(result["dqdObject"])
        dqdSystemToReturn.independentArrays = result["independentArrays"]
        dqdSystemToReturn.dependentArrays = result["dependentArrays"]
        dqdSystemToReturn.plottingInfo = result["plottingInfo"]

        if result["otherDQD"] is not None:
            dqdSystemToReturn.otherDQD = DQDSystem()
            dqdSystemToReturn.otherDQD.dqdObject.fromDict(result["otherDQD"]["dqdObject"])
            dqdSystemToReturn.otherDQD.independentArrays = result["otherDQD"]["independentArrays"]
            dqdSystemToReturn.otherDQD.dependentArrays = result["otherDQD"]["dependentArrays"]
            dqdSystemToReturn.otherDQD.plottingInfo = result["otherDQD"]["plottingInfo"]

        return dqdSystemToReturn

    def _fillTitle(self, titleOptions: Dict[str, Any]) -> str:
        """
        Fills the title string with the appropriate values from the DQD object.

        Args:
            titleOptions (Dict[str, Any]): A dictionary containing the title template and placeholders.

        Returns:
            str: The filled title string.
        """
        title_str = titleOptions["title"]
        placeholders = titleOptions["placeholders"]

        values = [
            self._getAttributeValueFromPlaceholder(placeholder)
            for placeholder in placeholders
        ]

        return title_str.format(*values)

    def _getAttributeValueFromPlaceholder(self, placeholder: str) -> str:
        """
        Retrieves the value of an attribute from the DQD object based on a placeholder.

        Args:
            placeholder (str): The placeholder string.

        Returns:
            str: The value of the attribute as a string.
        """
        name, axis, side = self.attributeInterpreter.parseAttributeString(placeholder)
        value = self.dqdObject.getAttributeValue(name)

        if axis is not None and side is not None:
            value = np.linalg.norm(value[side]) if axis == 3 else value[side, axis]
        elif axis is not None:
            value = np.linalg.norm(value) if axis == 3 else value[axis]
        elif side is not None:
            value = value[side]

        return str(value)
