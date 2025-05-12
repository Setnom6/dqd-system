import os
from typing import Dict, Any, List, Tuple

import numpy as np

from src.base.DQDAnnotationGenerator import DQDAnnotationGenerator
from src.base.DQDLabelFormatter import DQDLabelFormatter
from src.base.DQDParameterInterpreter import DQDParameterInterpreter
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
            self.parameterInterpreter = DQDParameterInterpreter(fixedParameters, iterationParameters)
            self.independentArrays = self.parameterInterpreter.getIndependentArrays()
            self.simulationName = self.parameterInterpreter.getSimulationName()
            self.labelFormatter = DQDLabelFormatter(self.parameterInterpreter.getIterationFeatures())
        else:
            self.parameterInterpreter = None
            self.independentArrays = None
            self.simulationName = ""
            self.labelFormatter = None

        self.dqdObject = self._createDqdObject()
        self.annotationGenerator = DQDAnnotationGenerator(self.dqdObject,
                                                          self.parameterInterpreter.getIterationFeatures()
                                                          if self.parameterInterpreter else [],
                                                          self.parameterInterpreter.getIndependentArrays()
                                                          if self.parameterInterpreter else [])
        self.dependentArrays = None
        self.folder = os.path.join(folder, "results", self.simulationName)
        self.plottingInfo = {}
        self.otherDQD = None
        self.iterationFeatures = None

        if self.simulationName:
            os.makedirs(self.folder, exist_ok=True)

    def _createDqdObject(self) -> DoubleQuantumDot:
        """
        Creates and initializes the DoubleQuantumDot object with fixed parameters.

        Returns:
            DoubleQuantumDot: The initialized Double Quantum Dot object.
        """
        if self.parameterInterpreter is not None:
            return DoubleQuantumDot(self.parameterInterpreter.getFixedParameters())
        return DoubleQuantumDot()

    def _updateDQDParameters(self, *indices: int) -> None:
        """
        Updates the parameters of the DoubleQuantumDot object based on the provided indices.

        Args:
            indices (int): Indices corresponding to the current point in the simulation grid.
        """
        updaterFunctions = self.parameterInterpreter.getUpdateFunctions(*indices)
        for updater, attributeName in updaterFunctions:
            if isinstance(attributeName, tuple):
                currentValues = self.dqdObject.getAttributeValue(*attributeName)
            else:
                currentValues = self.dqdObject.getAttributeValue(attributeName)
            updatedParams = updater(currentValues)
            self.dqdObject.setParameters(updatedParams)

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
        if self.parameterInterpreter is None:
            if self.dependentArrays is not None:
                raise RuntimeError("Simulation already computed.")
            raise RuntimeError("DQDSystem is not properly initialized.")

        lenArrays = [len(arr) for arr in self.independentArrays]
        simManager = SimulationManager(lenArrays, self._updateDQDParameters, self._simulatePoint)
        self.dependentArrays = simManager.runSimulation()

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

        if self.labelFormatter:
            title_str = self._fillTitle(self.labelFormatter.getTitle(title or []))
            self.plottingInfo = {
                "labels": [self.labelFormatter.getLabels(), self.labelFormatter.getDependentLabels()],
                "title": title_str,
                "options": options or {}
            }

        plotsManager = PlotsManager(self.independentArrays, self.dependentArrays, self.plottingInfo)
        baseFilename = getTimestampedFilename()

        if saveData:
            self._saveData(baseFilename)

        plotsManager.plotSimulation()
        if saveFigure:
            self._saveFigure(plotsManager, baseFilename)

        annotations = self.annotationGenerator.generateAnnotations()
        if annotations and options["annotate"]:
            plotsManager.plotSimulation(annotations=annotations)
            if saveFigure:
                self._saveFigure(plotsManager, f"annotated_{baseFilename}")

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

        subtractedArrays = [
            selfArray - otherArray
            for selfArray, otherArray in zip(self.dependentArrays, self.otherDQD.dependentArrays)
        ]

        if self.labelFormatter:
            title_str = self._fillTitle(self.labelFormatter.getTitle(title or []))
            independentLabels = [
                f"{self.labelFormatter.getLabels()[i]}-{self.otherDQD.labelFormatter.getLabels()[i]}"
                for i in range(len(self.labelFormatter.getLabels()))
            ]
            dependentLabels = self.labelFormatter.getDependentLabels()
            options = options or {}
            options["colormap"] = 'RdBu_r'
            self.plottingInfo = {
                "labels": [independentLabels, dependentLabels],
                "title": title_str,
                "options": options
            }

        plotsManager = PlotsManager(self.independentArrays, subtractedArrays, self.plottingInfo)
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
        filePath = os.path.join(self.folder, "data", f"{self.simulationName}_{baseFilename}")
        os.makedirs(os.path.dirname(filePath), exist_ok=True)

        otherDQDData = {}
        if self.otherDQD is not None:
            # Save independent and dependent arrays of otherDQD separately
            otherIndependentArrays = {f"independentArray_{i}": arr for i, arr in
                                      enumerate(self.otherDQD.independentArrays)}
            otherDependentArrays = {f"dependentArray_{i}": arr for i, arr in enumerate(self.otherDQD.dependentArrays)}

            otherDQDData = {
                "simulationName": self.otherDQD.simulationName,
                "dqdObject": self.otherDQD.dqdObject.toDict(),
                "independentArrays": otherIndependentArrays,
                "dependentArrays": otherDependentArrays,
                "plottingInfo": self.otherDQD.plottingInfo,
            }

        # Save independent and dependent arrays separately
        independentArraysData = {f"independentArray_{i}": arr for i, arr in enumerate(self.independentArrays)}
        dependentArraysData = {f"dependentArray_{i}": arr for i, arr in enumerate(self.dependentArrays)}

        np.savez_compressed(filePath,
                            simulationName=self.simulationName,
                            dqdObject=self.dqdObject.toDict(),
                            plottingInfo=self.plottingInfo,
                            iterationFeatures=self.parameterInterpreter.getIterationFeatures() if self.parameterInterpreter is not None else self.iterationFeatures,
                            otherDQD=otherDQDData,
                            **independentArraysData,
                            **dependentArraysData)

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
        tempSystem = DQDSystem({}, iterationParameters=iterationParameters, folder=folder)
        dataPath = os.path.join(tempSystem.folder, "data", f"{tempSystem.simulationName}_")
        simulationFile = getLatestSimulationFile(dataPath) if not simulationDate else dataPath + simulationDate + ".npz"

        if not os.path.exists(simulationFile):
            raise FileNotFoundError(f"Simulation data '{simulationFile}' not found.")

        data = np.load(simulationFile, allow_pickle=True)

        # Load independent and dependent arrays
        independentArrays = [data[f"independentArray_{i}"] for i in range(len(data.files)) if
                             f"independentArray_{i}" in data]
        dependentArrays = [data[f"dependentArray_{i}"] for i in range(len(data.files)) if f"dependentArray_{i}" in data]

        # Load otherDQD data if present
        otherDQDData = data.get("otherDQD", None)
        otherDQD = None
        if otherDQDData:
            otherIndependentArrays = [otherDQDData[f"independentArray_{i}"] for i in range(len(otherDQDData))
                                      if f"independentArray_{i}" in otherDQDData]
            otherDependentArrays = [otherDQDData[f"dependentArray_{i}"] for i in range(len(otherDQDData))
                                    if f"dependentArray_{i}" in otherDQDData]

            otherDQD = DQDSystem()
            otherDQD.simulationName = otherDQDData["simulationName"]
            otherDQD.dqdObject.fromDict(otherDQDData["dqdObject"])
            otherDQD.independentArrays = otherIndependentArrays
            otherDQD.dependentArrays = otherDependentArrays
            otherDQD.plottingInfo = otherDQDData["plottingInfo"]

        # Initialize the system
        system = DQDSystem()
        system.simulationName = data["simulationName"].item()
        system.dqdObject.fromDict(data["dqdObject"].item())
        system.independentArrays = independentArrays
        system.dependentArrays = dependentArrays
        system.plottingInfo = data["plottingInfo"].item()
        system.iterationFeatures = data["iterationFeatures"].tolist()
        system.labelFormatter = DQDLabelFormatter(data["iterationFeatures"].tolist())
        system.annotationGenerator = DQDAnnotationGenerator(system.dqdObject, data["iterationFeatures"].tolist())
        system.otherDQD = otherDQD
        system.folder = os.path.join(folder, "results", data["simulationName"].item())

        return system

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

    def _fillTitle(self, titleOptions: Dict[str, Any], maxLineLength: int = 80) -> str:
        """
        Fills and formats the title string from the DQD object values,
        wrapping long lines and formatting nested arrays with indentation.
        """
        titleStr = titleOptions["title"]
        placeholders = titleOptions["placeholders"]
        values = [self._getAttributeValueFromPlaceholder(ph) for ph in placeholders]
        filledTitle = titleStr.format(*values)

        parts = []
        current = ""
        bracketLevel = 0
        for char in filledTitle:
            if char == "[":
                bracketLevel += 1
            elif char == "]":
                bracketLevel -= 1
            if char == "," and bracketLevel == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        if current:
            parts.append(current.strip())

        formattedLines = []
        currentLine = ""
        for part in parts:
            isArray = part.startswith("[[") and part.endswith("]]")
            if isArray:
                if currentLine:
                    formattedLines.append(currentLine.strip())
                    currentLine = ""
                formattedLines.append(part)
            else:
                if len(currentLine + ", " + part) > maxLineLength:
                    formattedLines.append(currentLine.strip())
                    currentLine = part
                else:
                    currentLine += ", " + part if currentLine else part
        if currentLine:
            formattedLines.append(currentLine.strip())

        def formatNestedArrayWithIndent(fullLine: str) -> str:
            if "=" not in fullLine:
                return fullLine

            label, arrayPart = fullLine.split("=", maxsplit=1)
            arrayPart = arrayPart.strip()
            if not arrayPart.startswith("[[") or not arrayPart.endswith("]]"):
                return fullLine

            inner = arrayPart[2:-2]  # remove outer [[ ]]
            rows = inner.split("], [")
            formattedRows = [f"  [{row.strip()}]" for row in rows]
            return f"{label.strip()} = [[\n" + "\n".join(formattedRows) + "\n]]"

        finalLines = []
        for line in formattedLines:
            if "[[" in line and "]]" in line:
                finalLines.append(formatNestedArrayWithIndent(line))
            else:
                finalLines.append(line)

        return "\n".join(finalLines)

    def _getAttributeValueFromPlaceholder(self, placeholder: str) -> str:
        """
        Retrieves the value of an attribute from the DQD object based on a placeholder.

        Args:
            placeholder (str): The placeholder string.

        Returns:
            str: The value of the attribute as a string, formatted to 2 decimal places if numeric.
        """
        name, axis, side = DQDLabelFormatter.parseAttributeString(placeholder)
        value = self.dqdObject.getAttributeValue(name)

        if axis is not None and side is not None:
            value = np.linalg.norm(value[side]) if axis == 3 else value[side, axis]
        elif axis is not None:
            value = np.linalg.norm(value) if axis == 3 else value[axis]
        elif side is not None:
            value = value[side]

        if isinstance(value, (float, np.floating)):
            return f"{value:.2f}"
        elif isinstance(value, (np.ndarray, list)):
            # Handle multi-dimensional arrays
            if np.ndim(value) == 1:
                return "[" + ", ".join(f"{v:.2f}" for v in value) + "]"
            elif np.ndim(value) == 2:
                return "[" + ", ".join("[" + ", ".join(f"{v:.2f}" for v in row) + "]" for row in value) + "]"
            else:
                return f"Array with shape {value.shape}"

        return str(value)
