import warnings
from enum import Enum
from typing import Any, Dict

from matplotlib import pyplot as plt


class PlotOptions(Enum):
    GRID = "grid"
    LOG_COLOR_BAR = "logColorBar"
    COLOR_BAR_MIN = "colorBarMin"
    COLOR_BAR_MAX = "colorBarMax"
    COLOR_MAP = "colormap"
    APPLY_TO_ALL = "applyToAll"
    PLOT_ONLY = "plotOnly"
    DRAW_1D_LINES = "draw1DLines"
    GAUSSIAN_FILTER = "gaussianFilter"


class PlotOptionsManager:
    """
    Manages the plotting options for DQDSystemFactory and PlotsManager.

    This class provides a centralized way to define, update, and retrieve plotting options.
    """

    def __init__(self):
        # Default plotting options
        self.default_options = {
            "grid": False,  # Whether to show grid lines
            "logColorBar": False,  # Use logarithmic scale for color bar
            "colorBarMin": None,  # Minimum value for color bar
            "colorBarMax": None,  # Maximum value for color bar
            "colormap": None,  # Colormap to use for 2D plots
            "applyToAll": False,  # Apply options to all plots
            "plotOnly": None,  # Index of the dependent array to plot (only one)
            "draw1DLines": None,  # Axis index for drawing 1D lines in 2D plots
            "gaussianFilter": False,  # Apply Gaussian filter to 2D data
            "annotate": False,  # Apply possible annotations
        }
        self.options = self.default_options.copy()

    def _validateOption(self, key: str, value: Any) -> Any:
        """
        Validates the value of a plotting option. If the value is invalid, a warning is issued,
        and the option is reset to its default value.

        Args:
            key (str): The name of the option to validate.
            value (Any): The value to validate.

        Returns:
            Any: The validated value (or the default value if invalid).
        """
        if (key == PlotOptions.GRID.value or key == PlotOptions.LOG_COLOR_BAR.value
                or key == PlotOptions.APPLY_TO_ALL.value or key == PlotOptions.GAUSSIAN_FILTER.value):
            if not isinstance(value, bool):
                warnings.warn(f"Invalid value for '{key}': {value}. Resetting to default: {self.default_options[key]}.")
                return self.default_options[key]

        elif key == PlotOptions.COLOR_BAR_MIN.value or key == PlotOptions.COLOR_BAR_MAX.value:
            if value is not None and not isinstance(value, (int, float)):
                warnings.warn(f"Invalid value for '{key}': {value}. Resetting to default: {self.default_options[key]}.")
                return self.default_options[key]

        elif key == PlotOptions.COLOR_MAP.value:
            if value is not None:
                if not isinstance(value, str):
                    warnings.warn(
                        f"Invalid value for '{key}': {value}. Resetting to default: {self.default_options[key]}.")
                    return self.default_options[key]
                elif value not in plt.colormaps():
                    warnings.warn(f"Invalid colormap '{value}'. Resetting to default: {self.default_options[key]}.")
                    return self.default_options[key]

        elif key == PlotOptions.PLOT_ONLY.value:
            if value is not None and not isinstance(value, int):
                warnings.warn(f"Invalid value for '{key}': {value}. Resetting to default: {self.default_options[key]}.")
                return self.default_options[key]

        elif key == PlotOptions.DRAW_1D_LINES.value:
            if value is not None and not isinstance(value, int):
                warnings.warn(f"Invalid value for '{key}': {value}. Resetting to default: {self.default_options[key]}.")
                return self.default_options[key]

        return value

    def setOption(self, key: str, value: Any) -> None:
        """
        Sets a plotting option after validating its value.

        Args:
            key (str): The name of the option to set.
            value (Any): The value to assign to the option.

        Raises:
            KeyError: If the option does not exist.
        """
        if key not in self.options:
            raise KeyError(f"Plotting option '{key}' does not exist.")
        self.options[key] = self._validateOption(key, value)

    def getOption(self, key: str) -> Any:
        """
        Gets the value of a plotting option.

        Args:
            key (str): The name of the option to retrieve.

        Returns:
            Any: The value of the option.

        Raises:
            KeyError: If the option does not exist.
        """
        if key not in self.options:
            raise KeyError(f"Plotting option '{key}' does not exist.")
        return self.options[key]

    def getAllOptions(self) -> Dict[str, Any]:
        """
        Retrieves all plotting options.

        Returns:
            Dict[str, Any]: A dictionary of all plotting options and their values.
        """
        return self.options.copy()

    def resetOptions(self) -> None:
        """
        Resets all plotting options to their default values.
        """
        self.options = self.default_options.copy()
