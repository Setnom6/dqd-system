from typing import Any, Dict


class PlotOptionsManager:
    """
    Manages the plotting options for DQDSystemFactory and PlotsManager.

    This class provides a centralized way to define, update, and retrieve plotting options.
    """

    def __init__(self):
        # Default plotting options
        self.options = {
            "grid": False,  # Whether to show grid lines
            "logColorBar": False,  # Use logarithmic scale for color bar
            "colorBarMin": None,  # Minimum value for color bar
            "colorBarMax": None,  # Maximum value for color bar
            "colormap": None,  # Colormap to use for 2D plots
            "applyToAll": False,  # Apply options to all plots
            "plotOnly": None,  # Index of the dependent array to plot (only one)
            "Draw1DLines": None,  # Axis index for drawing 1D lines in 2D plots
            "gaussianFilter": False,  # Apply Gaussian filter to 2D data
        }

    def setOption(self, key: str, value: Any) -> None:
        """
        Sets a plotting option.

        Args:
            key (str): The name of the option to set.
            value (Any): The value to assign to the option.

        Raises:
            KeyError: If the option does not exist.
        """
        if key not in self.options:
            raise KeyError(f"Plotting option '{key}' does not exist.")
        self.options[key] = value

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
