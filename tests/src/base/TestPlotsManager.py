import unittest
from unittest.mock import MagicMock

import numpy as np

from src.PlotsOptionsManager import PlotOptionsManager
from src.base.PlotsManager import PlotsManager


class TestPlotsManager(unittest.TestCase):

    def setUp(self):
        """Set up test data and default plotting options."""
        # Mock independent and dependent arrays
        self.independentArrays = [np.linspace(0, 10, 100)]
        self.dependentArrays = [np.sin(self.independentArrays[0]), np.cos(self.independentArrays[0])]

        # Initialize PlotOptionsManager
        self.optionsManager = PlotOptionsManager()

        # Default plotting info
        self.plottingInfo = {
            "labels": [["X-axis"], ["Y1-axis", "Y2-axis"]],
            "title": "Test Plot",
            "options": self.optionsManager.getAllOptions()
        }

        # Create a PlotsManager instance
        self.plotsManager = PlotsManager(self.independentArrays, self.dependentArrays, self.plottingInfo)

    def testGridOption(self):
        """Test that the grid option is applied correctly."""
        self.optionsManager.setOption("grid", True)
        self.plottingInfo["options"] = self.optionsManager.getAllOptions()
        ax_mock = MagicMock()
        self.plotsManager._apply_options(ax_mock, idx=0, is2D=False)
        ax_mock.grid.assert_called_once_with(True)

    def testLogScaleOption(self):
        """Test that the log scale option is applied correctly for 1D plots."""
        self.optionsManager.setOption("logColorBar", True)
        self.plottingInfo["options"] = self.optionsManager.getAllOptions()
        ax_mock = MagicMock()
        self.plotsManager._apply_options(ax_mock, idx=0, is2D=False)
        ax_mock.set_yscale.assert_called_once_with("log")

    def testColorBarLimits(self):
        """Test that color bar min and max limits are applied correctly."""
        self.optionsManager.setOption("colorBarMin", 0.1)
        self.optionsManager.setOption("colorBarMax", 0.9)
        self.plottingInfo["options"] = self.optionsManager.getAllOptions()
        ax_mock = MagicMock()
        ax_mock.get_ylim.return_value = (0, 1)  # Mock axis limits
        self.plotsManager._apply_options(ax_mock, idx=0, is2D=False)

        # Validate the actual calls with the calculated values
        expected_top = max(0.9, np.max(self.dependentArrays[0]))
        ax_mock.set_ylim.assert_any_call(bottom=0.1)
        ax_mock.set_ylim.assert_any_call(top=expected_top)

    def testGaussianFilterOption(self):
        """Test that the Gaussian filter option is applied correctly for 2D plots."""
        self.independentArrays = [np.linspace(0, 10, 100), np.linspace(0, 10, 100)]
        self.dependentArrays = [np.random.random((100, 100))]
        self.optionsManager.setOption("gaussianFilter", True)
        self.plottingInfo["options"] = self.optionsManager.getAllOptions()
        self.plottingInfo["labels"] = [["X-axis", "Y-axis"], ["Z-axis"]]

        self.plotsManager = PlotsManager(self.independentArrays, self.dependentArrays, self.plottingInfo)
        annotations = []
        self.plotsManager._plot2D(annotations, indices=[0])

    def testColormapOption(self):
        """Test that the colormap option is applied correctly."""
        self.optionsManager.setOption("colormap", "plasma")
        self.plottingInfo["options"] = self.optionsManager.getAllOptions()
        self.independentArrays = [np.linspace(0, 10, 100), np.linspace(0, 10, 100)]
        self.dependentArrays = [np.random.random((100, 100))]
        self.plottingInfo["labels"] = [["X-axis", "Y-axis"], ["Z-axis"]]

        self.plotsManager = PlotsManager(self.independentArrays, self.dependentArrays, self.plottingInfo)
        annotations = []
        self.plotsManager._plot2D(annotations, indices=[0])

    def testAnnotations(self):
        """Test that annotations are drawn correctly."""
        annotations = [
            {"type": "line", "data": {"y": 0.5}, "style": {"color": "red", "linestyle": "--"}, "axis": 1},
            {"type": "point", "data": {"x": 5.0, "y": 0.0}, "style": {"color": "blue", "marker": "o"}, "axis": 0}
        ]
        ax_mock = MagicMock()
        ax_mock.get_xlim.return_value = (0, 10)  # Mock axis limits
        ax_mock.get_ylim.return_value = (-1, 1)
        self.plotsManager._drawAnnotations(ax_mock, annotations)

        # Check that the correct methods were called for the annotations
        ax_mock.axhline.assert_called_once_with(y=0.5, color="red", linestyle="--")
        ax_mock.plot.assert_called_once_with(5.0, 0.0, color="blue", marker="o")

    def testSaveFig(self):
        """Test that the figure is saved correctly."""
        self.plotsManager.fig = MagicMock()
        self.plotsManager.saveFig("test_plot")
        self.plotsManager.fig.savefig.assert_called_once_with("test_plot.pdf")

    def testSaveFigWithoutFigure(self):
        """Test that saving a figure without generating it raises an error."""
        self.plotsManager.fig = None
        with self.assertRaises(RuntimeError):
            self.plotsManager.saveFig("test_plot")

    def testTitleAdjustment(self):
        """Test that the title adjustment works correctly."""
        self.plotsManager.fig = MagicMock()
        self.plotsManager.fig.get_size_inches.return_value = (8, 6)  # Mock figure size
        self.plotsManager.fig.dpi = 100  # Mock DPI

        # Mock title and bbox
        title_mock = MagicMock()
        bbox_mock = MagicMock()
        bbox_mock.ymax = 700  # Simulate a title that exceeds the figure height
        title_mock.get_window_extent.return_value = bbox_mock
        self.plotsManager.fig._suptitle = title_mock

        self.plotsManager._setTitleAndAdjust("A very long title\nwith multiple lines")
        self.plotsManager.fig.subplots_adjust.assert_called()


if __name__ == "__main__":
    unittest.main()
