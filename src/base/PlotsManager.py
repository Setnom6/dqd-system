from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter


class PlotsManager:
    def __init__(self, independentArrays: List[np.ndarray], dependentArrays: List[np.ndarray],
                 plottingInfo: Dict[str, Any]):
        self.independentArrays = independentArrays
        self.dependentArrays = dependentArrays
        self.plottingInfo = plottingInfo
        self.fig = None

    def _apply_options(self, ax, idx, is2D=False):
        options = self.plottingInfo.get("options", {})
        applyToAll = options.get("applyToAll", False)

        if options.get("grid", False):
            ax.grid(True)

        # Y-axis or Colorbar Min/Max
        colorBarMin = options.get("colorBarMin", None)
        colorBarMax = options.get("colorBarMax", None)

        # Log scale (solo 1D)
        if not is2D and options.get("logColorBar", False) and (applyToAll or idx == 0):
            ax.set_yscale("log")

        if not is2D:
            if colorBarMin is not None and (applyToAll or idx == 0):
                ax.set_ylim(bottom=max(colorBarMin, np.min(self.dependentArrays[idx])))
            if colorBarMax is not None and (applyToAll or idx == 0):
                ax.set_ylim(top=max(colorBarMax, np.max(self.dependentArrays[idx])))

    def _adjustFigureSizeForTitle(self) -> None:
        """
        Adjusts the figure size if the title exceeds the current figure dimensions.
        """
        if self.fig is None:
            return

        renderer = self.fig.canvas.get_renderer()
        title = self.fig._suptitle
        if title is not None:
            bbox = title.get_window_extent(renderer=renderer)
            fig_width, fig_height = self.fig.get_size_inches()
            dpi = self.fig.dpi

            # Check if the title exceeds the figure's top boundary
            if bbox.ymax > fig_height * dpi:
                # Increase the figure height to accommodate the title
                extra_height = (bbox.ymax - fig_height * dpi) / dpi
                self.fig.set_size_inches(fig_width, fig_height + extra_height)

    def _formatTicks(self, ax) -> None:
        """
        Formats the ticks on the axes to have at most 2 decimals or scientific notation.
        """

        def formatFunc(x, _):
            if abs(x) < 1e-2 or abs(x) > 1e3:
                formatted = f"{x:.2e}"
                # Remove e+00 if present
                return formatted.replace("e+00", "")
            return f"{x:.2f}"  # Two decimals

        ax.xaxis.set_major_formatter(FuncFormatter(formatFunc))
        ax.yaxis.set_major_formatter(FuncFormatter(formatFunc))

    def _formatColorbar(self, colorbar) -> None:
        """
        Formats the colorbar values to have at most 3 decimals or scientific notation.
        """

        def formatFunc(x, _):
            if abs(x) < 1e-3 or abs(x) > 1e4:
                formatted = f"{x:.2e}"  # Scientific notation with 3 decimals
                # Remove e+00 if present
                return formatted.replace("e+00", "")
            return f"{x:.2f}"  # Three decimals

        colorbar.formatter = FuncFormatter(formatFunc)
        colorbar.update_ticks()

    def _drawAnnotations(self, ax, annotations: List[Dict[str, Any]]) -> None:
        """
        Draws annotations on the given axis, supporting both axis-aligned and full-coordinate annotations.

        Args:
            ax: The axis to draw the annotations on.
            annotations (List[Dict[str, Any]]): A list of annotations with coordinates, colors, and styles.
        """
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        for annotation in annotations:
            annotationType = annotation.get("type")
            data = annotation.get("data")
            style = annotation.get("style", {})
            axis = annotation.get("axis", None)
            useAbsoluteCoordinates = annotation.get("absoluteCoordinates", False)

            if annotationType == "line":
                if useAbsoluteCoordinates:
                    # Dibujo general con puntos (x, y)
                    xVals = data.get("x", [])
                    yVals = data.get("y", [])
                    if len(xVals) == len(yVals) and len(xVals) > 1:
                        ax.plot(xVals, yVals, **style)
                elif "y" in data and ylim[0] <= data["y"] <= ylim[1]:
                    ax.axhline(y=data["y"], **style)
                elif "x" in data and xlim[0] <= data["x"] <= xlim[1]:
                    ax.axvline(x=data["x"], **style)

            elif annotationType == "point":
                if useAbsoluteCoordinates:
                    # Full coordinates
                    x, y = data.get("x"), data.get("y")
                    if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]:
                        ax.plot(x, y, **style)
                elif axis is not None:
                    # Axis-aligned points
                    if axis == 0 and xlim[0] <= data["x"] <= xlim[1]:
                        ax.plot(data["x"], 0, **style)
                    elif axis == 1 and ylim[0] <= data["y"] <= ylim[1]:
                        ax.plot(0, data["y"], **style)

    def _setTitleAndAdjust(self, titleText: str) -> None:
        """
        Sets the title for the figure and adjusts the layout to accommodate it.
        """
        numTitleLines = titleText.count("\n") + 1
        topMargin = 0.88 - (numTitleLines - 1) * 0.05
        topMargin = max(0.65, topMargin)  # Ensure layout does not collapse
        self.fig.suptitle(titleText)
        self.fig.subplots_adjust(top=topMargin)
        self._adjustFigureSizeForTitle()

    def _plot1D(self, annotations: List[Dict[str, Any]], indices: List[int]) -> None:
        """
        Handles 1D plotting.
        """
        self.fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4), sharey=True,
                                      constrained_layout=True)

        if len(indices) == 1:
            axes = [axes]

        for ax, i in zip(axes, indices):
            ax.plot(
                self.independentArrays[0],
                self.dependentArrays[i],
                label=self.plottingInfo["labels"][1][i]
            )
            ax.set_ylabel(self.plottingInfo["labels"][1][i])
            ax.legend(loc='upper right')
            self._apply_options(ax, i, is2D=False)

            # Draw annotations if provided
            if annotations:
                self._drawAnnotations(ax, annotations)

        axes[-1].set_xlabel(self.plottingInfo["labels"][0][0])
        self._setTitleAndAdjust(self.plottingInfo["title"])

        for ax in axes:
            self._formatTicks(ax)
        plt.show()

    def _plot2D(self, annotations: List[Dict[str, Any]], indices: List[int]) -> None:
        """
        Handles 2D plotting.
        """
        self.fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4), constrained_layout=True)
        if len(indices) == 1:
            axes = [axes]

        xArray, yArray = self.independentArrays
        X, Y = np.meshgrid(xArray, yArray)
        options = self.plottingInfo.get("options", {})
        globalColormap = options.get("colormap", None)
        applyGaussianFilter = options.get("gaussianFilter", False)
        logScale = options.get("logColorBar", False)
        colorBarMin = options.get("colorBarMin", None)
        colorBarMax = options.get("colorBarMax", None)
        applyToAll = options.get("applyToAll", False)

        for ax, i in zip(axes, indices):
            Z = self.dependentArrays[i].T

            if applyGaussianFilter and (applyToAll or i == 0):
                background = gaussian_filter(Z, sigma=10)
                Z = np.maximum(Z - background, 0)

            colormap = globalColormap if globalColormap else ('viridis' if i == 0 else 'RdBu_r')

            if logScale and (applyToAll or i == 0):
                norm = LogNorm(vmin=max(np.min(Z[Z > 0]), 1e-12), vmax=np.max(Z))
            else:
                norm = None

            vmin = max(colorBarMin, np.min(Z)) if colorBarMin is not None and (applyToAll or i == 0) else None
            vmax = min(colorBarMax, np.max(Z)) if colorBarMax is not None and (applyToAll or i == 0) else None

            c = ax.pcolormesh(X, Y, Z, shading='auto', cmap=colormap, norm=norm, vmin=vmin, vmax=vmax, rasterized=True)

            if len(axes) == 1 or i == 0:
                colorbar = self.fig.colorbar(c, ax=ax, label=self.plottingInfo["labels"][1][i])
                self._formatColorbar(colorbar)

            if annotations:
                self._drawAnnotations(ax, annotations)

            ax.set_xlabel(self.plottingInfo["labels"][0][0])
            ax.set_ylabel(self.plottingInfo["labels"][0][1])
            ax.set_title(f"{self.plottingInfo['labels'][1][i]}")
            self._apply_options(ax, i, is2D=True)

        self._setTitleAndAdjust(self.plottingInfo["title"])

        for ax in axes:
            self._formatTicks(ax)
        plt.show()

    def plotSimulation(self, annotations: List[Dict[str, Any]] = None) -> None:
        numIndependentArrays = len(self.independentArrays)
        numDependentArrays = len(self.dependentArrays)
        options = self.plottingInfo.get("options", {})

        plotOnly = options.get("plotOnly", None)
        indices = [plotOnly] if plotOnly is not None else list(range(numDependentArrays))

        if numIndependentArrays == 1:
            self._plot1D(annotations, indices)
        elif numIndependentArrays == 2:
            self._plot2D(annotations, indices)
        elif numIndependentArrays == 3:
            raise NotImplementedError("3D plots are not implemented yet.")
        else:
            raise ValueError("Unsupported number of independent arrays.")

    def saveFig(self, filename: str) -> None:
        if self.fig is not None:
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            self.fig.savefig(filename)
        else:
            raise RuntimeError("No figure has been generated yet. Call plotSimulation() first.")
