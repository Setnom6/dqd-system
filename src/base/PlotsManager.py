from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


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

    def plotSimulation(self) -> None:
        numIndependentArrays = len(self.independentArrays)
        numDependentArrays = len(self.dependentArrays)
        options = self.plottingInfo.get("options", {})

        plotOnly = options.get("plotOnly", None)
        indices = [plotOnly] if plotOnly is not None else list(range(numDependentArrays))
        applyToAll = options.get("applyToAll", False)

        if numIndependentArrays == 1:
            self.fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4), sharey=True)
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

            axes[-1].set_xlabel(self.plottingInfo["labels"][0][0])
            self.fig.suptitle(self.plottingInfo["title"])
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        elif numIndependentArrays == 2:
            self.fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4))
            if len(indices) == 1:
                axes = [axes]

            xArray, yArray = self.independentArrays
            X, Y = np.meshgrid(xArray, yArray)

            from scipy.ndimage import gaussian_filter
            logScale = options.get("logColorBar", False)
            colorBarMin = options.get("colorBarMin", None)
            colorBarMax = options.get("colorBarMax", None)
            applyGaussianFilter = options.get("gaussianFilter", False)

            for ax, i in zip(axes, indices):
                Z = self.dependentArrays[i].T

                if applyGaussianFilter and (applyToAll or i == 0):
                    background = gaussian_filter(Z, sigma=10)
                    Z = np.maximum(Z - background, 0)

                colormap = 'viridis' if i == 0 else 'RdBu_r'
                if logScale and (applyToAll or i == 0):
                    from matplotlib.colors import LogNorm
                    norm = LogNorm(vmin=max(np.min(Z[Z > 0]), 1e-12), vmax=np.max(Z))
                else:
                    norm = None

                vmin = max(colorBarMin, np.min(Z)) if colorBarMin is not None and (applyToAll or i == 0) else None
                vmax = min(colorBarMax, np.max(Z)) if colorBarMax is not None and (applyToAll or i == 0) else None

                if norm is not None:
                    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap=colormap, norm=norm)
                else:
                    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap=colormap, vmin=vmin, vmax=vmax)

                self.fig.colorbar(c, ax=ax, label=self.plottingInfo["labels"][1][i])
                ax.set_xlabel(self.plottingInfo["labels"][0][0])
                ax.set_ylabel(self.plottingInfo["labels"][0][1])
                ax.set_title(f"{self.plottingInfo['labels'][1][i]}")
                self._apply_options(ax, i, is2D=True)

            self.fig.suptitle(self.plottingInfo["title"])
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

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
