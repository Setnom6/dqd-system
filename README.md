# dqd-system

A modular and extensible Python framework for simulating and visualizing double quantum dot (DQD) systems. The framework
focuses on spin-orbit interactions, Zeeman splittings, and AC field-driven transport phenomena.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Core Components](#core-components)
    - [DoubleQuantumDot](#1-core-class-doublequantumdot)
    - [DQDSystem](#2-high-level-controller-dqdsystem)
    - [DQDSystemFactory](#3-factory-interface-dqdsystemfactory)
    - [PlotsOptionsManager](#4-plot-configuration-plotsoptionsmanager)
3. [Environment Configuration](#5-environment-configuration)
4. [Development Tools and Tests](#6-development-tools-and-tests)
5. [Getting Started](#getting-started)
6. [Future Work](#future-work)

---

## Repository Structure

```plaintext
dqd-system/
│
├── src/                     # Main source code
│   ├── base/                # Core components (DQD logic, utilities)
│   ├── DQDSystem.py         # High-level simulation orchestration
│   ├── DQDSystemFactory.py  # Factory with preconfigured systems and defaults
│   ├── PlotsOptionsManager.py # Global manager for plotting options
│
├── tests/                   # Unit tests for development
│
├── .env.example             # Template for environment configuration
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

---

## Core Components

### 1. Core Class: `DoubleQuantumDot`

The `DoubleQuantumDot` class defines the core physical system and parameters of a double quantum dot under AC driving
and magnetic fields. It models the physics of the system and computes key observables.

#### Attributes:

- **System Parameters**:
    - `acAmplitude (float)`: Amplitude of the AC driving field.
    - `chi (float)`: Spin-flip tunneling factor.
    - `tau (float)`: Tunneling amplitude between the dots.
    - `gamma (np.ndarray, shape (2, 1))`: Coupling rates to the leads.
    - `zeeman (np.ndarray, shape (2, 3))`: Zeeman splitting for each dot.
    - `magneticField (np.ndarray, shape (3,))`: External magnetic field.
    - `gFactor (np.ndarray, shape (2, 3, 3))`: g-factor tensor for each dot.
    - `detuning (float)`: Energy detuning between the dots.
    - `groundRightEnergy (float)`: Right-dot ground state energy.

- **Spin-Orbit Coupling**:
    - `socThetaAngle (float)`: Theta angle for spin-orbit coupling.
    - `socPhiAngle (float)`: Phi angle for spin-orbit coupling.

- **Oscillatory Magnetic Field**:
    - `OME (np.ndarray, shape (2, 3))`: Oscillatory magnetic field.
    - `factorBetweenOMEAndZeeman (float)`: Proportionality constant for OME.

#### Observables:

- **Stationary Current**: The steady-state current through the system.
- **Current Polarity**: The directionality of the current.

> **Note**: A detailed mathematical description of the model and current will be added in future updates.

---

### 2. High-Level Controller: `DQDSystem`

The `DQDSystem` class manages simulations over parameter grids. It orchestrates the following tasks:

1. Instantiates a `DoubleQuantumDot` object with fixed parameters.
2. Configures variable parameters to be scanned over.
3. Executes simulations at each grid point.
4. Visualizes results using various plotting tools.

#### Delegated Components:

- **`DQDParameterInterpreter`**: Parses fixed and iteration parameters into a format interpretable by
  `DoubleQuantumDot`. Generates updater functions for parameter scans.
- **`SimulationManager`**: Handles the simulation loop over the parameter grid. Supports parallelization via `joblib`.
- **`DQDLabelFormatter`**: Generates LaTeX-formatted labels for axes and titles in visualizations.
- **`PlotsManager`**: Centralizes plotting logic, supporting 1D/2D visualizations, logarithmic scales, and smoothing.
- **`DQDAnnotationGenerator`**: Automatically generates plot annotations for specific parameter scans (e.g., resonance
  lines).

---

### 3. Factory Interface: `DQDSystemFactory`

The `DQDSystemFactory` class simplifies the creation of commonly used DQD simulations. It provides:

- Predefined simulation setups (e.g., `ZeemanXvsZeemanZ`, `MagneticFieldXvsMagneticFieldY`).
- Centralized management of global fixed parameters.
- Automatic title formatting via `DQDLabelFormatter`.

---

### 4. Plot Configuration: `PlotsOptionsManager`

The `PlotsOptionsManager` class manages global plotting preferences, such as:

- Axis limits
- Color maps
- Smoothing filters
- Logarithmic scales

These options are automatically injected into simulations created through `DQDSystemFactory`.

---

## Environment Configuration

To enable or disable parallel execution (e.g., on SLURM environments where `joblib` may cause issues), copy and modify
the `.env.example` file:

```bash
cp .env.example .env
```

Then edit `.env` as follows:

```dotenv
DQD_PARALLEL=0  # Disable parallel execution
```

> **Note**: The `.env` file is ignored by Git and should be user-specific.

---

## Development Tools and Tests

The `tests/` folder contains unit tests to validate the behavior of individual components. These tests are primarily for
developers and are not required for typical users.

To run the tests, use:

```bash
pytest tests/
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Install dependencies:

```bash
pip install -r requirements.txt
```

### Example Usage

```python
from src.DQDSystemFactory import DQDSystemFactory

# Create a predefined simulation setup
dqd_system = DQDSystemFactory.createZeemanXvsZeemanZ()

# Run the simulation
dqd_system.runSimulation()

# Plot the results
dqd_system.plotSimulation(title="Zeeman Interaction", options={"grid": True}, saveFigure=True)
```

---

## Future Work

- Add mathematical descriptions of the physical model.
- Extend the `DQDSystemFactory` with additional predefined setups.
- Improve parallelization support for large-scale simulations.
- Add support for 3D visualizations.
- Provide example notebooks for common use cases.

---

## License

---

## Acknowledgments

This framework was developed as part of research on quantum dot systems. Special thanks to the contributors and the
open-source community for their support.


