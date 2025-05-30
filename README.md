# SOLARIS-RETINA-Unified-System
# V. Lucian Borbeleac
SOLARIS Unified System** is a computational simulation framework , designed to model conceptual photonic interactions and explore adaptive processes using a genetic algorithm. The system simulates how a "photonic output," generated based on parameters like wavelength and angle of incidence, can influence the evolution of an "adaptive retina."
# Hybrid Solaris Unified System

## 📜 Overview

The **Hybrid Solaris Unified System** is an advanced Python-based computational framework for simulating conceptual photonic interactions and exploring adaptive processes through an enhanced genetic algorithm. This system models how a "photonic output"—generated based on parameters like wavelength and angle of incidence—influences the evolution of an "adaptive retina." Key enhancements include adaptive mutation rates and strength, re-scaled fitness targets for more effective evolution, detailed logging, and 3D visualization of outputs.

Designed for environments like Google Colab, it supports GPU checks via PyTorch, though core computations are primarily handled by NumPy/SciPy.

## ✨ Features

* **Conceptual Photonic Simulation:** Generates a numerical output (`spectral_fractal_refraction`) based on wavelength, angle, refractive index, and other system parameters. The "fractal" aspect is conceptual.
* **Advanced Adaptive Retina:** Implements a genetic algorithm (`AdaptiveRetina`) where:
    * A population of "agents" (numerical vectors) evolves.
    * **Adaptive Mutation Rate & Strength:** Both the probability and magnitude of mutations dynamically adjust based on population diversity, balancing exploration and exploitation.
    * **Re-scaled Fitness Target:** The fitness evaluation scales the raw photonic output to better match the operational range of agent values, promoting more effective adaptation.
* **Unified System:** The adaptive retina's evolution is directly driven by the (scaled) photonic output generated at each simulation step.
* **Parameterizable:** Key parameters for photonic simulation, genetic algorithm, and fitness scaling are configurable.
* **Detailed Logging:** Provides real-time feedback on simulation progress, diversity, mutation rate, and strength.
* **3D Visualization:** Offers insightful 3D scatter plots of the photonic output surface as a function of wavelength and angle.

## ⚙️ System Components

### 1. Photonic Module
* **`spectral_activation(λ_nm)`:** Returns a wavelength-dependent activation coefficient.
* **`spectral_fractal_refraction(...)`:** Core function calculating the raw photonic output.
* **`entropy_condensation(data)`:** Applies `log(1+abs(data))` to condense the raw output.

### 2. Adaptive Retina Module (`AdaptiveRetina`)
* **Population:** Agents are vectors of numbers (genes) in `[0,1]`.
* **Fitness Evaluation:** Fitness is based on how closely an agent's mean value matches a *scaled version* of the `raw_photonic_output`. Scaling maps the typically very small `raw_photonic_output` to the `[0,1]` range for a more meaningful comparison.
* **Evolution:**
    * **Selection (Elitism):** Top 20% of fittest agents are preserved.
    * **Adaptive Mutation:**
        * `mutation_rate` (probability) and `mutation_strength` (magnitude) are inversely proportional to population diversity.
        * Low diversity increases both rate and strength to encourage exploration.
        * High diversity decreases them for finer-grained exploitation.

### 3. Solaris Core (`SolarisUnifiedSystem`)
* Orchestrates the simulation, initializing and linking the photonic and adaptive retina modules.
* Manages parameters for fitness scaling passed to the `AdaptiveRetina`.
* `run_evolution_cycle`:
    1.  Iterates through wavelength/angle combinations.
    2.  Calculates `raw_output` via the photonic module.
    3.  Passes `raw_output` to the `AdaptiveRetina` for fitness evaluation (which includes scaling) and evolution.
    4.  Logs diversity and mutation metrics.
    5.  Condenses and stores `raw_output`.
    6.  Returns flattened meshgrid coordinates and condensed outputs for 3D plotting.

## 🚀 How It Works

1.  The system is initialized with parameters for optical properties, GA behavior (population size, initial mutation), and fitness target scaling.
2.  For each wavelength/angle step:
    a.  A `raw_photonic_output` is calculated.
    b.  This `raw_output` is scaled to match the typical range of agent mean values (e.g., `[0,1]`).
    c.  The retina's agents are evaluated against this `scaled_target`; agents whose mean value is closer are fitter.
    d.  Population diversity is calculated.
    e.  Mutation rate and strength are adapted based on diversity.
    f.  The retina's population evolves (selection, mutation).
    g.  Key metrics (diversity, mutation parameters) are logged.
    h.  The original `raw_output` is condensed and stored.
3.  Results (condensed outputs, corresponding wavelengths, and angles) are used for 3D visualization and statistical summary.

## 📋 Requirements

* Python 3.x
* Libraries: `numpy`, `scipy` (general dependency), `matplotlib`, `pandas`, `torch` (for GPU check).

## 🛠️ Installation (Google Colab Example)

At the start of your Colab notebook:
```python
!pip install numpy scipy matplotlib pandas torch
