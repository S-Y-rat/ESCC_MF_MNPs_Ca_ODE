# Magnetic Field-Induced Calcium Signaling Modulation in ESCC Cells

Mathematical modeling of calcium dynamics in esophageal squamous cell carcinoma (ESCC) under magnetic field effects with magnetic nanoparticles for therapeutic applications.

## Overview

This repository contains the computational model and analysis code for investigating the non-thermal effects of low-frequency magnetic fields on calcium signaling in cancer cells. The model predicts optimal magnetic field parameters (25 mT, 1.7π mHz) that can disrupt calcium oscillations and potentially inhibit cancer proliferation through MAPK and NF-κB pathway suppression.

## Key Features

- Modified Chang model incorporating mechanosensitive calcium channels
- Simulation of magnetic field-induced wall shear stress effects
- Analysis of calcium oscillation patterns under various magnetic field conditions
- Parameter optimization for therapeutic applications

## Requirements

- Python 3.12
- Install needed packages in current environment via `pip install -r requirements.txt`

## Install

Installation of developmental dependencies (`requirements-dev.txt`) is for linter. It is not required for scripts to work.

Linux/macOS:

```sh
uv venv --python 3.12

source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

Windows 11:

```powershell
uv venv --python 3.12

.venv\Scripts\activate

uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

## Usage

To run implementation with diffrax (jit-compiled with jax):

```sh
python plots.py
```

To run implementation with scipy (jit-compiled with jax):

```sh
python plots_alt.py
```

Generated `svg` and `pdf` figures will be located within `figures` directory for `diffrax` version, and within `alt_figures` directory for `scipy` version.

## Custom Parameters

To use custom parameters, please modify locally your source code copy.

You can use `CalciumModel` and `MagneticFieldParameters` directly to get methods that can be used with `diffrax.diffeqsolve` and `scipy.integrate.solve_ivp`.

Key parameters:
- Magnetic induction (B): Default $25\cdot 10^{-3}$ T.
- Frequency of oscillation ($\omega$): Default $1.7\pi\cdot 10^{-3}$ Hz.
- Simulation time: parameters T0 and T1 for start time and stop time respectively.

## Authors

- Salatskyi Y.A. (Igor Sikorsky Kyiv Polytechnic Institute), email: salatskiy.yevhen@lll.kpi.ua
- Gorobets O.U. (Igor Sikorsky Kyiv Polytechnic Institute), email: gorobets.oksana@gmail.com
- Gorobets S.V. (Igor Sikorsky Kyiv Polytechnic Institute), email: gorobetssv@gmail.com
