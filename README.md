# MC-CAST

**MC-CAST** - _Markov Chains and Cellular Automata Simulation Tool_ - is a project that simulates land use change for a Brazilian municipality, from 1985 to 2040.

The tool uses a combined model of Markov Chains and Cellular Automata to predict urban expansion, integrating historical data from the MapBiomas Project with neighborhood transition rules.

The project aims to serve as a practical framework for studying urban modeling with Markov Chains and Cellular Automata.

## Overview
**Data-driven simulation**: The model is trained on a 1985–2015 dataset and validated against real changes from 2015 to 2023.

**Predictive capability**: It generates projections of land-use change up to 2040, helping to visualize future urban sprawl and its impacts.

**Replicable framework**: The code is designed to be easily adapted for other municipalities, requiring only new data inputs.

**Modular design**: The project is organized into distinct modules for data processing, model simulation, and analysis.

## Repository structure
```
MC-CAST/
├── data/
│   ├── raw/           # Raw MapBiomas data (original downloads)
│   └── processed/     # Processed data (clippings, vectors, grids)
├── src/
│   ├── models/        # Source code for the simulation model
│   └── utils/         # Helper functions (download, clip, convert)
├── notebooks/         # Analyses, tests, and visualizations (training, validation)
├── docs/              # Additional documentation and literature
├── .gitignore         # List of files to be ignored by Git
├── .gitattributes     # Git LFS configuration
├── LICENSE            # Project license
└── README.md          # This file
```
