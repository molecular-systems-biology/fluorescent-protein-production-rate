# Fluorescent protein production rate
## Introduction
This repository contains code to simplify the calculation of the production rate of fluorescent proteins from time lapse microfluidic image series. Strictly, it calculates the volume-specific rate of abundance change of a cytoplasmic, or other widely distributed, fluorescent protein.

## Installation
All installation instructions assume that you have Anaconda installed. If you do not, please see the [Anaconda installation instructions](https://docs.anaconda.com/anaconda/install/).

### Users
Use these instructions if you want to use the code in your own projects, but do not plan to make any changes to it. If you (may) want to make changes, see the "Developers" section below.

1. Download the .zip format release file. You can look through the [releases list](https://github.com/molecular-systems-biology/fluorescent-protein-production-rate/releases) to find a specific version, or just download the [latest release](https://github.com/molecular-systems-biology/fluorescent-protein-production-rate/releases/latest).
2. Unzip the file to a location of your choice.
3. Open an (Anaconda) shell and navigate to the unzipped folder. You need to be in the folder that contains the `setup.py` file.
4. Run the following command to set up a new Conda environment with the required dependencies:
   ```
   conda env create -f environment.yml
   ```
   This will create a new Conda environment with the name "fppr_env". If you want to use a different name for the environment, you can supply a `--name` argument to the command, like this:
   ```
    conda env create -f environment.yml --name my_custom_env_name
    ```
5. Activate the new environment with the following command:
    ```
    conda activate fppr_env
    ```
    If you used a custom name for the environment, replace `fppr_env` with your custom name.
6. Install the package with the following command:
    ```
    pip install .
    ```
    This will install the package in the currently active Conda environment.
7. At this point, you can delete the .zip file and the unzipped folder, as they are no longer needed. However, if you would like to use the demonstration notebook, you should keep the folder, or move the "Demo" subfolder to another location

### Developers
If you want to make changes to the code, or contribute to the project, follow these instructions. Note that you may end up with a different version of the code than the one in the latest release if changes have been committed to `main`, but not yet released!

1. Open an (Anaconda) shell and navigate to the folder where you want to clone the repository. Then run:
    ```
    git clone github.com/molecular-systems-biology/fluorescent-protein-production-rate.git
    ```
2. Navigate into the cloned repository:
   ```
   cd fluorescent-protein-production-rate
   ```
3. Create a new Conda environment with the required dependencies by running:
   ```
    conda env create -f environment.yml
    ```
    This will create a new Conda environment with the name "fppr_env". If you want to use a different name for the environment, you can supply a `--name` argument to the command, like this:
    ```
    conda env create -f environment.yml --name my_custom_env_name
    ```
4. Activate the new environment with the following command:
    ```
    conda activate fppr_env
    ```
    If you used a custom name for the environment, replace `fppr_env` with your custom name.
5. Install the package in editable mode with the following command:
    ```
    pip install -e .
    ```
    This will install the package in the currently active Conda environment, allowing you to make changes to the code and have them reflected immediately without needing to reinstall. When working in a Jupyter notebook you will likely have to restart the kernel to see the changes take effect.
6. Before committing any changes, please ensure that you have checked out to your own branch.

## Usage
To use the package, you can import it in your Python scripts or Jupyter notebooks like this:
```python
from fluorescent_protein_production_rate import (
    CellCycle,
    FluorescentProteinProductionRateExperiment,
    get_version
)
```
The `FluorescentProteinProductionRateExperiment` class is the main class for coordinating the analysis of an experiment. The `CellCycle` class contains and analyses data for a single tracked cell cycle. The `get_version` function returns the version number and SHA256 hash of the source code, which is useful for tracking which version of the code was used in different analyses.

For a more detailed guide on how to use the package, please look through the `Demo_noteboook.ipynb` notebook in the Demo subfolder.

#### Data requirements
In order to use the code in this package, you must have, for each tracked cell cycle:
- Mother cell data:
    - A series of image frame Time IDs
    - A series of estimated volume values.
    - A series of estimated fluorescent protein concentration values.
    - A set of boolean values indicating whether data for each frame should be deleted
    and replaced with linearly interpolated values.
- Previous bud data:
    - A series of image frame Time IDs
    - A series of estimated volume values.
    - A set of boolean values indicating whether data for each frame should be deleted
    and replaced with linearly interpolated values.
- Bud data:
    - A series of image frame Time IDs
    - A series of estimated volume values.
    - A set of boolean values indicating whether data for each frame should be deleted
    and replaced with linearly interpolated values.
- Cell cycle events:
    - Cell cycle event labels and corresponding Time IDs. This must include _at least_:
        - "Bud_0" : Time ID of the bud event in the cycle before the current one.
        - "Bud_1" : Time ID of the bud event in the current cycle.
        - "'cycle_end'_0" : Time ID of the end of the cycle before the current one.
        - "'cycle_end'_1" : Time ID of the end of the current cycle.
    - Note that the 'cycle_end' label stands in for whichever event you used to mark the end of cell cycles, such as "Mitotic_exit" or "Cytokinesis".

__Note__: It is strongly recommended to have some datapoints for the mother cell (and previous bud, where appropriate) which are before and after the cycle end events. This helps to produce better smoothing of the data during analysis. As a rough guide, it's recommended to have at least 3 frames before and after the cycle end events, but 8 is ideal. These guidelines match the code's default parameters.

#### Version tracking
To ensure that two analyses can be compared, it is important to keep track of the version of the code used for each analysis. Ideally simply checking the code version number should indicate whether the same code as used. However, the version number must be updated manually, so it is possible that maintainers may forget this. To double check, you should compare the SHA256 hash of the source code used for the analysis. This value will be completely different if there is _any_ difference in the code used for two anaylses.

## Development
### Contributing
Please __do not__ commit directly to the `main` branch. Instead, create a new branch for your changes, and then create a pull request to merge your changes into the `main` branch. This allows for code to be checked before merging.

Please do not submit pull requests until any additional classes or methods you have added are sufficiently documented, including use of type hints.

### Issues
If there are problems or desirable new features, consider opening an [issue](https://github.com/molecular-systems-biology/fluorescent-protein-production-rate/issues) on the repository.

### Releases
Before creating a new release, make a dedicated commit where you set the version string in `setup.py` and `fluorescent_protein_production_rate/fluorescent_protein_production_rate.py` to match the tag that you will give to the new release.

Please indicate in the release changelog which changes were made in the new release.

### Versioning
Please use the following versioning scheme for releases in the form `MAJOR.MINOR`, e.g. `1.2`:
- The first number is the major version, which should be incremented for major changes to the code. Specifically, that means __any__ changes that would break compatibility or otherwise change the result of analyses performed with previous versions. Examples of such changes include:
  - Changing the data requirements.
  - Changing the underlying analysis technique.
  - Changing method default parameters.
  - Fixing bugs which affect the output of the code.
- The second number is the minor version, which should be incremented for minor changes to the code. Specifically, that means any changes that add new features or improve existing ones, but do not break compatibility with previous versions. Examples of such changes include:
  - Adding new features to the code.
  - Improving the performance of existing code in such a way that it would not change the output.
  - Fixing bugs in the code which do not affect the output.
  - Editing the documentation without underlying code changes.
  - Improving the error messages for existing exceptions.

