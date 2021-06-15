# Triplet-network-for-few-shot-learning

<h1 align="center">
  <br>
Triplet network for few shot learning
  <br>
  <img src="https://omoindrot.github.io/assets/triplet_loss/triplet_loss.png" height="200">
</h1>
  <p align="center">
    <a href="https://github.com/danielt17">Daniel Teitelman</a> •
    <a href="https://github.com/GonenWeiss">Gonen Wiess</a> 
  </p>

* Picture by <a href="https://omoindrot.github.io/triplet-loss">Olivier Moindrot</a>.

<h4 align="center">
    <a href="https://colab.research.google.com/github/taldatech/ee046211-deep-learning"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <a href="https://nbviewer.jupyter.org/github/taldatech/ee046211-deep-learning/tree/main/"><img src="https://jupyter.org/assets/main-logo.svg" alt="Open In NBViewer"/></a>
    <a href="https://mybinder.org/v2/gh/taldatech/ee046211-deep-learning/main"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a>

</h4>


- [Triplet-network-for-few-shot-learning](#Triplet-network-for-few-shot-learning)
  * [Agenda](#agenda)
  * [Running The Notebooks](#running-the-notebooks)
    + [Running Online](#running-online)
    + [Running Locally](#running-locally)
  * [Installation Instructions](#installation-instructions)
    + [Libraries to Install](#libraries-to-install)


## Agenda

|Folders       | Content |
|----------------|---------|
|`Main`| Contains all folders, additionaly project proposal is present in this folder|
|`code`| The folders contains all of the code written during the project. Including: triplet network training for metric learning, visulization (using TSNE), triplet network training in the few shot learning setting|
|`models`| The folder holds all models trained during the project, the saved model is dictionary of the model weights|
|`papers`| This folder holds the papers the project is based upon|
|`pictures`| The folder holdss pictures gathered during the project, in order to display our results|


## Running The Notebooks
You can view the tutorials online or download and run locally.

### Running Online

|Service      | Usage |
|-------------|---------|
|Jupyter Nbviewer| Render and view the notebooks (can not edit) |
|Binder| Render, view and edit the notebooks (limited time) |
|Google Colab| Render, view, edit and save the notebooks to Google Drive (limited time) |


Jupyter Nbviewer:

[![nbviewer](https://jupyter.org/assets/main-logo.svg)](https://nbviewer.jupyter.org/github/taldatech/ee046202-unsupervised-learning-data-analysis/tree/master/)


Press on the "Open in Colab" button below to use Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/taldatech/ee046202-unsupervised-learning-data-analysis)

Or press on the "launch binder" button below to launch in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/taldatech/ee046202-unsupervised-learning-data-analysis/master)

Note: creating the Binder instance takes about ~5-10 minutes, so be patient

### Running Locally

Press "Download ZIP" under the green button `Clone or download` or use `git` to clone the repository using the 
following command: `git clone https://github.com/taldatech/ee046211-deep-learning.git` (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)

Open the folder in Jupyter Notebook (it is recommended to use Anaconda). Installation instructions can be found in `Setting Up The Working Environment.pdf`.


## Installation Instructions

For the complete guide, with step-by-step images, please consult `Setting Up The Working Environment.pdf`

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/products/individual
2. Install the basic packages using the provided `environment.yml` file by running: `conda env create -f environment.yml` which will create a new conda environment named `deep_learn`. If you did this, you will only need to install PyTorch, see the table below.
3. Alternatively, you can create a new environment for the course and install packages from scratch:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name deep_learn`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
4. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate deep_learn`
5. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`pandas`|  `conda install -c conda-forge pandas`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`seaborn`|  `conda install -c conda-forge seaborn`|
|`tqdm`| `conda install -c conda-forge tqdm`|
|`opencv`| `conda install -c conda-forge opencv`|
|`optuna`| `pip install optuna`|
|`pytorch` (cpu)| `conda install pytorch torchvision torchaudio cpuonly -c pytorch` |
|`pytorch` (gpu)| `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` |
|`torchtext`| `conda install -c pytorch torchtext`|


5. To open the notebooks, open Ananconda Navigator or run `jupyter notebook` in the terminal (or `Anaconda Prompt` in Windows) while the `deep_learn` environment is activated.
