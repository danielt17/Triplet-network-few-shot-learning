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

</h4>


- [Triplet-network-for-few-shot-learning](#Triplet-network-for-few-shot-learning)
  * [Agenda](#agenda)
  * [Installation Instructions](#installation-instructions)


## Agenda

This project was done as the final project in the course deep learning (046211) at the technion. Our project focused on reimplmentating the paper 
<a href="https://arxiv.org/abs/1412.6622">"Deep Metric Learning Using Triplet Network"</a> by Elad Hoffer, et.al and testing it on a another dataset named: <a href="https://github.com/zalandoresearch/fashion-mnist">Fasion MNIST</a>. Afterwards we used the architecture in the setting of few shot learning and evaluted its performance.

|Folders       | Content |
|----------------|---------|
|`Main`| Contains all folders, additionaly project proposal is present in this folder|
|`code`| The folders contains all of the code written during the project. Including: triplet network training for metric learning, visulization (using TSNE), triplet network training in the few shot learning setting|
|`models`| The folder holds all models trained during the project, the saved model is dictionary of the model weights|
|`papers`| This folder holds the papers the project is based upon|
|`pictures`| The folder holdss pictures gathered during the project, in order to display our results|

### Running Locally

Press "Download ZIP" under the green button `Clone or download` or use `git` to clone the repository using the 
following command: `git clone https://github.com/danielt17/Triplet-loss-few-shot-learning.git` (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)

## Installation Instructions

For the complete guide, with step-by-step images, please consult `Setting Up The Working Environment.pdf`

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/products/individual
2. Install the basic packages using the provided `environment.yml` file by running: `conda env create -f environment.yml` which will create a new conda environment named `TechnionPytorch`.
3. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate TechnionPytorch`

