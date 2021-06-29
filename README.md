# Triplet-network-for-few-shot-learning

<h1 align="center">
  <br>
Triplet network for few shot learning
  <br>
  <img src="https://omoindrot.github.io/assets/triplet_loss/triplet_loss.png" height="200">
</h1>
  <p align="center">
    <a href="https://github.com/danielt17">Daniel Teitelman</a> •
    <a href="https://github.com/GonenWeiss">Gonen Weiss</a> 
  </p>

* Picture by <a href="https://omoindrot.github.io/triplet-loss">Olivier Moindrot</a>.

</h4>


- [Triplet-network-for-few-shot-learning](#Triplet-network-for-few-shot-learning)
  * [Agenda](#agenda)
  * [Installation Instructions](#installation-instructions)


## Agenda

This project was done as the final project in the Deep Learning course (046211) at the Technion. Our project focused on reimplementing the paper 
<a href="https://arxiv.org/abs/1412.6622">"Deep Metric Learning Using Triplet Network"</a> by Elad Hoffer et al. and testing it on another dataset, the <a href="https://github.com/zalandoresearch/fashion-mnist">Fasion MNIST</a>. Afterwards we used the architecture in the setting of few-shot learning and evaluated its performance. For the project <a href="https://github.com/danielt17/Triplet-network-few-shot-learning/blob/main/Triplet_Network_for_few_shot_learning.pdf">report</a>, a short PowerPoint <a href="https://github.com/danielt17/Triplet-network-few-shot-learning/blob/main/Triplet%20network%20for%20few%20shot%20learning.pptx">presentation</a> and a YouTube <a href="https://youtu.be/c-7FTQdXjI4">video (full video)</a> in which we present our project, follow the links.

|Folders       | Content |
|----------------|---------|
|`Main`| Root of the project, containing the final report, the final presentation, the project proposal and all other files and folders|
|`code`| Contains the code files of the project, including: triplet network training for metric learning, visualization (using T-SNE) and triplet network training in the few-shot learning setting|
|`models`| Contains the models trained during the project, where each saved model is a dictionary of the model weights|
|`papers`| Contains the papers upon which this project is based|
|`pictures`| Contains pictures gathered during the project, in order to display our results|

### Running Locally

Press "Download ZIP" under the green button `Clone or download` or use `git` to clone the repository using the 
following command: `git clone https://github.com/danielt17/Triplet-network-few-shot-learning.git` (in cmd/PowerShell in Windows or in the Terminal in Linux/Mac)

## Installation Instructions

For the complete guide, with step-by-step images, please consult `Setting Up The Working Environment.pdf`

1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/products/individual
2. Install the basic packages using the provided `environment.yml` file by running: `conda env create -f environment.yml` which will create a new conda environment named `TechnionPytorch`.
3. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate TechnionPytorch`

