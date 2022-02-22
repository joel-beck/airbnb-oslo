# Airbnb Oslo

The goal of this project is to create price-prediction models for Airbnb accomodations located in Oslo, Norway.

The underlying data is provided by Airbnb and is [freely available on the web](http://insideairbnb.com/get-the-data.html).
This repository was created in context of the *Statistical and Deep Learning* seminar offered by the Chair of Statistics at the University of Göttingen during Winter Term 2021/22.
Thus, the project focuses on the application of modern Machine Learning and Deep Learning methods that we implemented via the popular Python libraries [scikit-learn](https://scikit-learn.org/stable/) and [pytorch](https://pytorch.org/).

If you want to jump straight to the results of our project, please refer to the `paper.pdf` and the `presentation.pdf` files in the `term-paper` folder.
The former entails a detailed description of our methods and findings in text form whereas the latter provides a compact summary on slides.

## Installation

The easiest way to use our code is to first clone this repository and then create a new virtual environment with the command

```
conda env create -f environment.yml
```

*Remarks*:

- The two notebooks `cnn_colab.ipynb` and `cnn_pretrained_colab.ipynb` in the `models` folder require GPU computations and are thus run on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb?hl=de).
To save the results from these notebooks, we connected Google Colab with our individual Google Drive.
Hence, when reproducing our findings, file paths need to be adjusted according to your configuration.

- The two notebooks mentioned above as well as the `models/cnn_visualization.py` file depend on `data-clean/front_page_responses.pkl` which is only tracked locally due to the large file size.
Thus, you need to first run the script `data-clean/scrape_pictures.py` to generate the data on your local machine.

## Project Structure

The project contains 5 relevant folders:

1. The `data-raw` folder consists of all data files, that were directly downloaded from the Airbnb website and not modified by us in any way.

1. The `data-clean` folder collects all scripts and stored data files that are connected to data *preprocessing*.
This includes *data cleaning* of the raw Airbnb data, *feature engineering* via e.g. Natural Language Processing and Webscraping and finally *feature selection*.

1. After applying our models to the data for Oslo, one component of the course was to analyze the generalization performance to a new data set.
For this task the Airbnb data for Munich was chosen.
The `data-munich` folder gathers all raw and processed data files that are only relevant for this new prediction task and are completely independent from our original analysis for Oslo.

1. The `models` folder unites all scripts and notebooks relevant for modeling.
The `sklearn_helpers.py` and `pytorch_helpers.py` files are the core components of the repository and contain all functions and classes for constructing the model architecture, fitting the models and extracting the results.
These convenience functions are then imported and leveraged in the remaining files of the folder.

    **Note to our course instructors**:
    The following files were used to create the results of our term paper:

    - `rfe.py` for analyzing the performance on training and validation set of all classical Machine Learning Models and our Neural Network with a varying number of input features.
    - `rfe_results.py` for creating the figures corresponding to these results and evaluating all models on a separate test set.
    - `cnn_visualizations.py` for generating example images with original and predicted prices from the pretrained `ResNet`.
    - `mlp_dropout.py` for quantifying the impact of the Dropout probability on the performance of our Network.
    - `outliers.py` for quantifying the impact of outliers on the performance of our Network.
    - `mlp_latent_space.py` for embedding the input features into a two-dimensional latent space as the output of the Encoder part of a Variational Autoencoder.

1. Finally, the `term-paper` folder contains all files that were used to create the final term paper as well as the presentation.
This folder is structured into a `chapter` subfolder, which collects all contents for the term paper that are later imported into the main `paper.tex` file, as well as self-explaining `images` and `tables` subfolders.


*Remarks*:

The top-level notebooks `Data_Set_Introduction.ipynb`, `Model_Introduction.ipynb` and `MLP.ipynb` were provided by our course instructors as guidelines that can be build upon.




