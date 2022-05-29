# Airbnb Oslo

![Tests](https://github.com/joel-beck/airbnb-oslo/actions/workflows/tests.yaml/badge.svg)
![Pre-Commit](https://github.com/joel-beck/airbnb-oslo/actions/workflows/pre-commit.yaml/badge.svg)

The goal of this project is to create price-prediction models for Airbnb accommodations located in Oslo, Norway.

The underlying data is provided by the *Inside Airbnb Project* and [freely available on the web](http://insideairbnb.com/get-the-data.html).
This repository was created in context of the *Statistical and Deep Learning* seminar offered by the Chair of Statistics at the University of Göttingen during Winter Term 2021/22.
Thus, the project focuses on the application of modern Machine Learning and Deep Learning methods that we implemented via the popular Python libraries [scikit-learn](https://scikit-learn.org/stable/) and [pytorch](https://pytorch.org/).

The results of this project including visualizations and interpretations of our findings are presented in written form in the `term-paper/paper.pdf` file with corresponding slides in the `term-paper/presentation.pdf` file.


## Installation

In order to use the `airbnb_oslo` package you can clone the repository from GitHub and install all required dependencies with

```
pip install .
```

If you want to contribute to development please run the command

```
pip install -e .[dev]
```

to install the `airbnb_oslo` package in *editable mode* with all development dependencies.

Finally, in order to recreate the data sets in the `data` folder, you can install the full set of dependencies with

```
pip install -e .[dev, data]
```

**Remarks**:

- The two notebooks `cnn_colab.ipynb` and `cnn_pretrained_colab.ipynb` in the `notebooks/colab` folder require GPU computations and are thus run on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb?hl=de).
To store the results from these notebooks, we connected Google Colab with our individual Google Drive.
Hence, when reproducing our findings, file paths need to be adjusted according to your configuration.

- The two notebooks mentioned above as well as the script `airbnb_oslo/models/cnn_visualization.py` file depend on the contents of `data/clean/front_page_responses.pkl` which is not tracked by Git due to its large file size.
Thus, you need to first run the script `airbnb_oslo/data/scrape_pictures.py` to generate the data locally.

## Project Structure

The project contains 4 relevant folders:

1. The `airbnb_oslo` package contains the core package functionality with all executable Python scripts.

    - The files in the `data` subfolder generate processed data files hat are stored in the top-level `data` folder.
    - The `helpers` subfolder provides building block functions and classes for constructing the model architecture, fitting the models and extracting the results. These core components are used in multiple scripts inside the `models` subfolder.
    - Files in the `models` subfolder use the preprocessed data to construct `scikit-learn` and `pytorch` models.
    All results and visualizations of the term paper are produced here.

1. The `data` folder consists of raw and processed data files as well as `pickle` files that store model outputs and model weights.

    After applying our models to the data for Oslo, one component of the course was to analyze the generalization performance to a new data set.
    For this task the Airbnb data for Munich was chosen.
    Hence the `data/munich` subfolder gathers equivalent raw and processed data files for Munich that are only relevant for this new prediction task and thus completely independent from our original analysis for Oslo.

1. The `notebooks` folder contains all Jupyter Notebooks that show some of our models in action.
These were majorly used during the development stage for exploratory purposes and to present intermediary findings to guide subsequent development steps.

1. Finally, the `term-paper` folder contains all files that are immediately connected to the final term paper as well as the presentation.
This folder is structured into a `chapters` subfolder, which collects all contents for the term paper that are later imported into the main `paper.tex` file, as well as self-explaining `images` and `tables` subfolders.
