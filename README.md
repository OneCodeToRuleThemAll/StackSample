*StackSample Project*

This project includes notebooks for exploratory data analysis, feature engineering, and model training. It also explores a large language model (LLM) approach.
Project Structure
```
    01.EDA.ipynb: This Jupyter notebook contains all the exploratory data analysis (EDA) performed, along with the text sanitization steps taken to prepare the data for modeling.
    02.Baseline model.ipynb: This notebook details the feature engineering process and the training of a machine learning model using scikit-learn.
```
LLM Approach

The Large Language Model approach is currently available exclusively on Kaggle, as local resources did not include GPU access.

Getting Started

To get started with the StackSample project, follow the steps below to set up the project environment using Docker. This will ensure that all dependencies are correctly installed and configured.
Prerequisites

Ensure you have Docker installed on your system. If not, download and install Docker from Docker's official website.

Setup Instructions

1. Clone the StackSample repository to your local machine:

```
git clone https://github.com/OneCodeToRuleThemAll/StackSample
```
2. Change to the project directory:

```
cd StackSample
```
3. Build the Docker image from the Dockerfile included in the project:

```
docker build -t stacksample:v1 -f Dockerfile .
```
This command builds a Docker image named stacksample:v1 based on the instructions in the Dockerfile.
