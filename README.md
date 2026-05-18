# README

Code Accompanying the Paper  
“Enriching Historical Biologging Datasets on Seabirds Using Deep Neural Networks: A Transformer-Based Approach to Infer Energy Expenditure Proxy from GPS and Environmental Data”

## Contents

This folder contains the code and resources accompanying the paper. It includes:

- Data processing scripts (data not included)
- Model training code, including dataloaders and configuration files
- Results processing scripts

The core architecture (Earthformer) is not included here. It can be accessed at:  
https://github.com/amazon-science/earth-forecasting-transformer

Trained model weights are archive here: https://doi.org/10.5281/zenodo.20271368

## Setup

- Install the Earthformer architecture as a local package (e.g., using `pip install -e .` inside the Earthformer directory).
- Ensure that your Python environment can import the `earthformer` module before running the provided scripts.
- It is also possiblke to recover only the `earthformer` folder from the Earthformer repository and place it inside the ODBAFomer directory. 

## Training

Launch the following command in the terminal:

```bash
python train_ODBAFormer_covariate.py --cfg cfg_covariate.yaml --ckpt_name model_weights.ckpt --save experiment_name
```

## Evaluation

To evaluate the model, run:

```bash
python train_ODBAFormer_covariate.py --cfg cfg_covariate.yaml --ckpt_name model_weights.ckpt --save experiment_name --test true
```

## Data structure 

The datasets used in this study are avalable upon request. 
However, the repository keeps the expected directory structure so that the pipeline can be reproduced. 
