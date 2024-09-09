# Data Distribution Inference Attacks in FL
This repository provides the PyTorch implementation of data distribution inference attacks in the [TKDE'24 Paper](https://ieeexplore.ieee.org/document/10380460).
## Requirements
- torch 1.7.1
- tensorflow-privacy 0.5.1
- numpy 1.16.2

## Files
> FLModel.py: Core component of the DP-based cross-silo federated learning framework

> MLModel.py: ML Models

> DPMechanisms.py: generates gaussian noise

> train-$dataset-name$.ipynb: code for DP-based FL model training

> attack-$dataset-name$.ipynb: attack method implementation.attack method

## Usage
### Step 1: Install tensorflow-privacy
Install tensorflor-privacy for calibrating Gaussian noise in the DP-SGD algorithm:

```    pip install tensorflow-privacy```

**Note:** you may also need to clone the tensorflow-privacy GitHub repository into a directory of your choice:

```    git clone https://github.com/tensorflow/privacy```

and then install the local package

```    pip install -e .```

Please refer to [tensorflow-privacy]( https://github.com/tensorflow/privacy) for more details.
Alternatively, you can calibrate DP-SGD noise using other packages such as [autodp](https://github.com/yuxiangw/autodp) and [opacus](https://github.com/pytorch/opacus).

### Step 2: Train FL models
Run the Jupyter notebook ```train-$dataset$.ipynb``` to train FL models with DP guarantees.

Key parameters for FL training are as follows:
```python
# code snippet from train-$dataset$.ipynb
lr = 0.15				# initial learning rate
fl_param = {
    'output_size': 5,             # number of units in output layer
    'client_num': client_num,     # number of parties
    'model': LogisticRegression,  # ML model
    'data': d,          # dataset
    'lr': lr,           # learning rate
    'E': 100,           # number of local iterations
    'eps': 4.0,         # privacy budget
    'delta': 1e-5,      # approximate DP: (epsilon, delta)-DP
    'q': 0.05,          # sampling rate
    'clip': 2,          # pre-example gradient norm clipping threshold
    'tot_T': 5,         # number of global iterations
    ...
}
```

### Step 3: Launch the Attack 
Run ```attack_$dataset$.ipynb``` to launch data distribution inference attacks.