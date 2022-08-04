# DCGAN Experiments Package

Simulate experiments with a DCGAN architecture package.

<p align="center">
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.9.6-ff69b4.svg" /></a>
    <a href= "https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-1.13-2BAF2B.svg" /></a>
</p>

## Setting up the DCGAN Experiments package

- Setup the environment - 
```
make setup
```

- Allow the `.envrc` to activate vitual environment - 
```
direnv allow
```

- Install the requirements specified in `requirements.txt` - 
```
make install
```

## Using the DCGAN GAN Experiments package

For **every** experiment run:

- Clean the directory structure - 
```
make clean
```

- Reset the directory structure - 
```
make reset
```

- Define `config` parameters for package (each key in `config.py` - `config` dictionary **must** have value)

For **each** experiment run:

- Train the model - 
```
make experiments
```

- Visualize model training - inferred and true data distributions, losses [`Generator`, `Discriminator`, `Discriminator (Real Samples)`, `Discriminator (Fake Sampless)`] and generated samples (Uses `Tensorboard`) - 
```
make visualize
```

## Example For Using API

 - Training from scratch - 

```python
def train_from_scratch_example() -> None:
    '''
    Train a model from scratch.
    Returns:
        None
    '''
    # Create directory structure for the experiment
    create_directory_structure = DirectoryStructure(home_dir=config['device']['home directory'])
    create_directory_structure.create_directory_structure()

    # Create the experiments
    experiments = Experiments(config=config)

    # Train the model
    experiments.train(verbose=True, checkpoint=None)
```

- Training from checkpoint - 

```python
def train_from_checkpoint_example() -> None:
    '''
    Train a model from a checkpoint.
    Returns:
        None
    '''
    # Create the experiments
    experiments = Experiments(config=config)

    checkpoint = {
        'generator': './weights/generator_epoch_0_loss_0.pt',
        'discriminator': './weights/discriminator_epoch_0_loss_0.pt'
    }

    # Train the model
    experiments.train(verbose=True, checkpoint=checkpoint)
```

## References:

The *Deep Convolutional Generative Adversarial Network* training algorithm can be found [here](https://arxiv.org/abs/1511.06434) - 
```
@misc{https://doi.org/10.48550/arxiv.1511.06434,
  doi = {10.48550/ARXIV.1511.06434},
  url = {https://arxiv.org/abs/1511.06434},
  author = {Radford, Alec and Metz, Luke and Chintala, Soumith},
  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks},
  publisher = {arXiv},
  year = {2015},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
