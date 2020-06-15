# ClearText

With over 2 billion speakers, the English language is the most commonly spoken language in the world.
However, the majority of these speakers&mdash;a group estimated to be composed of up to 2 billion
people&mdash;do not speak it as their native tongue.

Leveraging natural language processing and deep learning, ClearText&mdash;a work in progress&mdash;
aims towards bridging the gap between elementary English proficiency and language mastery.

To learn more, take a look at the [notebook](notebooks/cleartext.html). You can also download and run your own version
of the notebook [here](notebooks/cleartext.ipynb).

## Installation

The ClearText package can be installed using `pip` and necessary language packs added using `make`.
All other required data will be downloaded on-the-fly.

```bash
>>> pip install git+https://github.com/bencwallace/cleartext
>>> make
```

## Training

The `train.py` script in the [scripts](https://github.com/bencwallace/cleartext/tree/master/scripts)
directory can be used to train a simplification model. Usage instructions can be printed as follows.

```bash
>>> cd scripts
>>> python train.py --help 
```

Time spent and training/validation losses will be printed at the end of each epoch. When training
completes or if you interrupt training, tests are run and diagnostics are printed.  
