# ClearText

With over 2 billion speakers, the English language is the most commonly spoken language in the world.
However, the majority of these speakers&mdash;a group estimated to be composed of up to 2 billion
people&mdash;do not speak it as their native tongue.

Leveraging natural language processing and deep learning, ClearText&mdash;a work in progress&mdash;
aims towards bridging the gap between elementary English proficiency and language mastery.

## Installation

The ClearText package can be installed using `pip` and necessary language packs added using `make`.
All other required data will be downloaded on-the-fly.

```bash
>>> pip install git+https://github.com/bencwallace/cleartexet
>>> make
```

## Training a simplification model

The `train.py` script in the [scripts](https://github.com/bencwallace/cleartext/tree/master/scripts)
directory can be used to train a simplification model. Usage instructions can be printed as follows.

```bash
>>> cd scripts
>>> python -m train.py --help 
```
