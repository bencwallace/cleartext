<img src="chrome/icon128.png" style="float: right;">

# ClearText

Leveraging natural language processing and deep learning technology to help English language learners on the road to
fluency.

## Quick Start

For installation and usage instructions, refer to the [official ClearText extension page](https://bcwallace.com/cleartext).

## What problem does ClearText address?

### The *English as a Second Language* Market

According to [TESOL][] (Teachers of English to Speakers of Other Languages), there are over
[1.5 billion][1]
English language learners worldwide. A huge amount of human labour is involved in educating these learners. However,
many learners may not be able to afford lesson costs and must resort to other methods.

There is a large market for assisted language learning applications. For instance, [Forbes][]
reports [Babbel's](https://www.babbel.com/) revenue at \$115 million and [Duolingo's][2] 2017 valuation at \$700
million.

However, apps like these are generally limited to basic language skills that do not transfer to real world use. In order
to retain users that would otherwise outgrow them, app developers are forced to design increasingly complex and
challenging language games, which requires extensive work by multi-lingual language and education experts.

### Assisted Reading for Language Learners

Many language learners make use of subtitles in film and other media to assist them in the learning process. An English
language learner, for instance, might turn on English subtitles while watching a movie in English. The additional visual
input helps learners with their oral comprehension skills.

Unfortunately, a similar solution for text media is missing. A learner who desires to regularly read reports from a
certain English language news source in order to improve their reading comprehension skills might be frustrated at the
difficulty they encounter and the crudeness of existing forms of assistance, such as dictionaries, which do not take
*context* into account.

ClearText solves this problem through the use of text simplification technology.

## Details and Development

ClearText uses sequence-to-sequence model trained on the WikiSmall/WikiLarge datasets. For more on these datasets, take a look at
the [notebook][]. A high-level overview of the development of ClearText can be found in [these slides][]

### Package Installation

The ClearText package can be installed using `pip`. Any required data (including word vectors) will be downloaded on-the-fly at runtime.

```bash
pip install git+https://github.com/bencwallace/cleartext
```

### Training

The `train.py` script in the [scripts][] directory can be used to train a simplification model.
Usage instructions can be printed as follows.

```bash
python -m cleartext.scripts.train --help 
```

Time spent and training/validation losses will be printed at the end of each epoch. When training
completes or if you interrupt training, tests are run and diagnostics are printed.  

It is recommended to execute a ClearText training run from within an [MLflow](https://mlflow.org/) environment;
MLflow will automatically track and log runtime parameters and training progress.
To execute a training run with MLflow, ensure you are in the current directory and use the following command (make sure **not to forget** the dot at the end):

```bash
mlflow run [parameters] .
```

where [parameters] is formatted according to the MLflow CLI [instructions](https://mlflow.org/docs/latest/cli.html#mlflow-run).
For instance, to run for 10 epochs with 100 hidden units, use the following command:

```bash
mlflow run -P num-epochs=10 -P rnn-units=100 .
```

[Forbes]: https://www.forbes.com/sites/susanadams/2019/07/16/game-of-tongues-how-duolingo-built-a-700-million-business-with-its-addictive-language-learning-app/
[TESOL]: https://www.tesol.org/

[notebook]: notebooks/cleartext.ipynb
[scripts]: https://github.com/bencwallace/cleartext/tree/master/scripts
[these slides]: https://docs.google.com/presentation/d/1X-X74s5Db-YFYO9kv7kX1GYn6aSDT_UkXKY-Jlb7cjo/edit?usp=sharing

[1]: https://www.internationalteflacademy.com/blog/report-from-tesol-2-billion-english-learners-worldwide
[2]: https://www.duolingo.com/
