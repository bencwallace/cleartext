<img src="chrome/icon128.png" style="float: right;">

# ClearText


Leveraging natural language processing and deep learning technology to help English language learners on the road to
fluency.

## Quick Start

For installation and usage instructions, refer to the [official ClearText extension page][extension].

## What problem does ClearText address?

### The *English as a Second Language* Market

According to [TESOL][tesol] (Teachers of English to Speakers of Other Languages), there are over
[1.5 billion][tesol-stats]
English language learners worldwide. A huge amount of human labour is involved in educating these learners. However,
many learners may not be able to afford lesson costs and must resort to other methods.

There is a large market for assisted language learning applications. For instance, [Forbes][forbes]
reports [Babbel's][babbel] revenue at \$115 million and [Duolingo's][duolingo] 2017 valuation at \$700
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

## Developing Simplification Models with ClearText

ClearText uses a sequence-to-sequence model trained on the WikiSmall/WikiLarge datasets. For more on these datasets,
take a look at the [notebook][]. A high-level overview of the development of ClearText can be found in
[these slides][slides].

There are two ways of training simplification models with ClearText.
In both cases, time spent and training/validation losses will be printed at the end of each epoch.
When training completes or if you interrupt training, tests are run and diagnostics are printed.

### Installing and Running ClearText as a Package

The ClearText package can be installed using `pip`. Any required data (including word vectors) will be downloaded
*on-the-fly* at runtime.

```bash
pip install git+https://github.com/bencwallace/cleartext
```

After installing ClearText, instructions for training a simplification model can be printed with the following command.

```bash
python -m cleartext.scripts.train --help 
```

### Running ClearText with MLflow

Running ClearText with [MLflow][mlflow] not only takes care of preparing and
isolating your environment, but has the additional advantage of automatically logging training progress and metadata
using [MLflow tracking][tracking].

To train with MLflow, first install MLflow, either using pip (`pip install mlflow`) or conda
(`conda install -c conda-forge mlflow`) and then run the following command
```bash
mlflow run [options] https://github.com/bencwallace/cleartext
```
where `[options]` is a sequence of options taking the form `-P parameter=[value]`.
For instance, to train for 10 epochs with 100 hidden units, use the following command:

```bash
mlflow run -P num-epochs=10 -P rnn-units=100 https://github.com/bencwallace/cleartext
```

For a list of available options, run
```bash
mlflow run -e help https://github.com/bencwallace/cleartext
```

## Repository Structure

This repository is divided into the following directories:

* chrome: Source for the ClearText Chrome extension
* cleartext: ClearText package
  * app: Main entrypoint for inference (Flask application).
  * data: Data loading modules.
  * models: PyTorch models.
  * pipeline: End-to-end pipeline (data loading and preprocessing, model training, inference, and evaluation).
  * scripts: Main entrypoints for training and evaluation.
  * utils: Miscellaneous utilities.
* data: Placeholder into which ClearText will save downloaded datasets.
* models: Placeholder into which ClearText will serialize models.
* notebooks: Jupyter notebooks for EDA.
* tests: Unit tests.
* vectors: Placeholder into which ClearText will save downloaded word vectors.

[babbel]: https://www.babbel.com/
[duolingo]: https://www.duolingo.com/
[extension]: https://bcwallace.com/cleartext
[forbes]: https://www.forbes.com/sites/susanadams/2019/07/16/game-of-tongues-how-duolingo-built-a-700-million-business-with-its-addictive-language-learning-app/
[mlflow]: https://mlflow.org/
[notebook]: notebooks/cleartext.ipynb
[scripts]: https://github.com/bencwallace/cleartext/tree/master/scripts
[slides]: https://docs.google.com/presentation/d/1X-X74s5Db-YFYO9kv7kX1GYn6aSDT_UkXKY-Jlb7cjo/edit?usp=sharing
[tracking]: https://mlflow.org/docs/latest/tracking.html
[tesol]: https://www.tesol.org/
[tesol-stats]: https://www.internationalteflacademy.com/blog/report-from-tesol-2-billion-english-learners-worldwide
