import html

from flask import Flask, request
from spacy.lang.en import English

from .. import PROJ_ROOT
from ..pipeline import Pipeline


BEAM_SIZE = 6
MAX_LEN = 50

MODELS_ROOT = PROJ_ROOT / 'models'
MODEL_DIR = MODELS_ROOT / 'jun-17-fixed'

nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))

pl = Pipeline.deserialize(MODEL_DIR)

app = Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    data = request.json['text']

    results = []
    for sentence in nlp(data).sents:
        sentence = str(sentence)
        capitalized = sentence[0].isupper()
        tokens = pl.src.preprocess(sentence)

        output = pl.beam_search(tokens, BEAM_SIZE, MAX_LEN)
        has_period = output[-1] == '.'
        if has_period:
            result = ' '.join(output[:-1]).strip()
        else:
            result = ' '.join(output).strip()
        result += '.'

        if capitalized:
            result = result.capitalize()
        results.append(result)

    results = ' '.join(results)
    return html.escape(results)
