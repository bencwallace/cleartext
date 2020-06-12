from flask import Flask, request

from .. import PROJ_ROOT
from ..pipeline import Pipeline


BEAM_SIZE = 6
MAX_LEN = 50

MODELS_ROOT = PROJ_ROOT / 'models'
MODEL_DIR = MODELS_ROOT / '10-11'

pl = Pipeline.deserialize(MODEL_DIR)

app = Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    data = request.json['text']

    tokens = pl.src.preprocess(data)
    output = pl.beam_search(tokens, BEAM_SIZE, MAX_LEN)
    result = ' '.join(output)

    return result
