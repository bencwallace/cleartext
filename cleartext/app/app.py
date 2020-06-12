from flask import Flask, request


app = Flask(__name__)


def process(data):
    return data


@app.route('/', methods=['POST'])
def main():
    data = request.get_data()
    result = process(data)
    return result
