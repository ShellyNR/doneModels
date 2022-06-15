import json

from flask import Flask

be = Flask(__name__)


# both - text and photos endpoint
@be.route('/', methods=['POST'])
def hello():
    with open('dummy.json', 'r') as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    be.run(host='0.0.0.0', port=8000,debug=True)
