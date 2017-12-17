from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps

app = Flask(__name__)
api = Api(app)




from sklearn.externals import joblib
from classifier import tokenize
from cleaner import clean

clf = joblib.load('filename.pkl')
vectorizer = joblib.load('vectorizer.pkl')

input = ["你见天怎么样", "但当它的同伴按照这只黑猩猩的示 意走过去时，那只撒谎的黑猩猩却朝真正有香蕉的地方跑"]





import numpy as np


class Prediction(Resource):
    def get(self, chinese):

        input = [chinese, '']
        input = clean(input)
        prepare = vectorizer.transform(clean(input))
        pred = clf.predict(prepare)
        level = np.asscalar(np.int16(pred[0]))

        result = {
            chinese: level
        }
        return result


api.add_resource(Prediction, '/HSKLevel/<string:chinese>')

if __name__ == '__main__':
    app.run()










