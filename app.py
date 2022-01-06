import time

import numpy as np
from flask import Flask
from flask_restful import Resource, Api, reqparse
from gensim.models import FastText, KeyedVectors

app = Flask(__name__)
api = Api(app)

start = time.time()
FastText_embedding = KeyedVectors.load_word2vec_format("./input/wiki.nl.vec")
end = time.time()
print("Loading the embedding took %d seconds" % (end - start))


def handle_out_of_vocab(embedding, in_txt):
    out = None
    for word in in_txt:
        try:
            tmp = embedding[word]
            tmp = tmp.reshape(1, len(tmp))

            if out is None:
                out = tmp
            else:
                out = np.concatenate((out, tmp), axis=0)
        except:
            pass

    return out


def assemble_embedding_vectors(embedding, sentence):
    out = None

    tmp = handle_out_of_vocab(embedding, sentence)
    if tmp is not None:
        dim = tmp.shape[1]
        if out is not None:
            vec = np.mean(tmp, axis=0)
            vec = vec.reshape((1, dim))
            out = np.concatenate((out, vec), axis=0)
        else:
            out = np.mean(tmp, axis=0).reshape((1, dim))
    else:
        pass

    return out


@app.route("/")
def hello_world():
    return "<p>Hello, World! How are you doing?</p>"


class ModelCheck(Resource):

    @staticmethod
    def get():
        return {'embedding_vector_size': FastText_embedding.vector_size}


class SimilarWords(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('w1', type=str, required=True, help="Word cannot be blank!")
        parser.add_argument('w2', type=str, required=True, help="Word 2 cannot be blank!")
        args = parser.parse_args()

        return {
            'w1': args['w1'],
            'w2': args['w2'],
            'found_similarity': float(FastText_embedding.similarity(args['w1'], args['w2']))
        }


class MostSimilarWord(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('word', type=str, required=True, help="Word cannot be blank!")
        args = parser.parse_args()

        most_similar_key = FastText_embedding.similar_by_word(args['word'])

        return {
            'word': args['word'],
            'found_word': most_similar_key
        }


class ClosestConcept(Resource):

    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('word', type=str, required=True, help="Word cannot be blank!")
        args = parser.parse_args()

        most_similar_key = FastText_embedding.most_similar_to_given(args['word'],
                                                                    [
                                                                        'voeding',
                                                                        'transport',
                                                                        'persoon',
                                                                        'dier',
                                                                        'gebouw',
                                                                        'sport',
                                                                        'detailhandel'
                                                                    ])

        return {
            'word': args['word'],
            'found_word': most_similar_key
        }


class FindAverageEmbedding(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()
        parser.add_argument('sentence', type=str, required=True, help="We need a sentence to do something!")
        args = parser.parse_args()

        words = args['sentence'].split(" ")
        value = assemble_embedding_vectors(FastText_embedding, words)

        return {
            'sentence': args['sentence'],
            'shape_sentence': value.shape
        }


api.add_resource(ModelCheck, '/model')
api.add_resource(SimilarWords, '/similar')
api.add_resource(FindAverageEmbedding, '/embedding')
api.add_resource(MostSimilarWord, '/most_similar')
api.add_resource(ClosestConcept, '/concept')

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
