from gensim.matutils import softcossim
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel

import pandas as pd
import numpy as np
import os

import math


def take_S(tar_list, ind):
    return tar_list[ind]


def make_dist_vec(vector, doc_vectors, similarity_matrix, length=5):
    """
        fuck off!! 
        Read code instead of searching for docstring
    """
    gen_vector = []

    for i in doc_vectors:
        if vector == i:
            continue

        gen_vector = gen_vector + [softcossim(vector, i, similarity_matrix)]

    gen_vector.sort(reverse=True)

    return gen_vector[:length+1]


def train_model_on_group(model, docs_data, word2vec_model300, length=5):
    """ 
    Parameters:

        model -- model object with methods train and predict

        docs_data -- pandas Data Frame with fields pair_id, content, target
        word2vec_model300 -- w2v model (object)


    Returns:

         model trained on data


    """

    dictionary = corpora.Dictionary()
    for i in docs_data.content:
        try:
            dictionary.add_documents([i])
        except:
            dictionary.add_documents([['a']])

    docs_data['vector'] = docs_data.content.apply(doc_opti, args=(dictionary,))

    corpus = []
    for line in docs_data.content:
        try:
            if math.isnan(line): line  = ["мимо"]
        except:
            pass
        corpus = corpus +[dictionary.doc2bow(line)]

    similarity_matrix = word2vec_model300.similarity_matrix(dictionary, tfidf=TfidfModel(corpus,
        dictionary=dictionary), threshold=0.0, exponent=2.0, nonzero_limit=100)

    docs_data['dist_vec'] = docs_data.vector.apply(
        make_dist_vec, args=(docs_data.vector, similarity_matrix))

    features = [str(i) for i in range(length)]

    
    for i in range(length):
        docs_data[str(i)] = docs_data.dist_vec.apply(take_S, args=(i,))

    print(docs_data.head())

    
    model = model.fit(
        docs_data[features], docs_data['target'])

    print(model.score(docs_data[features], docs_data['target']))

    return model


def predict_on_group(model, docs_data, word2vec_model300, length=5) -> 'pd.DataFrame of type :  pair_id  || target':
    """ 
    Parameters:

        model -- model object with methods train and predict

        docs_data -- pandas Data Frame with fields pair_id, content, target
        word2vec_model300 -- w2v model (object)


    Returns:

       pd.DataFrame of type : { pair_id  || target }   with predicted target for each pair_id

    """

    dictionary = corpora.Dictionary()
    for i in docs_data.content:
        try:
            dictionary.add_documents([i])
        except:
            dictionary.add_documents([['a']])

    docs_data['vector'] = docs_data.content.apply(doc_opti, args=(dictionary,))
    # except:
    #     docs_data['vector'] = docs_data.content.apply(dictionary.doc2bow)

    corpus = []
    for line in docs_data.content:
        try:
            if math.isnan(line): line  = ["мимо"]
        except:
            pass
        corpus = corpus +[dictionary.doc2bow(line)]



    similarity_matrix = word2vec_model300.similarity_matrix(dictionary, tfidf=TfidfModel(corpus,
        dictionary=dictionary), threshold=0.0, exponent=2.0, nonzero_limit=100)

    docs_data['dist_vec'] = docs_data.vector.apply(
        make_dist_vec, args=(docs_data.vector, similarity_matrix))

    features = [str(i) for i in range(length)]

    
    for i in range(length):
        docs_data[str(i)] = docs_data.dist_vec.apply(take_S, args=(i,))


    docs_data['target'] = model.predict(
        np.array(docs_data[features]))

    return docs_data[['pair_id', 'target']]


def score_on_group(model, docs_data, word2vec_model300, length=5) -> 'score':
    """ 
    Parameters:

        model -- model object with methods train and predict

        docs_data -- pandas Data Frame with fields pair_id, content, target
        word2vec_model300 -- w2v model (object)


    Returns:

       pd.DataFrame of type : { pair_id  || target }   with predicted target for each pair_id

    """

    dictionary = corpora.Dictionary()
    for i in docs_data.content:
        try:
            dictionary.add_documents([i])
        except:
            dictionary.add_documents([['a']])

    docs_data['vector'] = docs_data.content.apply(doc_opti, args=(dictionary,))

    corpus = []
    for line in docs_data.content:
        try:
            if math.isnan(line): line  = ["мимо"]
        except:
            pass
        corpus = corpus +[dictionary.doc2bow(line)]

    similarity_matrix = word2vec_model300.similarity_matrix(dictionary, tfidf=TfidfModel(corpus,
        dictionary=dictionary), threshold=0.0, exponent=2.0, nonzero_limit=100)

    docs_data['dist_vec'] = docs_data.vector.apply(
        make_dist_vec, args=(docs_data.vector, similarity_matrix))

    features = [str(i) for i in range(length)]

    
    for i in range(length):
        docs_data[str(i)] = docs_data.dist_vec.apply(take_S, args=(i,))

    score = model.score(
        np.array(docs_data[features]), np.array(docs_data['target']))

    return score


def load_doc(doc_id):
    path = os.path.join('', 'parsed', str(doc_id)+".bd")
    file = open(path, 'r')
    content = file.read()[:500]
    if len(content)  < 5  :
         print(doc_id)
         content = "молоко"
    file.close()
    content = str(content).split()

   

    return content

def doc_opti(doc,dictionary):
    print("here goes doc: " , doc, sep = "  ")

    try:
        if math.isnan(doc): doc  = ["мимо"]
    except:
        pass
    result = dictionary.doc2bow(doc)

    return result