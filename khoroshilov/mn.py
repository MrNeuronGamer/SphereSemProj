from smart_open import smart_open
import os
import re
import gensim.downloader as api
import pandas as pd

from gensim.matutils import softcossim
from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression


from engines import train_model_on_group, predict_on_group, load_doc, score_on_group


word2vec_model300 = api.load('word2vec-ruscorpora-300')


train_data = pd.read_csv('train_groups.csv')
test_data = pd.read_csv('test_groups.csv')
train_data = train_data[train_data.group_id < 5]

print('ok')
train_data['content'] = train_data.doc_id.apply(load_doc)
test_data['content'] = train_data.doc_id.apply(load_doc)
print('ok')

Model = LogisticRegression()

print('ok')

for i in train_data.group_id.unique():
    print(i)
    Model = train_model_on_group(
        Model, train_data[train_data.group_id == i], word2vec_model300)


print('ready almost')


for i in test_data.group_id.unique()[0:1]:
    print(i)
    prediction = predict_on_group(Model, test_data[test_data.group_id == i], word2vec_model300)

for i in test_data.group_id.unique()[1:]:
    print(i)
    prediction = prediction.append([ predict_on_group(Model, test_data[test_data.group_id == i], word2vec_model300)])
    


prediction.to_csv("results_ddt.csv", index=False)