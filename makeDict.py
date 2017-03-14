import gensim
import os
import pickle

model = gensim.models.KeyedVectors.load_word2vec_format('word2vec.bin',binary = True)
classes = os.listdir('caltech97Train')
'''
inDict = [category in model.vocab for category in classes]

for category in classes:
    if category not in model.vocab:
        print category

classes = os.listdir('caltech97Val')

inDict = [category in model.vocab for category in classes]

for category in classes:
    if category not in model.vocab:
        print category

'''
labels = {category:model.word_vec(category) for category in classes}

f = open('caltech95Dict.pkl','wb')
pickle.dump(labels,f)
f.close()
