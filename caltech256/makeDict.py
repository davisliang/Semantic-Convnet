import gensim
import os
import pickle

model = gensim.models.KeyedVectors.load_word2vec_format('../word2vec.bin',binary = True)
classes = os.listdir('resizedCaltech250Train')
classes.sort()
print(classes)
inDict = [category in model.vocab for category in classes]

labels = {category:model.word_vec(category) for category in classes}

f = open('caltech256Dict.pkl','wb')
pickle.dump(labels,f)
f.close()
