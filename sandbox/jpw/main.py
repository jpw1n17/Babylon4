from os import makedirs
from os.path import exists
import pandas as pd
import copy

import gensim
import sklearn
import matplotlib.pyplot as plt

from nltk import RegexpTokenizer
from nltk.corpus import stopwords

#global variables
g_data_path = '../../data/'
g_output_path = './gen_data/'

# source data columns
# id,Title,FullDescription,LocationRaw,LocationNormalized,ContractType,ContractTime,
# Company,Category,SalaryRaw,SalaryNormalized,SourceName

# a simple class for holding all state and passing it around
# for now we will just have a dataframe and dynamically add other properties.
# later we may choose to formalise the class 
class AppState(object):
    def __init__(self, data_frame):
        self.df = data_frame

def add_cleaned_ad(state):
    print('cleaning ads')
    tokenizer = RegexpTokenizer(r'\w+')
    stopword_set = set(stopwords.words('english'))
    state.cleaned_ad = []
    for d in state.df.FullDescription:
        new_str = d.lower()
        ordered_list = tokenizer.tokenize(new_str)
        clipped_list = list(set(ordered_list).difference(stopword_set)) # side effect of changing word order
        clipped_ordered_list = [word for word in ordered_list if word in clipped_list]
        state.cleaned_ad.append(clipped_ordered_list)

def add_link_doc_to_id(state):
    state.lookup = {}
    state.lookup_raw = {}
    state.tagged_docs = []
    for doc, raw, id in zip(state.cleaned_ad, state.df.FullDescription, state.df.Id):
        state.lookup[id]=doc
        state.lookup_raw[id]=raw
        state.tagged_docs.append(gensim.models.doc2vec.TaggedDocument(doc, [id]))
   
def load_or_create_vector_space(state):
    vec_space_model_path = g_output_path + 'doc2vec.model'
    vocab_model_path = vec_space_model_path + ".vocab"
    if exists(vec_space_model_path):
        print('Loading doc2vec model')
        state.d2vm = gensim.models.Doc2Vec.load(vec_space_model_path)
    else:
        print('Building doc2vec model')
        if exists(vocab_model_path):
            state.d2vm = gensim.models.Doc2Vec.load(vocab_model_path)
            print('    Loaded vocab model')
        else:
            state.d2vm = gensim.models.Doc2Vec(vector_size=state.vector_size, min_count=0, alpha=0.025, min_alpha=0.025)
            state.d2vm.build_vocab(state.tagged_docs)
            state.d2vm.save(vocab_model_path)
            print('    Created vocab model')
        
        state.d2vm.train(state.tagged_docs, total_examples=len(state.tagged_docs), epochs=100)
        print('    trained model')
        state.d2vm.save(vec_space_model_path)

def d2v_inspection(state):
    print('vectors space trained, most similar too "strom" are...')
    for item in state.d2vm.wv.most_similar(positive=['storm'], topn=4):
        print(item)

    print('\nvectors space trained, most similar too "storm" while suppresing "wind" are...')
    for item in state.d2vm.wv.most_similar(positive=['storm'], negative=['wind'], topn=4):
        print(item)

    print('\nvectors space trained, most similar too "storm" while suppresing "rain" are...')
    for item in state.d2vm.wv.most_similar(positive=['storm'], negative=['rain'], topn=4):
        print(item)

    test_str = 'looks like another sunny day tomorrow'
    print('\n\ad similar to "' +test_str+ '" are ...')
    test_vec = state.d2vm.infer_vector(test_str.split())
    for item in state.d2vm.docvecs.most_similar([test_vec], topn=3):
        ad_id = item[0]
        likelihood = item[1]
        ad_text = state.lookup_raw[ad_id]
        print(str(likelihood) + '\t' + ad_text)

def main():
    makedirs(g_output_path, exist_ok=True)

    print('Loading source data')
    state = AppState(pd.read_csv(g_data_path + 'Train_rev1.csv'))
    state.vector_size = 100
    add_cleaned_ad(state)
    add_link_doc_to_id(state)
    load_or_create_vector_space(state)
    d2v_inspection(state)

# start of main script 
main()
