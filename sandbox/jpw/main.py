from os import makedirs
import random
from os.path import exists
import pandas as pd
import copy
import numpy as np

import gensim
import sklearn
import matplotlib.pyplot as plt

from nltk import RegexpTokenizer
from nltk.corpus import stopwords

import logging

from sklearn import linear_model
import matplotlib.pyplot as plt

#global variables
g_data_path = '../../data/'
g_output_path = './gen_data/'

makedirs(g_output_path, exist_ok=True)
 # create logger with 'spam_application'
g_logger = logging.getLogger('gensim')
g_logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler(g_output_path + 'd2v.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
g_logger.addHandler(fh)
g_logger.addHandler(ch)


# source data columns
# id,Title,FullDescription,LocationRaw,LocationNormalized,ContractType,ContractTime,
# Company,Category,SalaryRaw,SalaryNormalized,SourceName

g_tokenizer = RegexpTokenizer(r'\w+')
g_stopword_set = set(stopwords.words('english'))

def get_clean_word_list(doc):
    new_str = doc.lower()
    ordered_list = g_tokenizer.tokenize(new_str)
    clipped_list = list(set(ordered_list).difference(g_stopword_set)) # side effect of changing word order
    clipped_ordered_list = [word for word in ordered_list if word in clipped_list]
    return clipped_ordered_list

def get_cleaned_ad_word_list(docs):
    g_logger.info('cleaning ads')
    cleaned_ad = []
    for doc in docs:
        cleaned_ad.append(get_clean_word_list(doc))
    return pd.Series(cleaned_ad, index=docs.index)

def index_to_tag(index):
    return 'index_' + str(index)

def tag_to_index(tag):
    return int(tag[6:])

def add_link_doc_to_id(ads):
    tagged_docs = []
    for ad, i in zip(ads, ads.index):
        tag = index_to_tag(i)
        tagged_docs.append(gensim.models.doc2vec.TaggedDocument(ad, [tag]))
    return tagged_docs
   
def load_or_create_vector_space(vector_size, raw_docs):
    d2vm = None
    vec_space_model_path = g_output_path + 'doc2vec.model'
    vocab_model_path = vec_space_model_path + ".vocab"
    if exists(vec_space_model_path):
        g_logger.info('Loading doc2vec model')
        d2vm = gensim.models.Doc2Vec.load(vec_space_model_path)
    else:
        cleaned_ads = get_cleaned_ad_word_list(raw_docs)
        g_logger.info('we have {:,} samples'.format(len(cleaned_ads)))
        tagged_docs = add_link_doc_to_id(cleaned_ads)
        g_logger.info('we have {:,} samples'.format(len(tagged_docs)))
        
        g_logger.info('Building doc2vec model')
        '''
        if exists(vocab_model_path):
            d2vm = gensim.models.Doc2Vec.load(vocab_model_path)
            g_logger.info('    Loaded vocab model')
        else:
            d2vm = gensim.models.Doc2Vec(vector_size=vector_size, min_count=0, alpha=0.025, min_alpha=0.025)
            d2vm.build_vocab(tagged_docs)
            d2vm.save(vocab_model_path)
            g_logger.info('    Created vocab model')
        d2vm.train(tagged_docs, total_examples=len(tagged_docs), epochs=100)
        g_logger.info('    trained model')
        '''
        d2vm = gensim.models.Doc2Vec(
            tagged_docs, 
            vector_size=vector_size, 
            min_count=0, 
            alpha=0.025,
            min_alpha=0.025,
            workers=8,
            iter=20,
            seed=99
            )
    d2vm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    d2vm.save(vec_space_model_path)
    #d2v_inspection(d2vm, tr)
    return d2vm

def dump_similar_word(d2vm, pos, neg):
    g_logger.info('vectors space trained, most similar too "' + str(pos) + '" while suppresing "' + str(neg) + '" are...')
    for item in d2vm.wv.most_similar(positive=pos, negative=neg, topn=4):
        g_logger.info(item)

def d2v_inspection(d2vm, df):
    dump_similar_word(d2vm,['developer'], None)
    dump_similar_word(d2vm, ['developer'], ['software'])
    dump_similar_word(d2vm, ['developer'], ['property'])

    g_logger.info('word vec len {:,}'.format(len(d2vm.wv.vocab)))
    g_logger.info('doc vec len {:,}'.format(len(d2vm.docvecs)))

    '''
    test_str = 'Software developer'
    g_logger.info('ad similar to "' +test_str+ '" are ...')
    test_vec = d2vm.infer_vector(test_str.split())
    '''
    for i in range(10):
        # pick random ad
        test_ad_index = random.choice(df.index)
        test_row = df.loc[test_ad_index]
        g_logger.info('\n\n')
        g_logger.info('ad similar to "' +test_row.Title+ '" are ...')
        test_vec = d2vm.docvecs[index_to_tag(test_ad_index)]
        for item in d2vm.docvecs.most_similar([test_vec], topn=3):
            tag = item[0]
            likelihood = item[1]
            index = tag_to_index(tag)
            row = df.loc[index]
            g_logger.info(str(likelihood) + '\t' + tag + '\t' + row.Title)

def gen_graph(base_title, target_s, pred_s):
    graph_dir = g_output_path + 'graphs/'+ base_title + '/'
    makedirs(graph_dir , exist_ok=True)
    plt.clf()
    plt.scatter(target_s, pred_s)
    plt.title(base_title + '\n SalaryNormalized')
    plt.xlabel('Actual')
    # plt.xlim(0.0, 1.0)
    plt.ylabel('Prediction')
    # plt.ylim(-1.0, 1.0)
    # plot ideal fit guide line
    plt.plot((0,1), 'r--')
    plt.savefig(graph_dir + base_title)
    plt.show()
    plt.close()

def train_and_evaluate_model(
    model_decsription, model, 
    tr_feature_vectors_series, tr_target, 
    ts_feature_vectors_series, ts_target
):
    g_logger.info('training ' + model_decsription)
    model.fit(tr_feature_vectors_series, tr_target)

    g_logger.info('testing model')
    predictions = model.predict(ts_feature_vectors_series)
    
    g_logger.info('generating results')
    gen_graph(model_decsription, ts_target, predictions)
    #g_logger.info(model_decsription + ' RMSE ' + str(evaluate_RMSE(ts_target_df, predictions_df)))

def extract_doc_vetors(docs, d2vm):
    vecs = pd.DataFrame(index=docs.index)
    for i in docs.index:
        tag = index_to_tag(i)
        row = d2vm.docvecs[tag]
        vecs.loc[i] = pd.Series(row)
    return vecs

def infer_vector_space(docs, d2vm):
    vecs = pd.DataFrame(index=docs.index)
    for index, doc in docs.iteritems():
        clean_word_list = get_clean_word_list(doc)
        row = d2vm.infer_vector(clean_word_list)
        vecs.loc[index] = pd.Series(row)
    return vecs

def get_doc_vectors(tr_docs, ts_docs):
    doc_vec_path = g_output_path + '/doc_vectors/'
    makedirs(doc_vec_path , exist_ok=True)
    tr_dv_path = doc_vec_path + 'training'
    ts_dv_path = doc_vec_path + 'test'
    tr_doc_vecs_df = None
    ts_doc_vecs_df = None

    if not exists(tr_dv_path) or not exists(ts_dv_path):
        vector_size = 100
        d2vm = load_or_create_vector_space(vector_size, tr_docs)
        
        g_logger.info('Extracting training doc vectors')
        tr_doc_vecs_df = extract_doc_vetors(tr_docs, d2vm)
        with open(tr_dv_path, 'w') as tr_f:
            tr_doc_vecs_df.to_csv(tr_f)

        g_logger.info('Infering test doc vectors')
        ts_doc_vecs_df = infer_vector_space(ts_docs, d2vm)
        with open(ts_dv_path, 'w') as ts_f:
            ts_doc_vecs_df.to_csv(ts_f)
    
    g_logger.info('Loading doc vectors')
    tr_doc_vecs_df = pd.read_csv(tr_dv_path)
    ts_doc_vecs_df = pd.read_csv(ts_dv_path)

    return tr_doc_vecs_df, ts_doc_vecs_df

def split_data_frame_train_test(df):
    # for now we want a reproduecable random selection
    np.random.seed(0)
    selection_mask = np.random.rand(len(df)) < 0.85 # 85/15 split
    return df[selection_mask], df[~selection_mask]

def load_data():
    original_filepath = g_data_path + 'Train_rev1.csv'
    tr_filepath = g_data_path + 'Train_split_tr.csv'
    ts_filepath =  g_data_path + 'Train_split_ts.csv'
    if not exists(tr_filepath) or not exists(ts_filepath):
        df = pd.read_csv(original_filepath)
        tmp_tr, tmp_ts = split_data_frame_train_test(df)
        tmp_tr.to_csv(tr_filepath)
        tmp_ts.to_csv(ts_filepath)
    tr = pd.read_csv(tr_filepath)
    ts = pd.read_csv(ts_filepath)
    return tr, ts

def main():
    g_logger.info('Loading source data')
    tr, ts = load_data()
    tr_doc_vecs_df, ts_doc_vecs_df = get_doc_vectors(tr.FullDescription, ts.FullDescription)
    train_and_evaluate_model('linear_regression', linear_model.LinearRegression(), tr_doc_vecs_df, tr.SalaryNormalized.values, ts_doc_vecs_df, ts.SalaryNormalized.values)

# start of main script 
main()
