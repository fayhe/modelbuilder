
# coding: utf-8

# In[33]:

from __future__ import unicode_literals
from sklearn.externals import joblib
from textacy import fileio, preprocess, Doc, vsm
import numpy as np
import json

# In[34]:

import os
import time
from flask import Flask
from flask_restful import reqparse, Api, Resource


# In[35]:

#load model
selected_model = 2

if(selected_model == 0):
    #svm model
    msg_model = joblib.load('/home/pretrain-model/category.pkl')
    id2term_itr = fileio.read_json('/home/pretrain-model/category-id2term')
    label2id_itr = fileio.read_json('/home/pretrain-model/category-label2id')
elif(selected_model == 1):
    #ann classifier
    msg_model = joblib.load('/home/pretrain-model/category-ann.pkl')
    id2term_itr = fileio.read_json('/home/pretrain-model/category-id2term-ann')
    label2id_itr = fileio.read_json('/home/pretrain-model/category-label2id-ann')
elif(selected_model == 2):
    msg_model = joblib.load('/home/pretrain-model/category-ann-reg.pkl')
    id2term_itr = fileio.read_json('/home/pretrain-model/category-id2term-ann-reg')
    label2id_itr = fileio.read_json('/home/pretrain-model/category-label2id-ann-reg')
else:
    print("failed to find model file")
    
id2term = [i for i in id2term_itr][0]
label2id = [i for i in label2id_itr][0]


# In[36]:

app = Flask("Category API")
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('text')


# In[43]:

def calcCategory(sample_msg):
    sample_doc = Doc(sample_msg)
    terms_lists_test = sample_doc.to_terms_list(ngrams={2, 3}, named_entities=True, as_strings=True)
    
    if(selected_model == 0):
        dtm_test, i2t_test = vsm.doc_term_matrix([terms_lists_test], weighting='tfidf', normalize=True, smooth_idf=True, max_n_terms=5000)
    else:
        dtm_test, i2t_test = vsm.doc_term_matrix([terms_lists_test], weighting='tf', normalize=True, max_n_terms=5000)
    #build new index
    test2train = dict()
    for (i, j) in i2t_test.items():
        if j in id2term.values():
            train_i = id2term.keys()[id2term.values().index(j)]
            test2train[i]=train_i
        
    #map features
    test_feature = np.zeros((dtm_test.shape[0], len(id2term)))
    dtm_test_array = dtm_test.toarray()

    for (i, j) in test2train.items():
        test_feature[:, j] = dtm_test_array[:, i]
    
    if(selected_model == 0):
        result_array = map(lambda x: label2id.keys()[label2id.values().index(x)], msg_model.predict(test_feature))
        result = dict({'category':result_array[0]})
    elif(selected_model == 1):
        result_index = msg_model.predict(test_feature)
        max_index = result_index.argmax()
        msg_category = [i for (i, j) in label2id.items() if j == max_index]
        result = dict({'category':msg_category[0]})
    else:
        result = dict(zip(label2id.keys(), msg_model.predict(test_feature)[0]))
    return result


# In[ ]:

class Category(Resource):

    def get(self):
        return 201
        
    def post(self):
        t0 = time.time()
        args = parser.parse_args()
        if not args.get('text'):
            return {'message': 'Text not found!', 'error': True}, 500

        sample_category = calcCategory(args.get('text'))
        
        return {
            'result': [
                json.dumps(sample_category)
                ],
            'performance': time.time() - t0,
            'version': '0.1.0'
        }, 201

api.add_resource(Category, '/api')


# In[ ]:

if __name__ == '__main__':
    app.run(host="0.0.0.0")


# In[45]:

#sample_msg = u'These were the evening news coverage of the BPLRT disruption on Monday. Mar 30.There was a train disruption along Bukit Panjang LRT again on Mon at around 6.45pm, the second time in three weeks, and passengers had to walk along the tracks again. Wanbao reported that this was the 10th train disruption in the last few months.Some passengers onboard the defective train commented that the train moved intermittently before coming to a halt. In-train announcements were barely audible and waited for 50 minutes before service staff detrained them to Bukit Panjang station.'
#sample_msg = u'Referring to the letters, "Circle Line train incident: SMRT explains cause and steps taken" and "Commuters distressed and kept in the dark", forum writer Larry Chong Tuck Lai asked several questions regarding the safety of passengers when there is smoke inside the train. These are the questions, assuming a fully packed peak-hour train: How much smoke is needed before the smoke detector is activated and the ventilation system shut? What happens after the ventilation system is shut? How long will the air supply last in a crowded train with no air-conditioning or ventilation and with the presence of smoke in the cabin? While the incident was resolved within five minutes, what happens if we have a case where staff are unable to reach the site of the breakdown for hours, and the system is not restored? What options are available to passengers trapped in the train? Are there manual systems to allow passengers to free themselves, or at least open windows to allow air to flow in?CIC/CMC is working with SMRT Trains to prepare a response.'


# In[46]:

#calcCategory(sample_msg)

