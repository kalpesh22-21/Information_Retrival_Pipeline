import re
from flask import Flask,request,session,jsonify
# from flask_httpauth import HTTPBasicAuth
import pandas as pd
import os
import json
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
import numpy.matlib as mb
import numpy as np
import textdistance
from collections import Counter
import numpy as np
import scipy.sparse as sp
from flask_cors import CORS,cross_origin
from datetime import datetime
import datetime as dt
import random
import requests
import mysql.connector
from mysql.connector.constants import ClientFlag
import jwt
from werkzeug.security import generate_password_hash,check_password_hash
from functools import wraps

class User(object):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = generate_password_hash(password, "sha256")

    def __str__(self):
        return "User(id='%s')" % self.id

users = [
    User(1, 'sas_user', 'abc123')
]

username_table = {u.username: u for u in users}


f = open('sql_config.json','r')
config_file = json.load(f)
f.close()
config = {
    'user': config_file['user'],
    'password': config_file['password'],
    'host': config_file['host'],
    'client_flags': [ClientFlag.SSL],
    'ssl_ca': 'ssl/server-ca.pem',
    'ssl_cert': 'ssl/client-cert.pem',
    'ssl_key': 'ssl/client-key.pem'
}

# now we establish our connection
conn = mysql.connector.connect(**config)
cur = conn.cursor()
cur.execute('use queries;')
courses = pd.read_sql('select * from course_data;',conn)

# Getting Current Directory
cwd = os.getcwd()
trans = 'offline'

# Complete Vocabulary including documents and search keywords
ac_vocab =  set(json.loads(open('autocorrect_vocabulary.txt', 'r').read())['vocabulary'])


## Functions for Autocorrect
## --------------------------------------------------------------------
def distance(input_word,vocabulary_word):
    input_word = list(input_word.keys())[0]
    vocabulary_word = list(vocabulary_word.keys())[0]
    return (1 - (textdistance.Jaccard(qval=1).distance(input_word,vocabulary_word)))


def autocorrect(input_word,vocabulary = ac_vocab):
    input_word = input_word.lower()
    if input_word in vocabulary:
        return(input_word)
    else:
        vector_distance = np.vectorize(distance)
        return(list(vocabulary)[np.argmax(vector_distance({input_word:0},[{x:0} for x in vocabulary]))])
## ---------------------------------------------------------------------


# Functions For Document Retrival
#---------------------------------------------------------------------
def get_articles(tf_idf,features_dict,course_list,kw_dict,kw_tf_idf,keywords):
    # vect = CountVectorizer(ngram_range=(1,3))
    # vect.fit([q.lower()])
    q_weights = session['query_weights']
    # print(q_weights)
    q_weights = match_keywords(q_weights,features_dict,kw_dict,kw_tf_idf,keywords,iteration = session['iteration'])
    # print(q_weights)
    vect = Counter({w:q_weights[w] for w in q_weights.keys() if w in features_dict.keys()})
    vect.update(features_dict)
    q_vec = np.array(list(({k:vect[k] for k in sorted(vect.keys())}).values()),dtype = 'float64')
    scores = tf_idf.dot(q_vec)/(sp.linalg.norm(tf_idf,axis=1)/np.linalg.norm(q_vec,axis=0))
    similar_scores = Counter({i: score for i,score in enumerate(list(scores))})
    all_scores = dict(similar_scores)
    domain_docs_scores = Counter({k: all_scores[k] for k in all_scores.keys() if k in list(course_list['index'])})
    sim_sorted = domain_docs_scores.most_common()
    top_10_docs = dict(list(sim_sorted)[:10])
    tfidf_docs = {}
    for doc_id,score in zip(top_10_docs.keys(),top_10_docs.values()):
        tfidf_docs[course_list[course_list['index'] == doc_id]['Course Code'].values[0]] = score
    if trans == 'online':
        trans_out = {}
        trans_out = Counter({doc.replace('.txt',''): score/100 for doc,score in trans_out.items()})
        tf_idf_out = Counter(tfidf_docs['Documents'])
        tf_idf_out.update(trans_out)
    else:
        tf_idf_out = Counter(tfidf_docs)
    
    final = tf_idf_out.most_common()
    # print(final)
    output = []
    for (code,score),i in zip (final,np.arange(1,len(final)+1)):
        if score > 0:
            course = courses[courses['course_code'] == code]
            details = {}
            details['Name'] = list(course['Name'])[0]
            details['url'] = list(course['url'])[0]
            details['learning_path'] = list(course['learning_path'])[0]
            details['Course_code'] = code
            details['similarity score'] = score
            output.append(details)
    return(output)

#---------------------------------------------------------------------

# Functions for Mapping keywords not in coprus
#---------------------------------------------------------------------
def match_keywords(query,features_dict,kw_dict,kw_tf_idf,keywords,iteration = 0):
    if iteration == 0:
        vec_ = Counter({key : query[key] for key in set(query.keys()).difference(set(features_dict.keys())).intersection(kw_dict.keys())})
        weights  = [query.pop(key) for key in vec_.keys()]
        query = Counter(query)
    else:
        vec_ = Counter({key : query[key] for key in set(query.keys()).intersection(kw_dict.keys())})
        weights  = [query.pop(key) for key in vec_.keys()]
        query = Counter({})
    if len(vec_)>0:
        vec_.update(kw_dict)
        q_vec = np.array(list(({k:vec_[k] for k in sorted(vec_.keys())}).values()),dtype = 'float64')
        scores = kw_tf_idf.dot(q_vec)/(sp.linalg.norm(kw_tf_idf,axis=1)/np.linalg.norm(q_vec,axis=0))
        similar_scores = Counter({i: score for i,score in enumerate(list(scores))})
        sim_sorted = Counter(similar_scores).most_common()
        top_5_topics = dict(list(sim_sorted)[:5])
        if iteration == 0:
            topics = [keywords.iloc[:,0][list(top_5_topics.keys())[0]]]
        else:
            topics = keywords.iloc[:,0][list(top_5_topics.keys())[iteration-1:iteration+1]]
        rank = 1

        for topic in topics:
            if len(topic.split())>2:
                cv = CountVectorizer(ngram_range=(2,2))
                cv.fit([str(topic)])
                words = cv.get_feature_names_out()
                query.update({topic.lower() : float(np.sum(weights)/ (len(words)*rank)) for topic in words})
            else:
                query.update({topic.lower(): np.sum(weights)/rank})
            rank = rank*2
    else:
        query = Counter(query)
    # print(query)
    return(query)
#---------------------------------------------------------------------



# File Loadind Functions
#---------------------------------------------------------------------
def load_files():
    tf_idf = sp.load_npz(os.path.join(cwd,'tf_idf.npz'))
    kw_tf_idf = sp.load_npz(os.path.join(cwd,'kw_tf_idf.npz'))
    with open(os.path.join(cwd,'features.txt')) as f:
        data = f.read()
    features_dict = json.loads(data)
    with open(os.path.join(cwd,'KW_features.txt')) as f:
        data = f.read()
    kw_dict = json.loads(data)
    return(tf_idf,features_dict,kw_dict,kw_tf_idf)

def load_courses(directory):
    os.chdir(directory)
    courses = pd.read_csv('course_list.csv')
    return(courses)

#---------------------------------------------------------------------


# Get things ready
# Keywords extracted from documents
keywords = pd.read_csv(os.path.join(cwd,'keywords.csv'))
# Loading tfidf and vocabulary of Documents and the search keywords.
tf_idf,features_dict,kw_dict,kw_tf_idf = load_files()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret'
app.secret_key = 'super-secret'
CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
# print(app.config['CORS_HEADERS'])

def token_required(f):
    @wraps(f)
    def token_dec(*args, **kwargs):
        token = request.args.get('token')
        if not token:
            return "Missing Token!"
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except Exception as e:
            print(e)
            return "Invalid Token"
        return f(*args, **kwargs)
    return token_dec



@app.route("/get_token")
# @cross_origin(supports_credentials= True)
def verify_password():
    username = request.args.get('username')
    password = request.args.get('password')
    user = username_table.get(username, None)
    try:
        if user and check_password_hash(user.password, password):
            token = jwt.encode({'user':username, 'exp': dt.datetime.utcnow() + dt.timedelta(minutes = 30)},app.config['SECRET_KEY'])
            return jsonify({"access_token":token})
        else:
            return(401)
    except Exception as e:
         print(e)


@app.route("/get_meta")
@token_required
@cross_origin(supports_credentials= True)
# @jwt_required()
def get_meta():
    if "spec" in session:
        spec = session['spec']
    else:
        return("No Specialization") 

    # Checking Level       
    if "level" in session:
        level = session['level']
    else:
        return("No Level")
    
    if level == 'All':
        directory = os.path.join(cwd,str(spec))
        courses = load_courses(directory)
        return {'Selected Concentration' : spec,'Selected Level': level, 'Dir': directory,'Number_of_Courses': len(courses)}
    else:
        # Beginner Level Selected
        directory = os.path.join(cwd,str(spec),str(level))
        courses = load_courses(directory)
        return {'Selected Concentration' : spec,'Selected Level': level, 'Dir': directory,'Number_of_Courses': len(courses)}



@app.route("/get_documents",methods = ["POST","GET"])
@token_required
@cross_origin(supports_credentials= True)
# @jwt_required()
def get_documents():
    if request.method == "GET":
        out = {}

        if 'sessionid' not in session:
            session['sessionid'] = 'search' + str(datetime.now()) + str(random.randint(200000,5000000000))
            session['sql_inserted'] = False

        if request.args.get('reset'):
            session['sessionid'] = session['sessionid'] + str(random.randint(1,9))
            session['sql_inserted'] = False
            session.pop("iteration")


        query = request.args.get('query')
        print('\n')
        print("Input Query: {0}".format(query))
        corrected_query = ' '.join(list(map(lambda x: autocorrect(x),query.split())))
        # print("Corrected Query: {0}".format(corrected_query))
        # print('\n')
        session['query'] = corrected_query
        s = requests.session()
        response = s.get("https://keybert2-vmnfeny4ra-uc.a.run.app/query_weights",params={'query': session['query'],'use_weights' : True, 'diversity' : 1})
        session['query_weights'] = json.loads(response.content)['Weights']       

        session['spec'] = request.args.get('spec')
        session['level'] = request.args.get('level')

        if request.args.get('iteration'):
            session['iteration'] = request.args.get('iteration')


        if "query" in session:
            query = session['query']
        else:
            return("No Search Query Specified")

        # Checking Specialization
        if "spec" in session:
            spec = session['spec']
        else:
            return("No Specialization") 

        # Checking Level       
        if "level" in session:
            level = session['level']
        else:
            return("No Level")         
        
        if level == 'All':
            directory = os.path.join(cwd,str(spec))
        else:
            directory = os.path.join(cwd,str(spec),str(level))

        # Search Iteration
        if "iteration" in session:
            session["iteration"] += 1
        else:
            session["iteration"] = 0

        # Inserting query in Database
        if session['sql_inserted'] == False:
            session['sql_inserted'] = True
            try:
                cur.execute('Insert into query values(%s, %s,%s,%s);',(session['sessionid'],session['query'],session['spec'],session['level']))
            except Exception as e:
                print(e)
            conn.commit()

        # Document Retrival
        if query != None:
            try:
                course_list = load_courses(directory)
                output = get_articles(tf_idf,features_dict,course_list,kw_dict,kw_tf_idf,keywords)
                session['docs'] = output
                docs = str({elem['Course_code'] for elem in session['docs']})
                try:
                    cur.execute('Insert into search_results values(%s, %s,%s);',(session['sessionid'],docs,session['iteration']))
                except Exception as e:
                    print(e)
                conn.commit()
                return ({'Selected Concentration' : spec,'Selected Level': level , 'query' : query, 'Documents' : output , 'iteration' : session["iteration"] , 'sessionid' : session['sessionid']})
            except Exception as E:
                return(E)
    else:
        return("Method Error")

@app.route("/put")
@token_required
@cross_origin()
# @jwt_required()
def put_queries():
    out ={}
    session['iteration'] = request.args.get('iteration')
    session['sessionid'] = request.args.get('sessionid')
    response = request.args.get('response')
    cur.execute('Insert into responses values(%s, %s,%s);',(session['sessionid'],session['iteration'],response))
    if request.args.get('reset') == 1:
        session.clear()
        out['session'] = 'Session Updated'
    else:
        session.pop('spec')
        session.pop('level')

    return('Updated')

@app.route("/")
def index():
	return "API for Returning Courses"

if __name__ == "__main__":
	app.run(debug=True)
