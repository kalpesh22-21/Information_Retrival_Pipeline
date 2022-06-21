from keybert import KeyBERT
from urllib import response
from flask import Flask,request,session
import os
import json
from collections import Counter

with open("stop.txt") as f:
    data = f.read()
stop_words= json.loads(data)

try:
    kw_model = KeyBERT()
except Exception as e:
    print(e)


app = Flask(__name__)
app.secret_key = 'sas'

@app.route("/query_weights")
def get_weights():
    try:
        if request.args.get('use_weights') != None:
            use_weights = request.args.get('use_weights')
        else:
            use_weights = 'False'
        
        if request.args.get('diversity') != None:
            diversity = float(request.args.get('diversity'))
        else:
            diversity = 0.5

        query = request.args.get('query')
        keywords = kw_model.extract_keywords(query,keyphrase_ngram_range=(1, 3),use_mmr= True,diversity= diversity,top_n= 10,stop_words = stop_words['stopwords'])
        keywords = Counter({key: score for key,score in keywords})

        weights = {
            3:1000,
            2:100,
            1:10
        }

        if use_weights == 'True':
            output = {}
            ordered = keywords.most_common()
            for key,score in ordered:
                size = len(key.split())
                output[key] = weights[size]*score
                weights[size] = weights[size]/2
                
        else:
            output = keywords

        return({'Weights': output})

    except Exception as e:
        print(e)
        return('Failed')



@app.route("/")
def index():
	return "KeyBert Model"


if __name__ == "__main__":
	app.run(debug=True)

