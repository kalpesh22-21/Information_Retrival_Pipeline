import os
import json
from fastapi import FastAPI
from google.cloud import storage
from haystack.nodes import FARMReader, TransformersReader
# In-Memory Document Store
from haystack.document_stores import InMemoryDocumentStore,ElasticsearchDocumentStore
from haystack.nodes import TfidfRetriever
from haystack.pipelines import ExtractiveQAPipeline
# from flask import Flask, request, jsonify
# from flask_restful import Api,Resource

# Instantiate a Google Cloud Storage client 
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_key.json'
storage_client = storage.Client()
bucket_name = 'sas_txt_files'

#Specify required bucket and file
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob('data.json')

# Download the contents of the blob as a string and then parse it using json.loads() method
data = json.loads(blob.download_as_string(client=None))

document_store = InMemoryDocumentStore()
# document_store = ElasticsearchDocumentStore( host = 'localhost',username = '',password = '')
document_store.write_documents(data)

# retriever = ElasticsearchRetriever(document_store=document_store)
retriever = TfidfRetriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/bert-base-cased-squad2", use_gpu=False)
pipe = ExtractiveQAPipeline(reader, retriever)


# Fast Api
app = FastAPI()

@app.get('/documents')
async def documents(query):
        prediction = pipe.run(
            query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 10}})
        docs = list(map(lambda x: x.meta,prediction['answers']))
        scores = list(map(lambda x: x.score,prediction['answers']))
        return ({doc['name']:score for doc,score in zip(docs,scores)})


# class Transformer_Prediction(Resource):
#     def get(self,query):
#         prediction = pipe.run(
#             query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 1}})
#         return (list(map(lambda x: x.meta,prediction['documents'])))





### Flask - Not Working In parallel Processing
# app = Flask(__name__)
# api = Api(app)

# api.add_resource(Transformer_Prediction,"/<string:query>")

# # @app.route("/", methods=["GET", "POST"])
# # def index():
# #     if request.method == "POST":
# #         query = request.query
# #         file = request.files.get('file')
# #         if file is None or file.filename == "":
# #             return jsonify({"error": "no file"})

# #         try:
# #             image_bytes = file.read()
# #             pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
# #             tensor = transform_image(pillow_img)
# #             prediction = predict(tensor)
# #             data = {"prediction": int(prediction)}
# #             return jsonify(data)
# #         except Exception as e:
# #             return jsonify({"error": str(e)})

# #     return "OK"


# if __name__ == "__main__":
#     app.run(debug=True)