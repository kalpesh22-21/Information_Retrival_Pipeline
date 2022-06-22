# Information_Retrival_Pipeline
An Information Retrival pipeline for searching relevant documents Using Customized Weighted TFIDF and Haystack Question Answering Model using Bert Base model integrated with Elastic Search for fast document retrival
The module consists of 3 API's
1) Main IR API (/main.py)
2) Haystack_deployment API (/Haystak_Deployment/main.py)
3) Keywords Extractor API (/Keyword_extractor/main.py)

![Pipeline](/assets/pipeline.png)

## All API's were pushed to cloudrun and employed in a Front-End Chatbot Interface

![chatbot](/assets/chatbot.png)

# Keywords Extractor API
This is the achine Learning component in the Information Retrival Pipeline as shown above which extracts the most important keywords in the search query so that the downward pipeline will put higher cosine similarity score associated with this keywords as shown in the image below.
![Pipeline](/assets/2.png)

#### We used the Transformer model called as [KeyBert] (https://github.com/MaartenGr/KeyBERT/) for the above task and we used "Bert-base" fine tuned on the documents using Masking.


# Haystack Deployment API

Haystack [https://haystack.deepset.ai/overview/intro] is a QA retrival algorithm for Information retrival. The Retriver can be a traditional TFIDF or Elastic Search we used elastic search for increased speed and the document were about 10000 words large.
![Pipeline](/assets/Picture3.png)

#### "deepset/bert-base-cased-squad2" was used as the Reader in ExtractiveQA Pipeline. The model was fine tuned on the frequently asked queries and answers to the question in relevant documents.



# Main API ( Includes Weighted TFIDF, Spell-checker & Keyword matcher)
The customized TFIDF was trained on the documents using the structure of the document it explained in the other repository (https://github.com/kalpesh22-21/Weighted_TFIDF).

The customized TFIDF provides more relevant documents to be recommended as the inherent structure of Heading and Appendix has more information about the topic
![Pipeline](/assets/Picture1.png)


# Final Solution
![Pipeline](/assets/final.png)
