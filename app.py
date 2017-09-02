from flask import Flask, render_template, request, redirect

import pandas as pd   
from bs4 import BeautifulSoup
import re
import string
#from spacy.en import STOPWORDS <-- gives error
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from spacy.en import English
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from string import punctuation
import dill


def remove_human_names(text):
    text = nlp(unicode(text))
    for w in text:
        try:
            if w.ent_type_ == 'PERSON':
                text = re.sub(unicode(w), ' ', unicode(text))
        except:
            text = text
    return unicode(text).strip()

def remove_punct_unless_aphostrophe(text):
    # print punctuation.replace("'", '') --> !"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~
    puncts = punctuation.replace("'", '')
    return ' '.join(re.split(r'[!"#$%&()*+,-./:;<=>?@\[\\\]^_`{|}~]+', text))


def lemmatize(text, do_not_lemm):
    #lemmatized = []
    #for w in nlp(unicode(text)):
    #    if str(w) in do_not_lemm:
    #        lemmatized.append(unicode(w))
    #    else:
    #        lemmatized.append(w.lemma_)
    return ' '.join([unicode(w) if (str(w) in do_not_lemm) else w.lemma_ for w in nlp(unicode(text))])

# Clean data set
def clean_sentences(text, do_not_lemm, STOPWORDS):
    
    text = BeautifulSoup(text, "lxml").get_text()       # Delete html

    text = re.sub('http.*:\/\/[^\s]*', ' ', text)
            
    text = re.sub('\\\\n', ' ', text)                         # Delete \n
    text = re.sub('\\\\xa0', ' ', text)                       # Delete \xa0
    text = re.sub('\\\\[a-z0-9]*', '', text)                  # Delete other things like \x0l2
    text = re.sub('@[a-z0-9A-Z]*', '', text)                  # Delete tweeter user names
    text = re.sub('$[0-9]*', '', text)                        # Delete dollar ammounts
    text = re.sub('[0-9]*%', '', text)                        # Delete percentages
    
    text = re.sub('[0-9]*', '', text)                         # Delete numbers
   
    text = remove_punct_unless_aphostrophe(text)
    

    text = re.sub(" '", ' ', text)                            # Delete ' for cases like   are 'friendly' but
    text = re.sub("' ", ' ', text)                            # Delete ' for cases like   are 'friendly' but

    #text = remove_human_names(text)                           # Remove human names
        
    text = ' '+text.lower()                                   # Change text to lower case
    
    text = re.sub(" don't ", ' do not ', text)                # Manually normalize text...
    text = re.sub(" won't ", ' will not ', text)
    text = re.sub(" doesn't ", ' does not ', text)
    text = re.sub(" can't ", ' cannot ', text)
    text = re.sub(" ain't ", ' is not ', text)
    text = re.sub(" wouldn't ", ' would not ', text)
    text = re.sub(" shouldn't ", ' should not ', text)
    text = re.sub(" couldn't ", ' could not ', text)
    text = re.sub(" isn't ", ' is not ', text)
    text = re.sub(" aren't ", ' are not ', text)
    text = re.sub(" wasn't ", ' was not ', text)
    text = re.sub(" weren't ", ' were not ', text)

    text = re.sub("'re ", ' are ', text)
    text = re.sub("'ve ", ' have ', text)
    text = re.sub("'d ", ' would ', text)
    text = re.sub("'s ", ' is ', text)       # (pero' cosi' ignora il genitivo sassone...)
    
    text = re.sub('id iot', 'idiot', text)
    text = text.replace('PieceOfShit','piece'+' '+'of'+' '+'shit')    
    
    text = re.sub(' u ', ' you ', text)
    text = re.sub(' em ', ' them ', text)
    text = re.sub(' da ', ' the ', text)
    text = re.sub(' yo ', ' you ', text)
    text = re.sub(' ya ', ' you ', text)
    text = re.sub(' ur ', ' you are ', text)

    text = re.sub(" doesnt ", ' does not ', text)
    text = re.sub(" dont ", ' do not ', text)
    text = re.sub(" im ", ' i am ', text)
    text = re.sub(' aint ', ' is not ', text)
    text = re.sub(' ill ', ' i will ', text)
    text = re.sub(' id ', ' i would ', text)
    text = re.sub(' ive ', ' i have ', text)
    text = re.sub(' wasnt ', ' was not ', text)
    text = re.sub(' werent ', ' were not ', text)
    text = re.sub(' your a ', ' you are a ', text)
    text = re.sub(' your such a ', ' you are a ', text)
    text = re.sub(' your an ', ' you are an ', text)
    text = re.sub(' your such an ', ' you are an ', text)
    
    text = re.sub(' has been ', ' was ', text)               # the lemmatizing does not recognise "has been" as "be"
    text = re.sub(' have been ', ' were ', text)
    text = re.sub(' fuck off ', ' fuck_off ', text)          # otherwise 'off' rientra nelle stopwords
    
    text = re.sub(" f'ing ", ' fucking ', text)              # otherwise 'off' rientra nelle stopwords
    text = re.sub(" fuckin ", ' fucking ', text)
    text = re.sub(" freaking ", ' fucking ', text)

    
    words = text.split()                                     # Split text in words    
    stops = STOPWORDS.difference(set(['you', 'your', 'yourself', 'be', 'are']))  
    text = ' '.join([w for w in words if not w in stops])    # Keep only words that are not stop_words
    
   #text = ' '.join(w.lemma_ for w in nlp(unicode(text)))    # Transform words in their lemmas
   #text = lemmatize(text, do_not_lemm)
    
    if text.strip()=='' or len(text.strip())<3:              # Sometimes only characters are left from cleaning
        text = None
    else:
        text = ' '.join([w for w in text.split()])           # for whatever reason lemma_ adds spaces... (?)

    return text


def model(text):
   do_not_lemm = ['ass', 'fucking'] # list all words that you do not want to lemmatize
   STOPWORDS = set(stopwords.words("english"))
   tfidf_transformer = dill.load(open('tfidf_transformer.dill', 'r'))
   tfidf_transformer = dill.load(open('tfidf_transformer_fit.dill', 'r'))
   clf = dill.load(open('clf.dill', 'r'))
   cleaned_test_s = clean_sentences(text, do_not_lemm, STOPWORDS)   
   test_s_tfidf = tfidf_transformer.transform(pd.Series(cleaned_test_s))
   predicted = clf.predict(test_s_tfidf)
   return predicted


app = Flask(__name__)

app.vars={}

@app.route('/')
def main():
  return redirect('/index')

@app.route('/index')
def index():
  return render_template('index.html')



# Custom functions:

@app.route('/gettext', methods=['POST'])
def gettext():
    if request.method == 'POST':
        #text = request.form['text']
        #f = open('text.txt', 'w')
        #f.write(text)
        #f.close()
        text = request.form['text']
	yesno = model(text)[0]
        if yesno==0:
           pred='non insulting!'
        else:
           pred='insulting!'
	return render_template('results.html', content=text, prediction=pred, probability=0)
    return 1


@app.route('/thanks', methods=['POST'])
def feedback():
	return render_template('thanks.html')

if __name__ == '__main__':
  app.run(port=33507)
