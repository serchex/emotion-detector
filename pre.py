import pandas as pd
import re
import string
import nltk as nl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nl.download('stopwords')
nl.download('punkt')
nl.download('punkt_tab')
nl.download('wordnet')


def clean_text(texto):
    

    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE) #Quitar URLs
    texto = re.sub(r'@\w+|#\w+', '',texto) #Quitar menciones o #
    texto = texto.translate(str.maketrans('','',string.punctuation)) # Quitar signos
    texto = re.sub(r'\d+','',texto) # Quitar numeros
    tokens = nl.word_tokenize(texto) # Separar oraciones en palabras (list)
    #print(tokens)
    
    stop_words = set(stopwords.words('english'))
    tokens = [pal for pal in tokens if pal not in stop_words] # Quitar conexiones de palabras

    lemmatizer = WordNetLemmatizer() 
    tokens = [lemmatizer.lemmatize(pal) for pal in tokens] # LLevar una palabra a su forma base

    texto_limpio = ' '.join(tokens)

    return texto_limpio
    #print(texto_limpio)

