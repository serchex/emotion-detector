import pandas as pd
import pre, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

def bert(nombre,archivo):
    modelo = SentenceTransformer('distilbert-base-nli-mean-tokens')
    #modelo = SentenceTransformer('mrm8488/distilroberta-finetuned-emotion') ingles + español mixed
    # "j-hartmann/emotion-english-distilroberta-base" english optimizated
    
    df = pd.read_csv(f'{archivo}',sep=';',header=None, names=['texto','emocion'])
    df['texto'] = df['texto'].astype(str)
    df['emocion'] = df['emocion'].astype(str)
    print(df)
    df['texto_limpio'] = df['texto'].apply(pre.clean_text)
    print(df.head())

    embeddings = modelo.encode(df['texto_limpio'])

    print(f'Shape de los embeddings {embeddings.shape}')
    print(embeddings[0])

    le = LabelEncoder()
    y = le.fit_transform(df['emocion'])

    df = pd.DataFrame({
        'frase':embeddings.tolist(),
        'labels':y
    })

    df.to_pickle(f'{nombre}.pkl')

def sentence(frase):
    modelo = SentenceTransformer('distilbert-base-nli-mean-tokens')
    #modelo = SentenceTransformer('mrm8488/distilroberta-finetuned-emotion') ingles + español mixed
    # "j-hartmann/emotion-english-distilroberta-base" english optimizated
    txt = pre.clean_text(frase)

    embeddings = modelo.encode(txt)

    return embeddings