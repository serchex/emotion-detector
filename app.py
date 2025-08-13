import streamlit as st
import numpy as np
from catboost import CatBoostClassifier
from deep_translator import GoogleTranslator
from bert import sentence
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model = CatBoostClassifier()
model.load_model('catboost_emociones.cbm')

df_original = pd.read_csv('train.txt',sep=';', header=None, names=['texto','emocion'])
le = LabelEncoder()
le.fit(df_original['emocion'])

def traductor(texto):
    return GoogleTranslator(source='auto', target='en').translate(texto)

st.title('Detector de emociones 😠🥺🥰')
st.write('Escribe una frase y te dire que emocion detecta el modelo.')

frase_user = st.text_area('Escribe tu frase aqui:', '')

if st.button('Detectar emocion'):
    if frase_user.strip():
        frase_traducida = traductor(frase_user)
        txt_embeddings = sentence(traductor(frase_traducida))
        txt_embeddings = np.array(txt_embeddings).reshape(1,-1)

        pred = model.predict(txt_embeddings)
        pred_class = int(pred.flatten()[0])
        emocion = le.inverse_transform([pred_class])[0]

        st.success(f'Emocion detectada: **{emocion}**')
        st.caption(f'(Texto traducido: {traductor(frase_traducida)})')
    else:
        st.warning('Por favor escribe una frase antes de detectar')
