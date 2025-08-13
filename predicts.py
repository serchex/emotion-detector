from catboost import CatBoostClassifier
import pickle, numpy as np
from bert import sentence
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from deep_translator import GoogleTranslator

# Traductor
def traductor(texto):
    return GoogleTranslator(source='auto', target='en').translate(texto)

df_original = pd.read_csv('train.txt',sep=';', header=None, names=['texto','emocion'])
le = LabelEncoder()
le.fit(df_original['emocion'])

model = CatBoostClassifier()
model.load_model('catboost_emociones.cbm')

s = input(f'Type your sentence: ')
txt = sentence(traductor(s))
txt = np.array(txt).reshape(1,-1)

pred = model.predict(txt)
pred_class = int(pred.flatten()[0])
pred_text = le.inverse_transform([pred_class])[0]

print(pred_text)