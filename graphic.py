import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import plotly.express as px
import pickle
from sklearn.preprocessing import LabelEncoder

model = CatBoostClassifier()
model.load_model('catboost_emociones.cbm')

df_test = pd.read_pickle('test_embeddings.pkl')

x_test = np.vstack(df_test['frase'].values)
y_test = df_test['labels'].values

df_original = pd.read_csv('train.txt',sep=';', header=None, names=['texto','emocion'])
le = LabelEncoder()
le.fit(df_original['emocion'])

y_pred = model.predict(x_test).flatten().astype(int)
y_pred_text = le.inverse_transform(y_pred)

df_counts = pd.DataFrame({'emocion':y_pred_text})
conteo = df_counts['emocion'].value_counts().reset_index()
conteo.columns = ['emocion', 'cantidad']

fig = px.bar(
    conteo,
    x='emocion',
    y='cantidad',
    title='Clasificación de emociones en el set de test',
    text='cantidad',
    color='emocion'
)

fig.show()