import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

df_cargado = pd.read_pickle('train_embeddings.pkl')
x_train = np.vstack(df_cargado['frase'].values)
y_train = df_cargado['labels'].values

df_test = pd.read_pickle('test_embeddings.pkl')
x_test = np.vstack(df_cargado['frase'].values)
y_test = df_cargado['labels'].values

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=100,
    random_seed=42
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('Reporte de clasificacion:\n', classification_report(y_test,y_pred))
print('Matriz de confusion:\n', confusion_matrix(y_test,y_pred))
model.save_model('catboost_emociones.cbm')