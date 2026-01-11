# Emotion Detection with BERT + CatBoost

This project is an **emotion detection system** that uses **BERT embeddings** combined with a **CatBoost classifier** to predict emotions from text.  
It supports **both Spanish and English** thanks to an integrated translation step.

## Visual Interaction In Streamlit
<img width="818" height="398" alt="Image" src="https://github.com/user-attachments/assets/1cb6a95a-f973-43f8-94fa-3833308355fb" />

**CLICK HERE**
-https://emotion-detector-sergio-gonzalez.streamlit.app

##  Features
- Preprocessing with text cleaning and stopwords removal.
- BERT-based embeddings (`distilbert-base-nli-mean-tokens`).
- CatBoost classifier for emotion prediction.
- Automatic translation from Spanish to English.
- Web interface using **Streamlit**.

##  Tech Stack
- **Python 3.8+**
- [Sentence-Transformers](https://www.sbert.net/) for BERT embeddings
- [CatBoost](https://catboost.ai/) for classification
- [Deep Translator](https://pypi.org/project/deep-translator/) for automatic translation
- [Streamlit](https://streamlit.io/) for the web interface

##  Installation
1. Clone this repository:
```bash
git clone https://github.com/serchex/emotion-detector.git
cd emotion-detector
pip install -r requirements.txt

Install NLTK resources:
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

 Training the Model
Run the embedding generation and training pipeline:
    python bert.py
    python catboost_train.py

This will create:

train_embeddings.pkl

catboost_emociones.cbm

label_encoder.pkl

 Making Predictions (CLI)
You can make predictions directly from the terminal:

python predicts.py
Example output:

Input: "Estoy muy feliz de verte"
Translated: "I am very happy to see you"
Predicted emotion: happy
 Running the Streamlit App
Launch the interactive web interface:

streamlit run app.py
Then open the local URL shown in your terminal.

 Visualization

You can visualize classification results using:
Bar charts of predicted emotion counts.
t-SNE scatter plots of embeddings clustered by predicted emotion.

 License
This project is licensed under the MIT License.

 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.
