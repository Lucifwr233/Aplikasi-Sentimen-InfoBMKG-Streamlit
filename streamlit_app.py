import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import joblib
import nltk
import ast
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
# Import library untuk metrik evaluasi dan visualisasi
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- Konfigurasi Awal dan Download NLTK ---
nltk_resources = ['punkt', 'punkt_tab']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# --- DEFINISI KELAS MODEL ---
class ImprovedNaiveBayesClassifier:
    def __init__(self):
        self.tf, self.class_probabilities, self.vocabulary = {}, {}, []
    def predict(self, text: str) -> str:
        words = text.split()
        scores = {}
        vocab_size = len(self.vocabulary)
        for class_label, class_prob in self.class_probabilities.items():
            scores[class_label] = np.log(class_prob)
            total_words = sum(self.tf[class_label].values())
            for word in words:
                if word:
                    word_count = self.tf[class_label].get(word, 0)
                    word_prob = (word_count + 1) / (total_words + vocab_size)
                    scores[class_label] += np.log(word_prob)
        return max(scores.items(), key=lambda x: x[1])[0]

class SVMClassifier:
    def __init__(self):
        self.weights, self.bias, self.vocabulary = None, None, None
    def text_to_vector(self, text: str) -> list[float]:
        if self.vocabulary is None: raise ValueError("Vocabulary belum ada.")
        words = text.split()
        word_counts = {word: words.count(word) for word in set(words)}
        return [word_counts.get(word, 0) for word in self.vocabulary]
    def predict(self, features: list[float]) -> int:
        if self.weights is None: raise ValueError("Model belum dilatih.")
        prediction = np.dot(features, self.weights) + self.bias
        if prediction > 0.5: return 1
        elif prediction < -0.5: return -1
        else: return 0

class KNNClassifier:
    def __init__(self, k=3):
        self.k, self.documents, self.labels = k, [], []
    def vectorize(self, text: str) -> dict[str, int]:
        words = text.split()
        return {word: words.count(word) for word in set(words)}
    def euclidean_distance(self, v1: dict, v2: dict) -> float:
        all_words = set(list(v1.keys()) + list(v2.keys()))
        return np.sqrt(sum((v1.get(w,0) - v2.get(w,0))**2 for w in all_words))
    def predict(self, text: str) -> str:
        vectorized = self.vectorize(text)
        distances = sorted([{'d': self.euclidean_distance(vectorized, doc), 'l': self.labels[i]} for i, doc in enumerate(self.documents)], key=lambda x: x['d'])
        labels = [n['l'] for n in distances[:self.k]]
        return max(set(labels), key=labels.count)

# --- FUNGSI-FUNGSI BANTUAN ---
@st.cache_resource
def get_stemmer():
    return StemmerFactory().create_stemmer()

def preprocess_text(komentar: str) -> str:
    stemmer = get_stemmer()
    stopwords = {'yang', 'dan', 'di', 'ke', 'dari', 'dengan', 'ini', 'itu', 'pada', 'untuk'}
    if pd.isna(komentar) or komentar.strip() == "": return ""
    komentar = komentar.lower()
    komentar = re.sub(r'[^a-z\s]', '', komentar)
    komentar = re.sub(r'\s+', ' ', komentar).strip()
    kata = word_tokenize(komentar)
    kata = [word for word in kata if word not in stopwords]
    return ' '.join([stemmer.stem(word) for word in kata])

@st.cache_resource
def load_models():
    try:
        nb_model = joblib.load('Model/naive_bayes_custom_model.pkl')
        svm_model = SVMClassifier()
        with open('Model/svm_custom_model_multi.json', 'r') as f:
            svm_data = json.load(f)
        svm_model.weights, svm_model.bias, svm_model.vocabulary = np.array(svm_data['weights']), svm_data['bias'], svm_data['vocabulary']
        knn_model = joblib.load('Model/knn_model.pkl')
        return nb_model, svm_model, knn_model
    except FileNotFoundError as e:
        st.error(f"Error: File model tidak ditemukan: {e.filename}")
        return None, None, None

@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Komentar_Tokenized' in df.columns:
            df['Komentar_Tokenized'] = df['Komentar_Tokenized'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' tidak ditemukan.")
        return None

# --- FUNGSI UNTUK VISUALISASI ---
def calculate_metrics(y_true, y_pred):
    return {
        "Akurasi": accuracy_score(y_true, y_pred),
        "Presisi": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

def plot_sentiment_barchart(data, column, title, ax):
    counts = data[column].value_counts().sort_index()
    color_map = {'positif': '#4CAF50', 'negatif': '#F44336', 'netral': '#2196F3'}
    colors = [color_map.get(sentiment, '#999999') for sentiment in counts.index]
    counts.plot(kind='bar', color=colors, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel('Jumlah')
    ax.tick_params(axis='x', rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + (counts.max()*0.01), str(v), ha='center')

def plot_wordcloud(df, sentiment, ax):
    filtered_df = df[df['Label'] == sentiment]
    if filtered_df.empty:
        ax.text(0.5, 0.5, f"Tidak ada data '{sentiment}'", ha='center', va='center')
        ax.set_axis_off()
        return
    all_words = [word for tokens in filtered_df['Komentar_Tokenized'] for word in tokens]
    if not all_words:
        ax.text(0.5, 0.5, f"Tidak ada kata '{sentiment}'", ha='center', va='center')
        ax.set_axis_off()
        return
    color_map = {'positif': 'Greens', 'negatif': 'Reds', 'netral': 'Blues'}
    wordcloud = WordCloud(width=400, height=200, background_color='white', colormap=color_map.get(sentiment, 'viridis')).generate_from_frequencies(Counter(all_words))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f"Sentimen: {sentiment.capitalize()}", fontsize=14)
    ax.set_axis_off()

def plot_confusion_matrix(y_true, y_pred, title, ax):
    labels = ['positif', 'negatif', 'netral']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Label Aktual')
    ax.set_xlabel('Label Prediksi')

def plot_roc_curve(y_true, y_pred, model_name, ax):
    lb = LabelBinarizer()
    y_true_bin, y_pred_bin = lb.fit_transform(y_true), lb.transform(y_pred)
    classes, colors = lb.classes_, ['#4CAF50', '#F44336', '#2196F3']
    for i, color in enumerate(colors):
        if i < len(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}'); ax.legend(loc="lower right")

# --- ANTARMUKA STREAMLIT ---
st.set_page_config(page_title="Analisis Sentimen BMKG", layout="wide")
st.title("📱 Analisis Sentimen Ulasan Aplikasi Info BMKG")
st.markdown("Masukkan teks ulasan untuk menganalisis sentimennya menggunakan tiga model berbeda.")

nb_model, svm_model, knn_model = load_models()

user_input = st.text_area("Masukkan Teks Ulasan di Sini:", height=150, placeholder="Contoh: Aplikasinya sangat bagus dan informatif!")

if st.button("Analisis Sentimen", type="primary"):
    if user_input and all([nb_model, svm_model, knn_model]):
        preprocessed_input = preprocess_text(user_input)
        st.subheader("Hasil Prediksi")
        cols = st.columns(3)
        models = {'Naive Bayes': nb_model, 'SVM': svm_model, 'KNN': knn_model}
        for col, (name, model) in zip(cols, models.items()):
            with col:
                st.markdown(f"#### {name}")
                with st.spinner("Memprediksi..."):
                    if name == 'SVM':
                        vector = model.text_to_vector(preprocessed_input)
                        pred_int = model.predict(vector)
                        pred = {1: 'positif', -1: 'negatif', 0: 'netral'}[pred_int]
                    elif name == 'KNN':
                         clean_input = re.sub(r'[^a-z\s]', '', user_input.lower()).strip()
                         pred = model.predict(clean_input)
                    else: # Naive Bayes
                        pred = model.predict(preprocessed_input)
                    if pred == 'positif': st.success("👍 Sentimen: Positif")
                    elif pred == 'negatif': st.error("👎 Sentimen: Negatif")
                    else: st.warning("😐 Sentimen: Netral")
    elif not user_input:
        st.warning("Mohon masukkan teks ulasan terlebih dahulu.")

st.markdown("---")

with st.expander("Tampilkan Dataset Asli"):
    if st.button("Muat Dataset"):
        df_asli = load_data('Dataset/1 Dataset Asli.csv')
        if df_asli is not None:
            st.dataframe(df_asli)

st.subheader("Visualisasi dan Analisis Data")
if st.button("Tampilkan Visualisasi & Evaluasi"):
    df_eval = load_data('Dataset/6 hasil_gabungan_prediksi.csv')
    
    if df_eval is not None:
        y_true = df_eval['Label']
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribusi Sentimen", "☁️ Word Cloud", "📈 Metrik Performa", "📉 Kurva ROC (AUC)"])

        with tab1:
            st.markdown("##### Perbandingan Distribusi Sentimen Aktual vs Prediksi")
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            plot_sentiment_barchart(df_eval, 'Label', 'Label Aktual', axes[0])
            plot_sentiment_barchart(df_eval, 'Prediksi_NaiveBayes', 'Prediksi Naive Bayes', axes[1])
            plot_sentiment_barchart(df_eval, 'Prediksi_SVM', 'Prediksi SVM', axes[2])
            plot_sentiment_barchart(df_eval, 'Prediksi_KNN', 'Prediksi KNN', axes[3])
            st.pyplot(fig)

        with tab2:
            st.markdown("##### Word Cloud Berdasarkan Sentimen Aktual")
            fig_wc, axes_wc = plt.subplots(1, 3, figsize=(18, 6))
            plot_wordcloud(df_eval, 'positif', axes_wc[0])
            plot_wordcloud(df_eval, 'negatif', axes_wc[1])
            plot_wordcloud(df_eval, 'netral', axes_wc[2])
            st.pyplot(fig_wc)

        with tab3:
            st.markdown("##### Perbandingan Metrik Performa")
            metrics_nb = calculate_metrics(y_true, df_eval['Prediksi_NaiveBayes'])
            metrics_svm = calculate_metrics(y_true, df_eval['Prediksi_SVM'])
            metrics_knn = calculate_metrics(y_true, df_eval['Prediksi_KNN'])
            metrics_df = pd.DataFrame([metrics_nb, metrics_svm, metrics_knn], index=['Naive Bayes', 'SVM', 'KNN'])
            st.table(metrics_df.style.format("{:.2%}"))
            
            st.markdown("##### Confusion Matrix")
            fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
            plot_confusion_matrix(y_true, df_eval['Prediksi_NaiveBayes'], "Naive Bayes", axes_cm[0])
            plot_confusion_matrix(y_true, df_eval['Prediksi_SVM'], "SVM", axes_cm[1])
            plot_confusion_matrix(y_true, df_eval['Prediksi_KNN'], "KNN", axes_cm[2])
            st.pyplot(fig_cm)

        with tab4:
            st.markdown("##### Kurva ROC (Receiver Operating Characteristic)")
            fig_roc, axes_roc = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
            plot_roc_curve(y_true, df_eval['Prediksi_NaiveBayes'], 'Naive Bayes', axes_roc[0])
            plot_roc_curve(y_true, df_eval['Prediksi_SVM'], 'SVM', axes_roc[1])
            plot_roc_curve(y_true, df_eval['Prediksi_KNN'], 'KNN', axes_roc[2])
            st.pyplot(fig_roc)

st.info("Aplikasi ini dibuat untuk mendemonstrasikan perbandingan model klasifikasi teks.")