# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import emoji
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from streamlit_option_menu import option_menu

# Page setup
st.set_page_config(page_title="Analisis Sentimen Webtoon", layout="wide")

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Upload dataset .csv", type="csv")
    selected = option_menu(None, ["Beranda", "Preview Data", "Wordcloud", "Distribusi Sentimen", "Panjang Review", "Evaluasi Model"],
                           icons=["house", "table", "cloud", "bar-chart", "bar-chart", "check-circle"], default_index=0)

# Preprocessing
normalization_dict = {
        'gk': 'gak', 'ga': 'gak', 'ngga': 'tidak', 'bgt': 'banget', 'btw': 'ngomong-ngomong',
        'pdhl': 'padahal', 'udh': 'sudah', 'dr': 'dari', 'tp': 'tapi', 'jg': 'juga',
        'sm': 'sama', 'klo': 'kalau', 'tdk': 'tidak', 'trs': 'terus', 'aja': 'saja',
        'blm': 'belum', 'sy': 'saya', 'km': 'kamu', 'org': 'orang', 'ny': 'nya',
        'yg': 'yang', 'bbrp': 'beberapa', 'dgn': 'dengan', 'ntr': 'nanti', 'bgini': 'begini',
        'bsk': 'besok', 'kyk': 'kayak', 'trs': 'terus', 'dlm': 'dalam', 'krn': 'karena',
        'kl': 'kalau', 'jd': 'jadi', "super": "sangat", "tp": "tapi", "jd": "jadi", 'loh': "", "kalo": "kalau", "dr": "dari", "gw": "saya", "gimana": "bagaimana", "pick up": "jemput", "girlfriend": "pacar","bgt": "banget", "sm": "sama", "dgn": "dengan", "yg": "yang", "sy": "saya", "p": "", "admin": "administrator", "skolahan": "sekolah", "tu": "itu", "admin": "administrator", "sih": "", "yg": "yang", "dgn": "dengan", "si": "", "ai": "saya", "udh": "sudah", "oke": "baik", "gw": "saya", "bgt": "banget", "bersih": "bersih", "sy": "saya",
        "kalo": "kalau", "banget": "sangat", "dr": "dari", "tp": "tapi", "ga": "tidak", "asik": "menyenangkan", "banget": "sangat",
        "skolahan": "sekolah", "jd": "jadi", "loh": "", "sm": "sama", "gimana": "bagaimana", "jg": "juga",
        "gua": "saya", "gw": "saya", "lu": "kamu", "loe": "kamu",
        "gak": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
        "aja": "saja", "doang": "saja", "aja": "saja",
        "banget": "sangat", "bgt": "sangat",
        "udah": "sudah", "dah": "sudah", "udh": "sudah",
        "lg": "lagi",  "mo": "mau",
        "vibe": "suasana",
        "gini": "begini", "gitu": "begitu",
        "kerasa": "terasa",
        "emang": "memang", "emg": "memang",
        "gimana": "bagaimana", "gmn": "bagaimana",
        "kmrn": "kemarin", "skg": "sekarang", "skrg": "sekarang",
        "kyk": "seperti", "kayak": "seperti",
        "yg": "yang", "dgn": "dengan", "utk": "untuk",
        "dr": "dari", "kmn": "kemana", "dmn": "dimana", "gf": "girlfriend",
        "byk": "banyak", "bnyk": "banyak",
        "jd": "jadi", "jgn": "jangan", "blm": "belum",
        "sgt": "sangat", "bgt": "sangat",
        "pgen": "ingin", "pengen": "ingin",
        "gpp": "tidak apa-apa", "gapapa": "tidak apa-apa",
        "trs": "terus", "trus": "terus",
        "bisa": "dapat", "bs": "bisa",
        "tpi": "tapi", "tp": "tapi",
        "klo": "kalau", "kalo": "kalau", "daebak": "keren",
        "org": "orang", "orng": "orang",
        "jg": "juga", "aja": "saja",
        "gtu": "begitu", "ngga": "tidak",
        "nih": "ini", "tuh": "itu",
        "liat": "lihat", "ngeliat": "melihat",
        "tar": "nanti", "ntr": "nanti",
        "gw": "saya", "ane": "saya", "uptodate": "terbaru",  "lu": "kamu", "elo": "kamu",
        "cuma": "hanya", "cuman": "hanya",
        "dpt": "dapat", "dapet": "dapat",
        "pake": "pakai", "pke": "pakai",
        "ampe": "sampai",
        "like": "suka",
        "it": "ini",
        "comfortable": "nyaman",
        "yu": "ayo",  "gada": "tidak ada", "temen": "teman", "malem": "malam",
        "kalo": "kalau", "gapapa": "tidak apa-apa", "banget": "sangat",
        "gitu": "begitu", "gini": "begini", "ngampus": "ke kampus",
        "ngerti": "mengerti", "ngasih": "memberi", "nyaman": "nyaman",
        "dsni": "disini", "trus": "terus", "dket": "dekat",
        "bgt": "sangat", "sgt": "sangat", "aja": "saja",
        "kmrn": "kemarin", "tp": "tapi", "yg": "yang",
        "krn": "karena", "dlu": "dulu", "skrg": "sekarang",
        "kyk": "seperti", "emg": "memang", "bnr": "benar",
        "hrs": "harus", "jd": "jadi", "bs": "bisa",
        "tgl": "tanggal", "byk": "banyak", "sma": "sama",
        "heree": "here",
        'dlmnya': "dalamnya",
        "ngafe": "minum kopi",
        "hitz": "populer",
        "knp": "kenapa",
        "ngebead": "membuat",
        "nyasangat": "sangat",
        "dtg": "datang", "yuk": "ayo", "gk": "tidak",
        "menonton": "menjaga",
        "recommend": "rekomendasi",
        "recomended": "rekomendasi",
        "bangettt": "banget",
        "bangettt": "sangat",
        "banget": "sangat", "nya": "", "bagu": "bagus"
    }
factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())

def preprocess(text):
    text = text.lower()
    text = emoji.replace_emoji(text, '')
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([normalization_dict.get(word, word) for word in text.split()])
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['content'] = df['content'].astype(str).fillna('')
    df['preprocessed'] = df['content'].apply(preprocess)
    df['panjang_content'] = df['content'].apply(len)

    # Sementara label awal dari score (untuk dapatkan bobot kata dominan)
    df['sentimen_temp'] = df['score'].apply(lambda x: 1 if x > 3 else 0)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['preprocessed'])
    feature_names = vectorizer.get_feature_names_out()

    # Bobot per kata untuk positif dan negatif
    pos_idx = np.where(df['sentimen_temp'] == 1)[0]
    neg_idx = np.where(df['sentimen_temp'] == 0)[0]
    pos_sum = tfidf_matrix[pos_idx].sum(axis=0).A1
    neg_sum = tfidf_matrix[neg_idx].sum(axis=0).A1
    dominant_sentiment = (pos_sum > neg_sum).astype(int)
    word_sentiment = dict(zip(feature_names, dominant_sentiment))

    # Labeling ulang tiap review berdasarkan bobot kata dominan
    def label_by_word_weight(text):
        tokens = text.split()
        sent_weights = [word_sentiment.get(word, -1) for word in tokens if word in word_sentiment]
        if not sent_weights:
            return -1  # netral/abu jika tidak ada kata yang bisa digunakan
        return 1 if sent_weights.count(1) > sent_weights.count(0) else 0

    df['sentimen'] = df['preprocessed'].apply(label_by_word_weight)
    df = df[df['sentimen'] != -1]  # buang yang tidak bisa dilabeli

    X = vectorizer.transform(df['preprocessed'])
    y = df['sentimen']

    # ======================== UI ========================
    if selected == "Beranda":
        st.title("Analisis Sentimen Review Webtoon")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Review", len(df))
        col2.metric("Review Positif", (df['sentimen'] == 1).sum())
        col3.metric("Review Negatif", (df['sentimen'] == 0).sum())

    elif selected == "Preview Data":
        st.subheader("Preview Data")
        df_tampil = df.copy()
        df_tampil['sentimen_label'] = df_tampil['sentimen'].map({1: 'Positif', 0: 'Negatif'})
        st.dataframe(df_tampil[['content', 'score', 'preprocessed', 'sentimen_label']], use_container_width=True)


    elif selected == "Wordcloud":
        st.subheader("Wordcloud Sentimen Positif & Negatif")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Positif**")
            text_pos = ' '.join(df[df['sentimen'] == 1]['preprocessed'])
            wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(text_pos)
            st.image(wc_pos.to_array())
        with col2:
            st.markdown("**Negatif**")
            text_neg = ' '.join(df[df['sentimen'] == 0]['preprocessed'])
            wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(text_neg)
            st.image(wc_neg.to_array())

    elif selected == "Distribusi Sentimen":
        st.subheader("Distribusi Sentimen")
        count = df['sentimen'].value_counts().sort_index()
        labels = ['Negatif', 'Positif']
        pie = px.pie(names=labels, values=count.values,
                     color=labels,
                     color_discrete_map={'Positif': '#4CAF50', 'Negatif': '#F44336'}, hole=0.4)
        bar = px.bar(x=labels, y=count.values,
                     color=labels,
                     color_discrete_map={'Positif': '#4CAF50', 'Negatif': '#F44336'},
                     labels={'x': 'Sentimen', 'y': 'Jumlah Review'})
        st.plotly_chart(pie, use_container_width=True)
        st.plotly_chart(bar, use_container_width=True)

    elif selected == "Panjang Review":
        st.subheader("Distribusi Panjang Review")
        hist = px.histogram(df, x='panjang_content', nbins=30, title="Histogram Panjang Review")
        df_box = df.copy()
        df_box['sentimen_label'] = df_box['sentimen'].map({1: 'Positif', 0: 'Negatif'})
        box = px.box(df_box, x='sentimen_label', y='panjang_content',
                    color='sentimen_label',
                    color_discrete_map={'Positif': '#4CAF50', 'Negatif': '#F44336'},
                    labels={'sentimen_label': 'Sentimen', 'panjang_content': 'Panjang Review'})

        st.plotly_chart(hist, use_container_width=True)
        st.plotly_chart(box, use_container_width=True)

    elif selected == "Evaluasi Model":
        st.subheader("Evaluasi Model SVM")
        
        X_train, X_test, y_train, y_test = train_test_split(X, df['sentimen'], test_size=0.2, random_state=42)
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, target_names=['Negatif', 'Positif'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.markdown("### Metrik Tiap Sentimen")
        st.dataframe(report_df.loc[['Negatif', 'Positif']], use_container_width=True)

        st.markdown("### Rangkuman Evaluasi")
        st.dataframe(report_df.drop(index=['Negatif', 'Positif']), use_container_width=True)

        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negatif', 'Positif'],
                    yticklabels=['Negatif', 'Positif'],
                    ax=ax)
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        st.pyplot(fig)


    st.markdown("""<div style='text-align:center;color:#aaa;padding:10px;'>© 2025 Indika Putra Ibrahim</div>""", unsafe_allow_html=True)
