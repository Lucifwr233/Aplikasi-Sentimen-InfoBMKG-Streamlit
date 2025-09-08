# Analisis Sentimen Ulasan Aplikasi Info BMKG

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

Aplikasi web interaktif yang dibangun menggunakan Streamlit untuk menganalisis sentimen dari ulasan pengguna aplikasi "Info BMKG". Aplikasi ini membandingkan kinerja tiga model klasifikasi machine learning: **Naive Bayes**, **Support Vector Machine (SVM)**, dan **K-Nearest Neighbors (KNN)**.

### Demo Aplikasi
![1](Assets/Screenshot1)
![2](Assets/Screenshot2)
![3](Assets/Screenshot3)
![4](Assets/Screenshot4)
![5](Assets/Screenshot5)

## Deskripsi

Proyek ini bertujuan untuk mengklasifikasikan sentimen (positif, negatif, atau netral) dari ulasan teks yang diberikan oleh pengguna aplikasi Info BMKG di Google Play Store. Dengan menggunakan model yang telah dilatih sebelumnya, aplikasi ini menyediakan platform untuk:
1.  Melakukan prediksi sentimen secara real-time pada teks baru.
2.  Mengevaluasi dan membandingkan performa dari tiga model klasifikasi populer.
3.  Menyediakan visualisasi data yang informatif untuk analisis lebih mendalam.

## Fitur-Fitur Utama

-   **Prediksi Sentimen Real-Time**: Masukkan teks ulasan apapun dan dapatkan hasil prediksi sentimen dari tiga model secara bersamaan.
-   **Perbandingan Model**: Lihat dan bandingkan hasil dari Naive Bayes, SVM, dan KNN secara berdampingan.
-   **Tampilan Dataset**: Muat dan tampilkan dataset asli yang digunakan dalam proyek ini.
-   **Dasbor Visualisasi & Evaluasi**:
    -   **Distribusi Sentimen**: Diagram batang yang membandingkan distribusi sentimen (positif, negatif, netral) antara label data asli dengan hasil prediksi setiap model.
    -   **Word Cloud**: Visualisasi kata-kata yang paling sering muncul untuk setiap kategori sentimen.
    -   **Metrik Performa**: Tabel ringkasan performa model yang mencakup **Akurasi, Presisi, Recall, dan F1-Score**.
    -   **Confusion Matrix**: Visualisasi matriks konfusi untuk setiap model guna melihat performa klasifikasi secara detail.
    -   **Kurva ROC (AUC)**: Grafik kurva ROC untuk mengevaluasi kemampuan diskriminasi model.

## Struktur Proyek
```
sentiment_analysis_app/
├── streamlit_app.py              # Kode utama aplikasi
├── Model/
│   ├── naive_bayes_custom_model.pkl
│   ├── svm_custom_model_multi.json
│   └── knn_model.pkl
├── Dataset/
│   ├── 1 Dataset Asli.csv
│   └── 6 hasil_gabungan_prediksi.csv
├── requirements.txt              # Daftar library yang dibutuhkan
└── README.md                     # File ini
```

## Instalasi dan Cara Menjalankan

Ikuti langkah-langkah berikut untuk menjalankan aplikasi ini di komputer lokal Anda.

### Prasyarat
-   Python 3.8 atau versi yang lebih baru
-   pip (Package Installer for Python)

### Langkah-langkah Instalasi

1.  **Clone Repositori**
    Buka terminal atau command prompt dan clone repositori ini:
    ```bash
    git clone [https://github.com/Lucifwr233/Aplikasi-Sentimen-InfoBMKG-Streamlit.git](https://github.com/Lucifwr233/Aplikasi-Sentimen-InfoBMKG-Streamlit.git)
    ```

2.  **Masuk ke Direktori Proyek**
    ```bash
    cd NAMA_REPOSITORI_ANDA
    ```

3.  **Instal Semua Dependensi**
    Instal semua library yang dibutuhkan dari file `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi Streamlit**
    Setelah semua instalasi selesai, jalankan perintah berikut:
    ```bash
    streamlit run streamlit_app.py
    ```
    Aplikasi akan otomatis terbuka di browser Anda.

## Model yang Digunakan

-   **Naive Bayes**: Sebuah algoritma klasifikasi probabilistik berdasarkan Teorema Bayes. Model ini cepat, sederhana, dan bekerja dengan baik pada klasifikasi teks.
-   **Support Vector Machine (SVM)**: Model yang bertujuan untuk menemukan *hyperplane* terbaik yang memisahkan data ke dalam kelas-kelas yang berbeda. Efektif dalam ruang berdimensi tinggi.
-   **K-Nearest Neighbors (KNN)**: Algoritma *instance-based learning* yang mengklasifikasikan data baru berdasarkan mayoritas kelas dari 'k' tetangga terdekatnya.

