# Analisis Sentimen Kepuasan Masyarakat terhadap Aplikasi "INFO BMKG" Menggunakan Naive Bayes, SVM, dan KNN
<br>


***
### **Abstrak**

Penelitian ini bertujuan untuk menganalisis sentimen kepuasan masyarakat terhadap aplikasi **Info BMKG** di Google Play Store[cite: 10, 24]. Dengan semakin banyaknya pengguna aplikasi berbasis informasi, memahami bagaimana pengguna menilai aplikasi ini menjadi hal penting[cite: 11, 25]. Penelitian ini menggunakan tiga algoritma klasifikasi: **Naive Bayes**, **Support Vector Machine (SVM)**, dan **K-Nearest Neighbors (KNN)**, yang diterapkan untuk mengklasifikasikan ulasan pengguna ke dalam kategori sentimen positif, netral, atau negatif[cite: 12, 26].

Dataset diperoleh melalui *scraping* dari Google Play Store, mencakup data *username*, tanggal, *rating* bintang, dan komentar pengguna[cite: 13, 27]. Tahap *preprocessing* dilakukan untuk membersihkan data dan mempersiapkannya sebelum analisis[cite: 14, 28]. Penelitian ini juga mencakup pengembangan program *data mining* berbasis web yang mempermudah pengolahan data dan visualisasi hasil analisis[cite: 15, 29].

Berdasarkan evaluasi model, algoritma **Naïve Bayes** menunjukkan performa terbaik dengan akurasi 79,84%, presisi 60%, *recall* 58%, dan *runtime* tercepat 0,19 detik[cite: 18, 31, 727]. KNN memiliki akurasi 74,35% dengan *recall* terendah 44% [cite: 19, 32], sedangkan SVM mencapai akurasi 72,26% namun membutuhkan *runtime* terlama yaitu 611 detik[cite: 19, 32]. Validasi AUC memperkuat keunggulan Naïve Bayes dengan nilai tertinggi pada seluruh kategori sentimen[cite: 20, 33]. Dengan demikian, Naïve Bayes terbukti paling optimal untuk analisis sentimen pada studi ini, sementara KNN dan SVM memiliki keterbatasan masing-masing terutama dalam efisiensi dan ketepatan klasifikasi[cite: 21, 34, 41].

**Kata kunci**: sentimen, info bmkg, naive bayes, svm, knn[cite: 22, 43].

***
### **1. Pendahuluan**

Dalam era digital, penggunaan aplikasi seluler telah menjadi bagian integral dari kehidupan sehari-hari[cite: 45]. Salah satu aplikasi yang memiliki peran penting adalah **"Info BMKG,"** yang dirancang untuk memberikan informasi cuaca, gempa bumi, dan peringatan dini[cite: 46]. Namun, seiring dengan meningkatnya jumlah pengguna, muncul berbagai ulasan yang mencerminkan kepuasan dan ketidakpuasan masyarakat terhadap aplikasi ini[cite: 48]. Beberapa ulasan di Google Play Store menunjukkan adanya keluhan terkait fitur yang tidak berjalan dengan baik, lambatnya pembaruan informasi, dan kurangnya respons dari pihak pengembang[cite: 51].

Dari perspektif ilmu pengetahuan dan teknologi (IPTEK), analisis sentimen merupakan alat yang sangat berguna untuk memahami pandangan dan opini masyarakat terhadap produk dan layanan[cite: 53]. Meskipun banyak penelitian telah dilakukan, analisis sentimen yang fokus pada aplikasi khusus seperti "Info BMKG" masih jarang dibahas, sehingga ada kesenjangan dalam literatur yang ada[cite: 55]. Penelitian ini berkontribusi dengan menerapkan dan mengevaluasi tiga metode *machine learning* (Naive Bayes, SVM, dan KNN) dengan pendekatan *preprocessing* yang seragam untuk memastikan perbandingan yang lebih objektif[cite: 74].

***
### **2. Tinjauan Literatur**

* **Naive Bayes** adalah metode klasifikasi berbasis probabilistik yang efisien untuk analisis teks, mengasumsikan bahwa setiap fitur dalam data bersifat independen[cite: 57, 58].
* **Support Vector Machine (SVM)** bekerja dengan mencari *hyperplane* terbaik yang memisahkan data, menjadikannya unggul dalam menangani data yang tidak terdistribusi secara linear[cite: 59, 60].
* **K-Nearest Neighbors (KNN)** adalah metode berbasis *instance-based learning* yang mengklasifikasikan data berdasarkan kedekatan dengan data latih terdekatnya[cite: 61].

Meskipun penelitian sebelumnya telah mengkaji metode-metode tersebut, masih jarang ditemukan studi yang secara langsung membandingkan ketiganya dalam konteks aplikasi layanan kebencanaan di Indonesia, khususnya aplikasi Info BMKG[cite: 67]. Penelitian ini bertujuan untuk mengisi celah tersebut dengan membandingkan efektivitas ketiga metode tersebut[cite: 77].

***
### **3. Metode Penelitian**

Penelitian ini mencakup beberapa tahapan proses yang meliputi pengumpulan data, *preprocessing*, pembobotan TF-IDF, pembagian data, pelatihan model *machine learning*, evaluasi hasil, serta visualisasi analisis[cite: 79].

1.  **Pengumpulan Data**: Ulasan aplikasi "INFO BMKG" diperoleh dari Google Play Store menggunakan teknik *web scraping* dengan Python[cite: 104, 105, 278]. Total 3.100 ulasan diambil, termasuk *username*, tanggal, *rating* bintang, dan komentar[cite: 13, 27, 279].
2.  **Preprocessing Data**: Setelah data diperoleh, dilakukan proses pembersihan untuk memastikan teks siap diproses lebih lanjut[cite: 108, 109]. Proses ini mencakup:
    * *Cleaning* simbol, karakter khusus, dan angka[cite: 111, 112].
    * *Stopword Removal* untuk menghilangkan kata-kata umum[cite: 114, 115].
    * *Stemming* untuk mengubah kata ke bentuk dasarnya[cite: 123].
    * *Tokenization* untuk membagi teks menjadi unit-unit yang lebih kecil[cite: 126, 127].
3.  **Pembobotan TF-IDF**: Teks yang telah diproses diubah menjadi representasi numerik menggunakan metode **Term Frequency-Inverse Document Frequency (TF-IDF)**[cite: 129, 130]. Secara matematis, TF-IDF dihitung dengan rumus: $TF-IDF(t,d)=TF(t,d)x~IDF(t)$[cite: 132].
4.  **Pembagian Data**: Data yang telah diproses dibagi menjadi dua bagian, yaitu **80% untuk data latih** dan **20% untuk data uji**[cite: 95, 98, 145, 146].
5.  **Pelatihan Model (Training Data)**: Data digunakan untuk melatih model menggunakan tiga algoritma: Naïve Bayes, SVM, dan KNN[cite: 149].
6.  **Evaluasi dan Visualisasi**: Evaluasi hasil prediksi dilakukan dengan membandingkan metrik performa seperti **akurasi**, **presisi**, **recall**, **F1-score**, dan validasi **AUC (Area Under Curve)**[cite: 210].

***
### **4. Hasil dan Pembahasan**

#### **Perbandingan Metrik**

| No | Metriks | Naïve Bayes | SVM | KNN |
| :--- | :--- | :--- | :--- | :--- |
| 1. | Akurasi | **79,84%** | 72,26% | 74,35% |
| 2. | Presisi | **60,21%** | 55,24% | 52,90% |
| 3. | Recall | **58,07%** | 53,60% | 43,52% |
| 4. | F1-Score | **57,31%** | 53,20% | 44,55% |
| 5. | Runtime | **0,1857 detik** | 611 detik | 10,32 detik |

Berdasarkan hasil evaluasi, **Naïve Bayes** menunjukkan performa terbaik dalam analisis sentimen aplikasi Info BMKG[cite: 627]. Model ini memiliki akurasi tertinggi sebesar 79,84% [cite: 628] dan *runtime* paling efisien, hanya 0,1857 detik[cite: 626, 627, 659]. Sebaliknya, SVM memiliki akurasi terendah dan *runtime* terlama karena kompleksitas optimasi bobotnya[cite: 628, 626, 661].

#### **Validasi AUC**

| No | Metode | Positif | Negatif | Netral |
| :--- | :--- | :--- | :--- | :--- |
| 1. | Naïve Bayes | **0.798** | **0.544** | **0.820** |
| 2. | SVM | 0.471 | 0.484 | 0.526 |
| 3. | KNN | 0.498 | 0.539 | 0.505 |

Validasi AUC menunjukkan bahwa Naïve Bayes memiliki kemampuan terbaik dalam membedakan kategori positif dan netral[cite: 700, 705]. Sebaliknya, SVM memiliki nilai AUC terendah, menunjukkan kurangnya kemampuan model ini untuk membedakan kelas sentimen[cite: 701, 702, 705].

***
### **5. Kesimpulan**

Penelitian ini menganalisis sentimen kepuasan masyarakat terhadap aplikasi **"INFO BMKG"** menggunakan algoritma Naïve Bayes, SVM, dan KNN[cite: 725]. Hasil evaluasi menunjukkan bahwa **Naïve Bayes** memiliki performa terbaik dengan akurasi tertinggi 79,84% [cite: 726], presisi dan *recall* tertinggi [cite: 727], dan *runtime* tercepat sebesar 0,19 detik[cite: 727]. Validasi AUC juga menunjukkan keunggulan Naïve Bayes dalam membedakan sentimen positif, negatif, dan netral, sementara SVM mencatatkan nilai terendah[cite: 728]. Dengan demikian, Naïve Bayes terbukti sebagai metode paling optimal dalam klasifikasi sentimen pada studi ini[cite: 729].

Penelitian ini memiliki keterbatasan dalam jumlah data dan variasi ulasan yang dapat memengaruhi generalisasi model[cite: 730]. Untuk penelitian selanjutnya, disarankan menggunakan dataset yang lebih besar dan beragam serta mengeksplorasi algoritma lain atau teknik *ensemble* untuk hasil yang lebih akurat dan stabil[cite: 731].