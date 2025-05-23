# Laporan Proyek Machine Learning - Andreas Wirawan Dananjaya

## Domain Proyek
Penyakit jantung merupakan salah satu penyebab kematian terbesar di 
Indonesia dan berbagai jenis penyakit jantung dapat berkontribusi pada statistik 
ini. Pada tahun 2024 di Indonesia, Serangan jantung atau iskemik jantung 
menjadi penyakit terbanyak kedua yang menyebabkan kematian dengan jumlah 
kasus mencapai 95,68 per 100.000 penduduk.

![alt text](https://github.com/andreaswd31/Laporan-Terapan-ke-1/blob/main/Kematian%20Jantung.png?raw=true)

Pada tahun 2020, PJK diperkirakan menjadi penyebab kematian 
tersering, yakni sebesar 36% dari seluruh kematian (Suciana et al., 2021). PJK 
terjadi ketika arteri koroner yang memasok darah ke jantung, menjadi sempit 
atau tersumbat, biasanya oleh plak aterosklerotik. Hal ini dapat mengurangi 
aliran darah ke jantung, menyebabkan angina (nyeri dada) jika arteri sepenuhnya 
tersumbat, serangan jantung (Budhiadnya & Kurniawidjaja, 2022). Program
program kesehatan seperti Program Keluarga Harapan (PKH) dan program 
Krama Badung Sehat telah diluncurkan untuk meningkatkan kualitas hidup dan 
kesehatan masyarakat, yang pada gilirannya diharapkan dapat meningkatkan 
AHH (Sugianto et al., 2020).

tantangan utama tetap ada, yaitu keterbatasan teknologi untuk 
mendeteksi risiko penyakit jantung secara dini dan akurat di tingkat populasi. 
Sebagian besar metode deteksi jantung saat ini bergantung pada perangkat medis 
mahal, seperti elektrokardiogram (EKG) dan treadmill test yang hanya tersedia 
di fasilitas kesehatan tertentu, terutama di kota besar (Islamuddin & Widasari, 
2024). Hal ini memperburuk kondisi di daerah terpencil yang di mana akses 
terhadap alat diagnostik tersebut hampir tidak tersedia. 

Keterbatasan akses dan tingginya biaya perangkat medis konvensional 
berkontribusi pada keterlambatan diagnosis penyakit jantung, yang 
meningkatkan risiko komplikasi serius seperti serangan jantung atau gagal 
jantung. Selain itu, proses klasifikasi risiko penyakit jantung secara manual oleh tenaga medis masih memerlukan waktu yang relatif lama karena harus melalui berbagai tahapan analisis klinis dan interpretasi data medis yang kompleks. Padahal, keterlambatan dalam penanganan dapat berujung pada kondisi fatal yang memerlukan tindakan segera Almansouri et al., 2024). 

Oleh karena itu, diperlukan solusi yang mampu mempercepat proses identifikasi pasien berisiko tinggi secara otomatis dan efisien. Melihat urgensi tersebut, proyek ini bertujuan untuk mengembangkan model klasifikasi berbasis machine learning yang dapat memprediksi risiko penyakit jantung dengan akurasi tinggi menggunakan data medis sederhana. Pendekatan ini diharapkan mampu menjadi solusi awal deteksi dini tanpa harus bergantung pada alat medis mahal.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah diidentifikasi sebelumnya, berikut adalah problem statements dari proyek ini :
- Bagaimana probabilitas model machine learning dalam memprediksi risiko penyakit jantung berdasarkan data pasien yang terbatas namun relevan secara klinis?
- Algoritma klasifikasi mana yang memberikan keseimbangan terbaik antara akurasi, presisi, dan recall untuk kasus deteksi penyakit jantung?
- Sejauh mana solusi berbasis machine learning dapat diandalkan dalam pengambilan keputusan awal diagnosis penyakit jantung sebagai sistem pendukung tenaga medis?

### Goals

Berangkat dari pernyataan masalah sebelumnya, berikut tujuan utama dari proyek ini:
- Mengevaluasi penerapan algoritma klasifikasi untuk memahami probabilitas dan performa dalam memprediksi risiko penyakit jantung berdasarkan data klinis sederhana.
- Mengidentifikasi algoritma klasifikasi terbaik yang dapat memberikan hasil optimal berdasarkan metrik evaluasi seperti akurasi, presisi, recall, dan f1-score.
- Membangun dasar keputusan medis awal berbasis model klasifikasi dengan tingkat akurasi yang meyakinkan dengan mengarahkan hasil model sebagai pendukung diagnosis awal yang mudah dipahami dan dijustifikasi

### Solution statements
- Membangun dan membandingkan beberapa algoritma klasifikasi seperti Decision Tree, Random Forest dan Support Vector Machine mendukung probabilistic output melalui metode predict_proba().
- melakukan evaluasi model menggunakan teknik cross-validation dan membandingkan metrik Accuracy, Precision, Recall, dan F1-Score untuk menentukan keseimbangan terbaik dalam performa.
- Menampilkan confusion matrix untuk mengevaluasi jenis kesalahan yang dilakukan oleh tiap model (false positive / false negative

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Heart Failure Prediction Dataset, yang tersedia di platform Kaggle. Dataset ini berisi informasi medis pasien yang berkaitan dengan kondisi jantung, dan digunakan untuk memprediksi kemungkinan seseorang mengalami penyakit jantung. Dataset ini dapat diunduh melalui tautan berikut:
https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
Dataset ini berisi 918 observasi dengan 12 fitur input dan 1 label output (HeartDisease) yang merepresentasikan kondisi apakah pasien mengalami penyakit jantung (1) atau tidak (0).

Sejumlah variabel atau fitur yang merepresentasikan kondisi klinis pasien. Penjelasan masing-masing variabel adalah sebagai berikut:
- Age: Usia pasien dalam satuan tahun.
- Sex: Jenis kelamin pasien (M: Male/Laki-laki, F: Female/Perempuan).
- ChestPainType: Jenis nyeri dada yang dialami (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic).
- RestingBP: Tekanan darah saat istirahat (dalam mm Hg).
- Cholesterol: Kadar kolesterol serum (dalam mg/dl).
- FastingBS: Gula darah puasa (1 jika > 120 mg/dl, 0 jika ≤ 120 mg/dl).
- RestingECG: Hasil elektrokardiogram saat istirahat (Normal, ST, LVH).
- MaxHR: Denyut jantung maksimum yang dicapai (nilai numerik antara 60–202).
- ExerciseAngina: Apakah pasien mengalami angina akibat olahraga (Y: Ya, N: Tidak).
- Oldpeak: Tingkat depresi segmen ST (nilai numerik yang mengindikasikan iskemia).
- ST_Slope: Kemiringan segmen ST selama latihan (Up: naik, Flat: datar, Down: turun).
- HeartDisease: Label atau kelas keluaran (1: menderita penyakit jantung, 0: normal).

Tidak terdapat data duplikat maupun missing value pada dataset ini (Jumlah duplikasi: 0). Namun, terdapat nilai tidak logis pada fitur RestingBP dan Cholesterol, yaitu adanya nilai nol (0) yang secara medis tidak memungkinkan karena tekanan darah dan kadar kolesterol tidak bisa bernilai nol, serta terdapat outliers di beberapa fitur.

![alt text](https://github.com/andreaswd31/Laporan-Terapan-ke-1/blob/main/Deteksi%20Outliers.png?raw=true)

### Exploratory Data Analysis (EDA)
Untuk memahami pola data, distribusi, serta hubungan antar variabel numerik, dilakukan dua teknik visualisasi utama:
1. Distribusi Variabel Numerikal
![alt text](https://github.com/andreaswd31/Laporan-Terapan-ke-1/blob/main/Distribusi%20Variabel%20Numerikal.png?raw=true)
Distribusi dari masing-masing fitur numerik pada dataset. Usia (Age) dan detak jantung maksimum (MaxHR) menunjukkan distribusi mendekati normal, sedangkan tekanan darah (RestingBP) dan kolesterol (Cholesterol) memiliki beberapa outlier yang signifikan. Nilai Oldpeak cenderung right-skewed dan FastingBS serta HeartDisease merupakan variabel biner dengan distribusi yang tidak seimbang. Sebagian besar pasien memiliki gula darah puasa normal (FastingBS = 0), dan lebih dari setengahnya mengidap penyakit jantung (HeartDisease = 1). Visualisasi ini memberikan gambaran awal tentang karakteristik data dan potensi outlier yang perlu diperhatikan dalam analisis lebih lanjut.

3. Korelasi Antar Variabel Numerikal
![alt text](https://github.com/andreaswd31/Laporan-Terapan-ke-1/blob/main/HeatmapFitur.png?raw=true)
Hubungan linear antar variabel numerik. Ditemukan bahwa Oldpeak dan MaxHR memiliki korelasi paling kuat dengan penyakit jantung (HeartDisease), masing-masing positif (0.40) dan negatif (-0.40), yang berarti semakin tinggi depresi ST cenderung meningkatkan risiko, sementara detak jantung maksimum yang lebih tinggi justru menurunkan risiko. Usia (Age) dan FastingBS juga memiliki korelasi positif terhadap HeartDisease, sedangkan variabel lainnya seperti kolesterol dan tekanan darah menunjukkan korelasi lemah. Visualisasi ini membantu mengidentifikasi fitur-fitur yang paling relevan untuk membangun model prediksi penyakit jantung.

## Data Preparation
1. Menghapus Nilai Tidak Logis pada Fitur RestingBP dan Cholesterol
Pada tahap awal praproses, dilakukan pembersihan data dengan menghapus nilai-nilai tidak logis pada fitur RestingBP (tekanan darah saat istirahat) dan Cholesterol. Nilai nol pada kedua fitur tersebut dianggap tidak valid secara medis, karena tekanan darah dan kadar kolesterol pada manusia tidak mungkin bernilai nol. Nilai-nilai tersebut berpotensi menjadi anomali yang dapat menurunkan performa model jika tidak ditangani dengan tepat. Oleh karena itu, baris data yang memiliki nilai RestingBP = 0 atau Cholesterol = 0 dihapus dari dataset. Hasil dari proses pembersihan ini menunjukkan penurunan jumlah data dari semula 918 baris menjadi 746 baris data bersih yang digunakan untuk tahap analisis dan pemodelan selanjutnya.

2. Menghapus Outlier pada Beberapa Fitur Numerik
Outlier atau nilai pencilan merupakan nilai ekstrem yang secara signifikan berbeda dari sebagian besar data lainnya. Jika tidak ditangani, outlier dapat mempengaruhi distribusi data dan performa model prediktif. Oleh karena itu, dilakukan proses deteksi dan penghapusan outlier pada fitur numerik RestingBP, Cholesterol, MaxHR, dan Oldpeak menggunakan metode Interquartile Range (IQR). IQR adalah teknik statistik yang mengidentifikasi outlier berdasarkan rentang antara kuartil pertama (Q1) dan kuartil ketiga (Q3), dengan batas bawah Q1 - 1.5×IQR dan batas atas Q3 + 1.5×IQR.
Baris data yang mengandung nilai di luar rentang tersebut dihapus dari dataset. Langkah ini dilakukan untuk memastikan distribusi data yang lebih normal dan meningkatkan akurasi model machine learning di tahap selanjutnya.

![alt text](https://github.com/andreaswd31/Laporan-Terapan-ke-1/blob/main/Penanganan%20Outliers.png?raw=true)


4. Memisahkan Fitur dan Label
Dataset kemudian dipisahkan antara fitur independen (X) dan target/label (y) yang akan diprediksi.
    ```python
    X = heart_df.drop('HeartDisease', axis=1)
    y = heart_df['HeartDisease']
    ```
    
3. Identifikasi Fitur Kategorikal dan Numerik
mengidentifikasi tipe data dari setiap fitur, yang dibedakan menjadi dua jenis utama: fitur kategorikal dan numerik. Fitur kategorikal terdiri dari Sex, ChestPainType, RestingECG, ExerciseAngina, dan ST_Slope, sedangkan fitur numerik mencakup Age, RestingBP, Cholesterol, FastingBS, MaxHR, dan Oldpeak. Identifikasi ini diperlukan untuk menerapkan teknik praproses yang sesuai, karena fitur kategorikal membutuhkan encoding, sementara fitur numerik perlu distandarisasi.
    - Fitur Kategorikal (cat_cols): Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope
    - Fitur Numerik (num_cols): Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
     
6. Preprocessing: Encoding dan Scaling
Proses preprocessing terdiri dari dua tahap utama:
    - Encoding fitur kategorikal menggunakan OneHotEncoder (dengan parameter drop='first' untuk menghindari dummy variable trap).
    - Scaling fitur numerik menggunakan StandardScaler agar setiap fitur memiliki distribusi dengan rata-rata 0 dan standar deviasi 1 yang sangat penting untuk algoritma berbasis jarak seperti SVM.
Pipeline preprocessing dibuat menggunakan ColumnTransformer:
    ```python
    preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), cat_cols),
    ('num', StandardScaler(), num_cols)
    ])
    ```
7. Pembagian Data: Train-Test Split
Dataset dibagi menjadi data latih dan data uji dengan proporsi 80:20, menggunakan metode stratified split untuk memastikan proporsi label seimbang di kedua subset.
     ```python
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
    ```
    Pembagian ini dilakukan menggunakan fungsi train_test_split dari scikit-learn, dengan parameter stratify=y untuk menjaga proporsi kelas target tetap seimbang pada kedua subset. Pembagian ini bertujuan untuk menguji performa model secara objektif, yaitu dengan mengevaluasi hasil prediksi model pada data uji yang tidak pernah dilihat selama proses pelatihan.

## Modeling
 Pada penelitian ini, digunakan tiga algoritma pembelajaran terawasi, yaitu Decision Tree, Random Forest, dan Support Vector Machine (SVM). Masing-masing model dipasangkan dengan pipeline praproses agar proses transformasi data dan pelatihan model berjalan secara terpadu.
 
 1. Decision Tree Classifier
 Model Decision Tree digunakan sebagai baseline karena interpretabilitasnya yang tinggi dan kemampuan dalam menangani fitur numerik maupun kategorikal. Model ini menggunakan parameter sebagai berikut:
    - criterion='gini': Menggunakan indeks Gini untuk mengukur kualitas pemisahan
    - max_depth=8: Membatasi kedalaman pohon agar tidak terlalu kompleks.
    - min_samples_split=10: Minimal jumlah sampel untuk membagi node.
    - min_samples_leaf=5: Minimal jumlah sampel pada setiap daun (leaf node).
    - random_state=42: Menjamin replikasi hasil.- 
    
    Model ini dilatih menggunakan Pipeline yang mengintegrasikan tahapan praproses (OneHotEncoder dan StandardScaler) dengan DecisionTreeClassifier. Setelah pelatihan, model melakukan prediksi terhadap data uji.

    Kelebihan:
    - Mudah diinterpretasikan (visualisasi pohon keputusan).
    - Cepat untuk pelatihan dan prediksi.
    - Tidak memerlukan normalisasi data secara eksplisit.
    
    Kekurangan:
    - Rentan terhadap overfitting, terutama pada data dengan banyak fitur atau noise.
    - Sensitif terhadap variasi kecil pada data.

2. Random Forest Classifier
Model Random Forest merupakan ensembel dari beberapa pohon keputusan yang digabungkan untuk meningkatkan akurasi dan mengurangi overfitting. Pipeline dibuat serupa dengan Decision Tree, namun menggunakan RandomForestClassifier(random_state=42): Dengan parameter default untuk memulai proses baseline modeling. Model ini bekerja dengan membuat beberapa pohon keputusan berdasarkan subset acak dari data dan fitur, kemudian hasilnya di-vote secara mayoritas.

    Kelebihan:
    - Memiliki performa yang baik dalam banyak kasus karena reduksi overfitting.
    - Lebih stabil dibandingkan Decision Tree tunggal.
    - Mampu menangani data besar dan fitur dalam jumlah banyak.
    
    Kekurangan:
    - Kurang dapat diinterpretasikan secara langsung karena kompleksitas tinggi.
    - Waktu pelatihan lebih lama dibanding Decision Tree.

3. Support Vector Machine (SVM)
Algoritma SVM digunakan dengan kernel Radial Basis Function (RBF) yang cocok untuk kasus klasifikasi non-linear. Parameter yang digunakan pada model adalah:
    - C=2: Parameter regularisasi untuk mengontrol trade-off antara margin dan misclassifications.
    - kernel='rbf': Kernel non-linear yang banyak digunakan untuk data yang tidak dapat dipisahkan secara linear.
    - gamma='scale': Nilai gamma otomatis berdasarkan jumlah fitur.
    - probability=True: Mengaktifkan estimasi probabilitas (diperlukan untuk evaluasi metrik probabilistik).
    - random_state=4: Untuk memastikan reproducibility.
    
    Kelebihan:
    - Efektif untuk data berdimensi tinggi.
    - Mampu memisahkan kelas secara optimal menggunakan hyperplane maksimum margin.
    
    Kekurangan:
    - Waktu komputasi cukup tinggi, terutama pada dataset besar.
    - Parameter tuning yang sensitif dan memerlukan proses optimasi yang cermat.
    - Sulit diinterpretasikan secara intuitif.
    
Setelah ketiga model diuji menggunakan dataset yang sama, dilakukan evaluasi menggunakan metrik akurasi, precision, recall, dan F1-score untuk membandingkan performa masing-masing algoritma. Berdasarkan ketiga model yang telah dilatih dan diuji, proses seleksi model akan dilakukan pada bagian evaluasi. Namun, dari hasil awal pelatihan, Random Forest menunjukkan potensi sebagai kandidat terbaik karena kombinasi performa yang stabil dan kemampuan menangani overfitting.

## Evaluation
Evaluasi model dilakukan dengan menggunakan beberapa metrik performa yang umum dalam permasalahan klasifikasi, yaitu akurasi, precision, recall, dan F1-score. Pemilihan metrik ini didasarkan pada kebutuhan untuk tidak hanya mengukur jumlah prediksi yang benar secara keseluruhan, tetapi juga memperhatikan performa model dalam mengenali masing-masing kelas (positif dan negatif), yang dalam konteks ini adalah adanya indikasi penyakit jantung atau tidak.

Metrik Evaluasi dan Formulanya
1. Akurasi
Mengukur proporsi prediksi yang benar terhadap keseluruhan data:
$$\text{Akurasi} = \frac{TP + TN}{TP + TN + FP + FN}$$
Di mana:
    - TP: True Positive
    - TN: True Negative
    - FP: False Positive
    - FN: False Negative

2. Precision
Mengukur ketepatan prediksi positif:
$$\text{Precision} = \frac{TP}{TP + FP}$$    
Precision penting ketika kesalahan positif (false positive) berdampak besar, seperti salah mendiagnosis orang sehat sebagai sakit.

3. Recall
Mengukur kemampuan model dalam menemukan semua data positif:
$$\text{Recall} = \frac{TP}{TP + FN}$$
Recall krusial dalam kasus penyakit jantung karena lebih berisiko jika model gagal mendeteksi orang yang sebenarnya sakit.

4. F1-Score
Rata-rata harmonis dari precision dan recall:
$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
F1-score cocok digunakan jika ingin menjaga keseimbangan antara precision dan recall.

Berikut adalah hasil evaluasi untuk masing-masing model berdasarkan data uji:
1. Decision Tree Classifier
    - Accuracy: 84.89%
    - Precision (class 1): 81%
    - Recall (class 1): 88%
    - F1-Score (class 1): 84%
      
![alt text](https://github.com/andreaswd31/Laporan-Terapan-ke-1/blob/main/Decision%20Tree%20-%20Confusion%20Matrix.png?raw=true)

Model Decision Tree menunjukkan recall yang tinggi (88%), artinya model cukup baik dalam mendeteksi pasien yang benar-benar sakit, namun precision-nya relatif rendah (81%), yang berarti cukup banyak pasien sehat yang justru salah diklasifikasikan sebagai sakit (false positive sebanyak 13 kasus).

2. Random Forest Classifier
    - Accuracy: 87.77%
    - Precision (positif): 84%
    - Recall (positif): 91%
    - F1-Score (positif): 87%
      
![alt text](https://github.com/andreaswd31/Laporan-Terapan-ke-1/blob/main/Random%20Forest%20-%20Confusion%20Matrix.png?raw=true)

Random Forest memiliki kinerja terbaik dibanding dua model lainnya. Dengan recall sebesar 91%, model ini sangat efektif mendeteksi pasien sakit dan hanya gagal mendeteksi 6 pasien sakit (FN). Precision-nya juga cukup tinggi (84%), menunjukkan bahwa sebagian besar prediksi "positif" memang benar-benar sakit. Hal ini penting agar pasien sakit tidak terlewat, mengingat konteksnya adalah penyakit jantung yang serius.

3. Support Vector Machine (SVM)
    - Accuracy: 87.77%
    - Precision (positif): 85%
    - Recall (positif): 89%
    - F1-Score (positif): 87%
      
![alt text](https://github.com/andreaswd31/Laporan-Terapan-ke-1/blob/main/SVM%20-%20Confusion%20Matrix.png?raw=true)

SVM memiliki akurasinya setara dengan Random Forest. Precision-nya lebih tinggi sedikit (85%), artinya prediksi "sakit" oleh model ini lebih akurat, namun recall-nya sedikit lebih rendah dibanding Random Forest (89% vs 91%). Ini berarti model masih gagal mendeteksi 7 pasien sakit yang sebenarnya perlu perhatian.

### Model Case Testing
Untuk menguji keandalan model dalam situasi nyata, dua kasus pasien diuji menggunakan model yang telah diekspor:
- Case 1 (Prediksi Negatif)
Ketiga model secara konsisten memprediksi pasien tidak mengidap penyakit jantung (label 0).
Decision Tree menunjukkan keyakinan penuh (100%),
    - Decision Tree menunjukkan keyakinan penuh (100%),
    - SVM dan Random Forest menunjukkan probabilitas tinggi (90%).
    
Hal ini menandakan bahwa data pasien berada dalam wilayah yang aman secara klinis, dan model mampu mengenali itu dengan baik.
    
- Case 2 (Prediksi Positif)
Semua model memprediksi pasien berisiko tinggi mengidap penyakit jantung (label 1).
    - Decision Tree dan SVM menunjukkan keyakinan 100%,
    - Random Forest memberikan 85% probabilitas.
    
Prediksi ini didukung oleh fitur-fitur kritis seperti tekanan darah tinggi, kolesterol tinggi, dan nyeri dada saat aktivitas fisik.

### Kesimpulan Evaluasi

![alt text](https://github.com/andreaswd31/Laporan-Terapan-ke-1/blob/main/Perbandingan%20Akurasi%20Model.png?raw=true)

Berdasarkan hasil evaluasi terhadap tiga model klasifikasi, yaitu Decision Tree, Random Forest, dan Support Vector Machine (SVM), serta pengujian pada studi kasus representatif, diperoleh sejumlah temuan yang merefleksikan keterkaitan langsung antara performa model dengan problem statements serta sejauh mana capaian terhadap tujuan yang telah ditetapkan dalam Business Understanding.

1. Model-model yang diuji mampu menghasilkan probabilitas prediksi dengan tingkat keyakinan yang tinggi. Contohnya, pada case testing, Decision Tree dan SVM mampu memberikan keyakinan 100% baik untuk kasus positif maupun negatif, dan Random Forest menunjukkan probabilitas tinggi (>85%) untuk kedua kasus.  Ini menunjukkan bahwa meskipun data yang digunakan terbatas, model dapat memberikan probabilistic output yang bermakna dan dapat diinterpretasikan secara klinis.
2. Random Forest menunjukkan performa terbaik secara keseluruhan dengan:
    - Akurasi: 87.77%
    - Precision: 84%
    - Recall: 91%
    - F1-score: 87%
    
    Model ini memiliki recall tertinggi, artinya sangat baik dalam mengidentifikasi pasien yang benar-benar sakit, dan tetap menjaga precision yang layak.

3. Random Forest dan SVM terbukti mampu mengenali pasien dengan indikasi penyakit jantung secara akurat, dengan kesalahan minimum (False Negative (FN) hanya sebanyak 6 dan 7 pasien pada masing-masing model). False Negative (FN) terjadi ketika model memprediksi bahwa pasien tidak mengidap penyakit jantung, padahal sebenarnya pasien tersebut positif. Ini sangat berbahaya dalam konteks medis karena pasien yang sebenarnya membutuhkan penanganan bisa saja tidak ditindaklanjuti. Oleh karena itu, semakin kecil nilai FN, semakin tinggi tingkat keandalan model dalam mendeteksi pasien yang benar-benar sakit, terutama untuk digunakan dalam tahap awal diagnosis atau early screening. Model dengan FN rendah mengurangi risiko missed diagnosis, yang sangat penting dalam kasus penyakit serius seperti jantung.

## Daftar Pustaka

Almansouri, N. E., Awe, M., Rajavelu, S., Jahnavi, K., Shastry, R., Hasan, A., Hasan, H., Lakkimsetti, M., AlAbbasi, R. K., Gutiérrez, B. C., & Haider, A. (2024). Early Diagnosis of Cardiovascular Diseases in the Era of Artificial Intelligence: An In-Depth Review. Cureus. https://doi.org/10.7759/cureus.55869

Budhiadnya, A. K., & Kurniawidjaja, M. (2022). HUBUNGAN ANTARA KARAKTERISTIK PEKERJA DAN PERILAKU PEKERJA DENGAN TINGKAT RISIKO PENYAKIT JANTUNG KORONER DI PT X. PREPOTIF Jurnal Kesehatan Masyarakat, 6, 1963–1971. https://doi.org/10.31004/prepotif.v6i2.5022

Islamuddin, J., & Widasari, E. R. (2024). Fakultas Ilmu Komputer Sistem 
Monitoring Kesehatan Jantung Menggunakan Metode Adaptive Threshold 
Berbasis Shimmer Electrocardiogram Dan Matlab (Vol. 1, Issue 1). http://jptiik.ub.ac.id 

Suciana, Hengky, H. K., & Usma. (2021). ANALISIS FAKTOR RISIKO PENYAKIT 
JANTUNG KORENER PADA PENDERITA DIABETES MELITUS TIPE 2 DI 
RSUD ANDI MAKKASAU KOTA PAREPARE (Vol. 4, Issue 2). 
https://doi.org/10.31850/makes.v4i2.612  

Sugianto, M. A., Agung, A., Widyawati, I. A., Penelitian, B., Kabupaten, P., & 
Indonesia, B. (2020). MANFAAT PROGRAM KRAMA BADUNG SEHAT 
DALAM MENINGKATKAN KESEJAHTERAAN MASYARAKAT. Bali 
Health Published Journal, 2(1). 
