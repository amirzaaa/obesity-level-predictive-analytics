# Laporan Proyek Machine Learning - Amir Hamzah

## Domain Proyek

Domain proyek yang saya pilih pada proyek ini adalah tentang tema kesehatan dengan judul proyek "Klasifikasi Tingkat Obesitas Berdasarkan Kebiasaan Makan dan Kondisi Fisik"

### Latar Belakang

Kelebihan berat badan, juga dikenal sebagai obesitas, adalah kondisi di mana seseorang mengalami sejumlah anomali dan lemak berlebih, yang merupakan salah satu faktor risiko kesehatan yang signifikan. Standar yang ditetapkan oleh Organisasi Kesehatan Dunia (WHO) untuk batas berat badan diukur dengan menggunakan indeks massa tubuh (BMI), di mana nilai 25 atau lebih dianggap kelebihan berat badan dan nilai 30 atau lebih dianggap obesitas [[1](https://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2443)]. Menurut Fabio, ada tujuh level
dalam obesitas diantaranya Insufficient Weight,
Normal Weight, Overweight Level I, Overweight
Level II, Obesity Type I, Obesity Type II, dan
Obesity Type III [[2](https://pdfs.semanticscholar.org/b5f0/8012dfee726d261da0cb0758ca29d3276111.pdf)].

Menurut hasil penelitian kesehatan dasar yang dilakukan pada tahun 2018, tingkat obesitas pada individu berusia lebih dari 18 tahun meningkat dari 14,8% menjadi 21,8%. Obesitas juga dapat meningkatkan risiko penyakit jantung dan stroke [[3](https://ejournal.pelitaindonesia.ac.id/ojs32/index.php/JOISIE/article/view/2467/1009)]. Obesitas adalah masalah kesehatan yang signifikan di seluruh dunia dengan dampak yang signifikan. Dalam beberapa dekade terakhir, obesitas menjadi masalah kesehatan masyarakat utama di Meksiko. Di Meksiko, tingkat obesitas dewasa mencapai 74,5% wanita dan 69,1% pria [[4](https://proceeding.unpkediri.ac.id/index.php/inotek/article/view/5062)]. Data WHO terbaru menunjukkan bahwa pada tahun 2022, 2,5 miliar orang dewasa berusia 18 tahun ke atas mengalami kelebihan berat badan, dengan 890 juta termasuk dalam kategori obesitas, dan jika tingkat pertumbuhan ini terus berlanjut, proporsi orang yang masuk dalam kategori obesitas di Indonesia akan meningkat menjadi 2 miliar pada tahun 2030. Survei Kesehatan Indonesia pada tahun 2023 menunjukkan bahwa jumlah orang yang masuk dalam kategori obesitas di Indonesia akan meningkat menjadi 2 miliar [[5](https://mail.ejournal.itn.ac.id/index.php/jati/article/view/13397/7526)].

Oleh karena itu, memahami dan mengklasifikasikan tingkat obesitas sangat penting untuk mencegah dan menangani obesitas. Penelitian ini menggunakan algoritma machine learning untuk melakukan klasifikasi terhadap tingkat obesitas pada individu dengan menggunakan data mengenai kebiasaan makan, aktivitas fisik, dan variabel lainnya yang terkait dengan tingkat obesitas. Tujuan dari metode ini adalah untuk memahami dan mengidentifikasi faktor-faktor yang memengaruhi tingkat obesitas [[6](https://ojs3.unpatti.ac.id/index.php/parameter/article/view/11875)]. Dataset yang akan digunakan adalah dengan menggunakan dataset yang didapat dari website UCI Machine Learning Repository terkait _Estimation of Obesity Levels Based On Eating Habits and Physical Condition_ [[7](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)].

## Referensi

- [[1](https://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2443)] Sitanggang, D., & Sherly, S. (2022). Model Prediksi Obesitas dengan Menggunakan Support Vector Machine. Jurnal Sistem Informasi dan Ilmu Komputer, 5(2), 172-175.
- [[2](https://pdfs.semanticscholar.org/b5f0/8012dfee726d261da0cb0758ca29d3276111.pdf)] Setiyani, L., Indahsari, A. N., & Roestam, R. (2023). Analisis Prediksi Level Obesitas Menggunakan Perbandingan Algoritma Machine Learning dan Deep Learning. JTERA (Jurnal Teknol. Rekayasa), 8(1), 139.
- [[3](https://ejournal.pelitaindonesia.ac.id/ojs32/index.php/JOISIE/article/view/2467/1009)] Wie, J. V., & Siddik, M. (2023). Penerapan Metode Naïve Bayes Dalam Mengklasifikasi Tingkat Obesitas Pada Pria. JOISIE (Journal Of Information Systems And Informatics Engineering), 6(2), 69-77.
- [[4](https://proceeding.unpkediri.ac.id/index.php/inotek/article/view/5062)] Aini, E. D. N., Khasanah, R. A., Ristyawan, A., & Diniati, E. (2024, July). Penggunaan Data Mining untuk Prediksi tingkat Obesitas di Meksiko Menggunakan Metode Random Forest. In Prosiding SEMNAS INOTEK (Seminar Nasional Inovasi Teknologi) (Vol. 8, No. 3, pp. 1256-1265).
- [[5](https://mail.ejournal.itn.ac.id/index.php/jati/article/view/13397/7526)] Khikam, A., Anggadimas, N. M., & Udin, M. (2025). IMPLEMENTASI DECISION TREE UNTUK KLASIKASI OBESITAS. JATI (Jurnal Mahasiswa Teknik Informatika), 9(3), 3946-3952.
- [[6](https://ojs3.unpatti.ac.id/index.php/parameter/article/view/11875)] Fitriani, D. N. (2024). Prediksi PREDIKSI TINGKAT OBESITAS MENGGUNAKAN NEURAL NETWORK: PENDEKATAN KLASIFIKASI BINER. PARAMETER: Jurnal Matematika, Statistika dan Terapannya, 3(01), 85-92.
- [[7](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)] _Estimation of Obesity Levels Based On Eating Habits and Physical Condition_, UCI Machine Learning Repository.

## Business Understanding

Pada proyek ini, peneliti menggunakan dataset [_Estimation of Obesity Levels Based On Eating Habits and Physical Condition_](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) yang peneliti dapatkan pada UCI Machine Learning Repository untuk memngklasifikasikan tingkat obesitas berdasarkan kebiasaan dan gaya hidup dari suatu individu. Untuk mewujudkan hal tersebut, peneliti menggunakan algoritma machine learning yang berhubungan dengan klasifikasi.

### Problem Statements

Berdasarkan latar belakang yang telah peneliti paparkan sebelumnya, berikut ini adalah problem statements yang dapat peneliti jabarkan:

- Bagaimana mengklasifikasikan tingkat obesitas individu berdasarkan data kebiasaan makan, aktivitas fisik, dan faktor gaya hidup lainnya?
- Seberapa akurat model prediksi yang dibangun dalam mengklasifikasikan tingkat obesitas pada individu?
- Bisakah model klasifikasi digunakan untuk membantu dalam merancang intervensi gaya hidup personalisasi guna menurunkan risiko obesitas?

### Goals

- Untuk menerapkan penggunaan dari model algoritma machine learning untuk klasifikasi tingkat obesitas berdasarkan data yang ada
- Untuk mengevaluasi performa dari model menggunakan matriks seperti akurasi, precision, recall, dan f1-score
- Untuk memberikan manfaat praktis berdasarkan model yang telah dibangun

### Solution statements

- Menggunakan 3 Algoritma Klasifikasi dari Machine Learning, seperti Random Forest Classifier, Gradient Boosting Classifier, dan Support Vector Classifier.
- Menerapkan Principle Component Analysis untuk mengurangi dimensi dari dataset

## Data Understanding

### Informasi Dataset

Adapun dataset yang digunakan oleh peneliti adalah dataset [_Estimation of Obesity Levels Based On Eating Habits and Physical Condition_](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). Dataset ini memiliki fitur target berdasarkan tingkatan dari obesitasnya, seperti _Normal_Weight, Overweight_Level_I, Overweight_Level_II,
Obesity_Type_I, Insufficient_Weight, Obesity_Type_II,
Obesity_Type_III_ yang akan digunakan untuk mengklasifikasikan tingkat dari obesitas individual.

### Variabel-variabel pada Obesity UCI dataset adalah sebagai berikut:

| Variabel                       | Tipe        | Deskripsi                                                        |
| ------------------------------ | ----------- | ---------------------------------------------------------------- |
| Gender                         | Kategorikal | Jenis kelamin dari tiap individu                                 |
| Age                            | Kontinu     | Usia dari tiap individu                                          |
| Height                         | Kontinu     | Tinggi badan dari tiap individu                                  |
| Weight                         | Kontinu     | Berat badan dari tiap individu                                   |
| family_history_with_overweight | Biner       | Apakah memiliki riwayat keluarga yang mengalami kelebihan berat? |
| FAVC                           | Biner       | Apakah sering mengonsumsi makanan tinggi kalori?                 |
| FCVC                           | Integer     | Apakah biasanya mengonsumsi sayur dalam makanan harian?          |
| NCP                            | Kontinu     | Berapa kali makan utama dalam sehari?                            |
| CAEC                           | Kategorikal | Apakah mengonsumsi makanan di antara waktu makan?                |
| SMOKE                          | Biner       | Apakah merokok?                                                  |
| CH2O                           | Kontinu     | Berapa banyak air yang dikonsumsi setiap hari? (0 hingga 3)                   |
| SCC                            | Biner       | Apakah memantau kalori yang dikonsumsi setiap hari?              |
| FAF                            | Kontinu     | Seberapa sering melakukan aktivitas fisik? (0 hingga 3)                       |
| TUE                            | Integer     | Waktu yang digunakan dengan perangkat teknologi (TV, HP, dll.) (0 hingga 2)   |
| CALC                           | Kategorikal | Seberapa sering mengonsumsi alkohol?                             |
| MTRANS                         | Kategorikal | Moda transportasi yang biasa digunakan                           |
| NObeyesdad                     | Kategorikal | Tingkat obesitas                                                 |
   
### Explanatory Data Analysis

Agar dapat lebih memahami terhadap dataset yang akan digunakan, berikut ini adalah tahapan eksplorasi data yang peneliti lakukan:

<h4>1. Memahami Informasi Dataset</h4>

<p>Berikut adalah beberapa hal yang peneliti lakukan untuk memahami dataset yang digunakan:</p>

<ul>
  <li><strong>Menampilkan Informasi dari Dataset</strong>
    <p style="text-indent: 40px;">
      Tahapan ini dilakukan untuk menampilkan data dalam dataframe sebanyak 5 baris, kode yang digunakan adalah <code>df.head()</code>. 
      Kemudian peneliti juga menggunakan <code>df.info()</code> untuk menampilkan informasi terkait jenis tipe data, dan memberikan informasi jumlah baris dan kolom. 
      Berikut adalah hasil dari penerapan kode-kode tersebut.
    </p>

  <table border="1">
      <thead>
        <tr>
          <th>Gender</th><th>Age</th><th>Height</th><th>Weight</th><th>Family History</th>
          <th>FAVC</th><th>FCVC</th><th>NCP</th><th>CAEC</th><th>SMOKE</th>
          <th>CH2O</th><th>SCC</th><th>FAF</th><th>TUE</th><th>CALC</th><th>MTRANS</th><th>NObeyesdad</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Female</td><td>21.0</td><td>1.62</td><td>64.0</td><td>yes</td><td>no</td><td>2.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.0</td><td>no</td><td>0.0</td><td>1.0</td><td>no</td><td>Public_Transportation</td><td>Normal_Weight</td></tr>
        <tr><td>Female</td><td>21.0</td><td>1.52</td><td>56.0</td><td>yes</td><td>no</td><td>3.0</td><td>3.0</td><td>Sometimes</td><td>yes</td><td>3.0</td><td>yes</td><td>3.0</td><td>0.0</td><td>Sometimes</td><td>Public_Transportation</td><td>Normal_Weight</td></tr>
        <tr><td>Male</td><td>23.0</td><td>1.80</td><td>77.0</td><td>yes</td><td>no</td><td>2.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.0</td><td>no</td><td>2.0</td><td>1.0</td><td>Frequently</td><td>Public_Transportation</td><td>Normal_Weight</td></tr>
        <tr><td>Male</td><td>27.0</td><td>1.80</td><td>87.0</td><td>no</td><td>no</td><td>3.0</td><td>3.0</td><td>Sometimes</td><td>no</td><td>2.0</td><td>no</td><td>2.0</td><td>0.0</td><td>Frequently</td><td>Walking</td><td>Overweight_Level_I</td></tr>
        <tr><td>Male</td><td>22.0</td><td>1.78</td><td>89.8</td><td>no</td><td>no</td><td>2.0</td><td>1.0</td><td>Sometimes</td><td>no</td><td>2.0</td><td>no</td><td>0.0</td><td>0.0</td><td>Sometimes</td><td>Public_Transportation</td><td>Overweight_Level_II</td></tr>
      </tbody>
    </table>

  <br>
    <table border="1">
      <thead>
        <tr>
          <th>No</th><th>Kolom</th><th>Non-Null Count</th><th>Tipe Data</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>0</td><td>Gender</td><td>2111</td><td>object</td></tr>
        <tr><td>1</td><td>Age</td><td>2111</td><td>float64</td></tr>
        <tr><td>2</td><td>Height</td><td>2111</td><td>float64</td></tr>
        <tr><td>3</td><td>Weight</td><td>2111</td><td>float64</td></tr>
        <tr><td>4</td><td>family_history_with_overweight</td><td>2111</td><td>object</td></tr>
        <tr><td>5</td><td>FAVC</td><td>2111</td><td>object</td></tr>
        <tr><td>6</td><td>FCVC</td><td>2111</td><td>float64</td></tr>
        <tr><td>7</td><td>NCP</td><td>2111</td><td>float64</td></tr>
        <tr><td>8</td><td>CAEC</td><td>2111</td><td>object</td></tr>
        <tr><td>9</td><td>SMOKE</td><td>2111</td><td>object</td></tr>
        <tr><td>10</td><td>CH2O</td><td>2111</td><td>float64</td></tr>
        <tr><td>11</td><td>SCC</td><td>2111</td><td>object</td></tr>
        <tr><td>12</td><td>FAF</td><td>2111</td><td>float64</td></tr>
        <tr><td>13</td><td>TUE</td><td>2111</td><td>float64</td></tr>
        <tr><td>14</td><td>CALC</td><td>2111</td><td>object</td></tr>
        <tr><td>15</td><td>MTRANS</td><td>2111</td><td>object</td></tr>
        <tr><td>16</td><td>NObeyesdad</td><td>2111</td><td>object</td></tr>
      </tbody>
    </table>

  <p style="text-indent: 40px;">
      Dari hasil yang ada di tabel tersebut, dapat diketahui bahwa dataset yang digunakan memiliki total 2111 baris dengan total 17 kolom, dan tanpa nilai null atau missing value. 
      Adapun tipe data yang terdapat pada dataset tersebut memiliki 2 jenis, yakni tipe data object (string) dan float.
    </p>
  </li>

  <li><strong>Menampilkan Total Data Terduplikasi</strong>
    <p style="text-indent: 40px;">
      Kode di atas digunakan untuk menampilkan jumlah dari data yang terduplikasi dalam dataset yang akan digunakan. Berikut adalah hasil dari kode tersebut.
    </p>
    <img src="https://github.com/user-attachments/assets/ebda31d1-9028-4409-b1da-bd847b0be898" alt="duplicated data">

  <p style="text-indent: 40px;">
      Dari hasil tersebut dapat diketahui bahwa terdapat total 24 data yang terduplikasi pada dataset yang akan digunakan. Agar model yang akan dibuat menjadi lebih optimal, data terduplikasi tersebut perlu untuk dilakukan pembersihan agar tidak mempengaruhi performa dari model.
    </p>
  </li>

  <li><strong>Menampilkan Statistika Deskriptif</strong>
    <p style="text-indent: 40px;">
      Kode di atas digunakan untuk menampilkan statistika deskriptif seperti mean, median, max, min, standard deviation, count dan kurtil. Berikut adalah hasil dari kode tersebut.
    </p>

  <table border="1">
      <thead>
        <tr><th>Statistik</th><th>Age</th><th>Height</th><th>Weight</th><th>FCVC</th><th>NCP</th><th>CH2O</th><th>FAF</th><th>TUE</th></tr>
      </thead>
      <tbody>
        <tr><td>count</td><td>2111.00</td><td>2111.00</td><td>2111.00</td><td>2111.00</td><td>2111.00</td><td>2111.00</td><td>2111.00</td><td>2111.00</td></tr>
        <tr><td>mean</td><td>24.31</td><td>1.70</td><td>86.59</td><td>2.42</td><td>2.69</td><td>2.01</td><td>1.01</td><td>0.66</td></tr>
        <tr><td>std</td><td>6.35</td><td>0.09</td><td>26.19</td><td>0.53</td><td>0.78</td><td>0.61</td><td>0.85</td><td>0.61</td></tr>
        <tr><td>min</td><td>14.00</td><td>1.45</td><td>39.00</td><td>1.00</td><td>1.00</td><td>1.00</td><td>0.00</td><td>0.00</td></tr>
        <tr><td>25%</td><td>19.95</td><td>1.63</td><td>65.47</td><td>2.00</td><td>2.66</td><td>1.58</td><td>0.12</td><td>0.00</td></tr>
        <tr><td>50%</td><td>22.78</td><td>1.70</td><td>83.00</td><td>2.39</td><td>3.00</td><td>2.00</td><td>1.00</td><td>0.63</td></tr>
        <tr><td>75%</td><td>26.00</td><td>1.77</td><td>107.43</td><td>3.00</td><td>3.00</td><td>2.48</td><td>1.67</td><td>1.00</td></tr>
        <tr><td>max</td><td>61.00</td><td>1.98</td><td>173.00</td><td>3.00</td><td>4.00</td><td>3.00</td><td>3.00</td><td>2.00</td></tr>
      </tbody>
    </table>

  <p>Berikut adalah beberapa informasi penting dari hasil tersebut:</p>
    <ul>
      <li><strong>Mean</strong>: Menampilkan nilai rata-rata. Dari hasil tersebut dapat dilihat bahwa rata-rata usia berada pada usia 24 tahun, dengan tinggi rata-rata 1.70 cm dan rata-rata berat badan sebesar 86.59 kg</li>
      <li><strong>Min</strong>: Menampilkan nilai paling minimal. Dari hasil tersebut dapat dilihat bahwa usia minimal berada pada 14 tahun, dengan rata-rata tinggi 1.45 cm dan berat badan seberat 39 kg</li>
      <li><strong>Max</strong>: Menampilkan nilai paling maksimal. Dari hasil tersebut dapat dilihat bahwa usia maksimal berada pada 61 tahun, dengan rata-rata tinggi 1.98 cm dan berat badan seberat 173 kg</li>
    </ul>
  </li>
</ul>

<h4>2. Analisis Distribusi Tingkat Obesitas</h4>

Tahapan ini digunakan untuk memahami distribusi dari tiap kelas pada tingkat obesitas yang ada. Visualisasi dibuat menggunakan library matplotlib, dengan menghitung jumlah nilai dari tiap kelas target (NObeyesdad) terlebih dahulu dengan menggunakan `value_counts()`. Kemudian memvisualisasikannya dengan menggunakan `.plot(kind="bar")`. Warna dibedakan berdasarkan 4 jenis warna, yang dimulai dari warna biru tua pekat untuk kelas tingkat obesitas paling banyak, dan warna biru keabuan untuk 4 kelas tingkat obesitas paling remdah.

![image](https://github.com/user-attachments/assets/091c1645-665a-4488-981a-cc563cf0625f)

Dari hasil visualisasi di atas dengan tingkat obesitas dengan 3 jenis yang berbeda berada pada 3 peringkat teratas. Tingkat obesitas _Obesity_Type_I_ menjadi tipe obesitas dengan jumlah paling tinggi, dengan jumlah sekitar 350. Kemudian disusul oleh _Obesity_Type_III_ dan _Obesity_Type_II_ dengan jumlah masing masing nilai berada di kurang lebih 300 jumlah data. Berikut adalah hasil dari analisis pada tahapan ini.

<h4>2. Analisis Korelasi Antar Fitur Numerik</h4>

Tahapan ini dilakukan agar peneliti dapat memahami korelasi antar fitur numerik pada dataset yang akan digunakan tersebut. Analisis ini dilakukan dengan menggunakan `sns.heatmap` yang telah disediakan oleh library Seaborn. Adapun tahapan dari analisis korelasi antar fitur numerik ini dapat dibedakan berdasarkan 3 jenis, yakni -1, 0 dan 1. Nilai 1 dapat didefinisikan sebagai Korelasi Positif, nilai -1 dapat didefinisikan sebagai Korelasi Negatif, dan nilai 0 dapat didefinisikan sebagai tidak ada korelasi. Hasil dari analisis ini diharapkan mampu untuk memberikan gambaran antar kedua variabel numerik. Berikut adalah hasil dari analisis pada tahapan ini.

![image](https://github.com/user-attachments/assets/7c5fabb6-d890-4f35-b133-e102fc29acff)

<h4>3. Analisis Distribusi SMOKE</h4>

Tahapan ini dilakukan agar peneliti memiliki gambaran terkait bagaimana distribusi dari perokok pada dataset yang sedang digunakan. Seseorang yang tidak merokok sering kali lebih sering untuk mengonsumsi makanan berlebih, seperti mengonsumsi camilan yang tinggi kalori dan sebagainya. Tahapan ini dibuat dengan menggunakan `plt.pie` yang telah disediakan oleh library Matplotlib. Berikut adalah hasil dari analisis pada tahapan ini.

![image](https://github.com/user-attachments/assets/0b503486-4e83-4937-9887-07cfed7cbfdd)

<h4>4. Analisis Outlier pada Fitur Numerik</h4>

Tahapan ini dilakukan agar peneliti dapat mengidentifikasi outlier pada tiap fitur numerik yang ada. Peneliti menggunakan `sns.boxplot()` yang disediakan oleh library Seaborn untuk membantu mendapatkan gambaran berdasarkan visualisasi yang dihasilkan. Boxplot adalah alat visualisasi statistik yang digunakan untuk menunjukkan sebaran (distribusi), simpangan, dan pencilan (outlier) dalam set data numerik. Boxplot sangat berguna karena dapat memberikan ringkasan visual yang cepat dan informatif tentang distribusi dan outlier dari fitur numerik. berikut adalah hasil dari tahapan ini.

![image](https://github.com/user-attachments/assets/aebd1864-0c10-458b-b55d-cecb68acd663)


## Data Preparation

Tahapan ini bertujuan untuk mengubah data awal menjadi format yang dapat digunakan untuk pelatihan atau analisis model pembelajaran mesin.  Untuk menjamin bahwa data tersebut berkualitas, relevan, dan representatif, proses ini melibatkan langkah-langkah seperti pembersihan data (menangani nilai yang hilang atau duplikat), transformasi data (seperti normalisasi atau pengurangan dimensi dengan PCA), dan pembagian data (seperti split uji pelatihan).  Dengan mempersiapkan data, model dapat dilatih dengan lebih akurat, efektif, dan bebas dari bias atau gangguan, yang meningkatkan kinerja prediksi dan generalisasi.

### 1. Menghapus Duplikasi Data

Tahapan ini dilakukan untuk menghapud data yang terduplikasi pada dataset yang akan digunakan. Adanya duplikasi data pada dataset yang akan kita lakukan pemodelan dapat memberikan pengaruh yang kurang baik, sehingga dapat menyebabkan kurang optimal nya model yang kita miliki. penghapusan data dapat digunakan dengan menggunakan `drop_duplicates()`

![image](https://github.com/user-attachments/assets/45585f57-1f23-4e06-9df6-f69dda00088c)

Dari hasil tersebut, dapat dilihat bahwa jumlah data yang semulanya berjumlah 2111 baris, kini menjadi 2087 baris setelah melakukan penghapusan data. itu menandakan terdapat 24 jumlah baris yang terduplikasi pada dataset tingkat obesitas ini.

### 2. Menangani Outlier

Tahapan ini dilakukan dengan tujuan untuk menghapus outlier pada fitur-fitur numerik. Berdasarkan hasil dari analisis outlier dengan menggunakan boxplot sebelumnya, dapat diketahui bahwa fitur seperti Age, Height, Weight, dan NCP memiliki data yang terindikasi outlier. Namun setelah peneliti melakukan pengecekan pada nilai min dan max pada `df.describe()` sebelumnya, nilai dari Age dan NCP (frekuensi makanan utama per harinya) tersebut tidak mengindikasikan adanya data yang tidak normal. Jadi, peneiti memutuskan untuk menghapus outlier hanya pada fitur Height dan Weight. berikut adalah kodenya.

```
from scipy.stats.mstats import winsorize

outlier_handling_features = ['Height','Weight']
for feature in outlier_handling_features:
    df[feature] = winsorize(df[feature], limits=[0.01, 0.01])
``` 

Penerapan outlier handling diatas adalah dengan menggunakan metode winsorizing. Winsorizing adalah teknik statistik yang digunakan untuk mengurangi pengaruh outlier dengan menggantikan nilai-nilai ekstrem dalam dataset. Berbeda dengan metode trimming atau pemangkasan, data ekstrim pada winsorizing tidak dibuang melainkan diganti dengan nilai yang lebih moderat yang berada dalam persentil.

### 3. Memetakan Kelas Target Berdasarkan Tingkat Obesitas

Tahapan ini dilakukan agar kelas target dapat berubah menjadi nilai numerik berdasarkan dengan urutan dari tingkat obesitas nya. Hal ini dilakukan agar algoritma machine learning yang akan digunakan bisa langsung diproses dengan baik. Beberapa contoh algoritma tersebut adalah seperti RandomForestClassifier, SVC dan lain sebagainya. berikut ini adalah contoh kode dari pemetaan kelas target yang dibat dengan menggunakan dictionary dengan urutan 0 hingga 6.

```
obesity_map = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}
```

### 4. Memisahkan Kelas Fitur dan Kelas Target

Tahapan ini dilakukan agar kelas fitur dan kelas target tidak berada di dalam satu dataframe. Hal ini dilakuan agar pemodelan agar model dapat belajar dari data dengan cara yang terstruktur dan efektif. Kelas fitur akan dijadikan sebagai input dan kelas target akan dijadikan sebagai output dari modelnya. Berikut ini adalah kode dari pemisahan kelas target dan fitur.

```
X = df.drop(['NObeyesdad'], axis=1)
y = df['NObeyesdad']
```

### 5. Melatih data

Tahapan ini dilakukan agar dataset dapat dibagi menjadi dua bagian, yaitu data latih dan data uji dengan proporsi 80% untuk data latih dan 20% untuk data uji. Fungsi `train_test_split()` digunakan untuk agar fitur X dan target y dapat dipisahkan secara acak. Parameter `stratify=y` agar dapat memastikan distribusi kelas pada data pelatihan dan pengujian tetap seimbang, sesuai dengan distribusi kelas pada data asli, sehingga mencegah bias akibat ketidakseimbangan kelas. Parameter `random_state=123` digunakan untuk menetapkan seed acak, memastikan pembagian data konsisten dan dapat direproduksi pada setiap eksekusi kode. Berikut adalah kode dari tahapan ini.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=123)
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')
```

![image](https://github.com/user-attachments/assets/0d97023d-e755-461e-8de6-5e61ded7415c)

### 6. Normalisasi Data Numerik

Tahapan ini dilakukan agar data numerik yang digunakan berada pada skala yang sama. Pada tahapan ini peneliti menggunakan metode `StandardScaler().` Teknik preprocessing machine learning yang dikenal sebagai `StandardScaler()` digunakan untuk menstandarisasi fitur-fitur dalam kumpulan data dengan menghilangkan nilai rata-rata (mean) dan menskalakan data sehingga memiliki variansi satu (standar deviasi 1). Proses ini mengubah setiap fitur sehingga memiliki distribusi dengan rata-rata nol dan standar deviasi satu, yang dicapai dengan mengurangi nilai rata-rata setiap fitur dengan rata-rata tersebut dan kemudian membaginya dengan standar deviasi satu.

```
scaler = StandardScaler()
scaler.fit(X_train[numeric_features])
X_train[numeric_features] = scaler.transform(X_train.loc[:, numeric_features])
X_test[numeric_features] = scaler.transform(X_test.loc[:, numeric_features])
```

### 7. Encoding Data Kategorikal

Tahapan ini dilakukan agar data kategorikal diubah menjadi dalam bentuk numerik. Pada dataset yang akan saya gunakan, terdapat 2 jenis data kategorikal, yaitu binary categorical dan multi categorical. Pada fitur dengan binary categorical, `binary_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']`, fitur akan disimpan dalam suatu variabel biner, kemudian akan dilakukan encoding dengan menggunakan `LabelEncoder()`. `LabelEncoder` digunakan dengan tujuan untuk mengubah data kategorikal tersebut menjadi nilai 0 dan 1. berikut adalah kode dari tahapan ini.

```
label_encoder = LabelEncoder()

for feature in binary_features:
    X_train[feature] = label_encoder.fit_transform(X_train[feature])
    X_test[feature] = label_encoder.transform(X_test[feature])
```

Kemudian, langkah selanjutnya adalah melakukan encoding dengan menggunakan OneHotEncoder(). Teknik ini dilakukan pada fitur nominal_features = ['CAEC', 'CALC', 'MTRANS'] dengan tujuan agar setiap kategori diubah menjadi vektor biner, di mana hanya satu elemen yang bernilai 1 (menunjukkan kategori yang aktif), sementara elemen lainnya bernilai 0. Berikut adalah potongan kode pada langkah ini.

```
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_array = encoder.fit_transform(X_train[nominal_features])

encoded_df = pd.DataFrame(encoded_array, 
                          columns=encoder.get_feature_names_out(nominal_features),
                          index=X_train.index)

X_train = X_train.drop(columns=nominal_features).join(encoded_df)
```

```
encoded_test_array = encoder.transform(X_test[nominal_features])
encoded_test_df = pd.DataFrame(encoded_test_array,
                               columns=encoder.get_feature_names_out(nominal_features),
                               index=X_test.index)

X_test = X_test.drop(columns=nominal_features).join(encoded_test_df)
```

### 8. Principal Component Analysis

Tahapan ini dilakukan untuk mengurangi jumlah dari dimensi yang ada pada dataset. `PCA()` digunakan untuk mengurangi dimensi data dengan mempertahankan 95% varians, mengubah data pelatihan dan pengujian ke ruang fitur baru yang lebih ringkas untuk meningkatkan efisiensi dan mengurangi kompleksitas model machine learning.

```
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
```

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Berikut ini adalah ketiga opsi model yang akan dicoba untuk mengklasifikasikan tingkat obesitas.

### 1. Random Forest Classifier

Random Forest adalah algoritma pembelajaran mesin berbasis kelompok yang digunakan untuk tugas regresi dan klasifikasi. Algoritma ini menghasilkan output yang lebih akurat dan stabil dengan membangun banyak pohon keputusan secara acak dan menggabungkan hasil prediksi mereka. Berikut adalah contoh penerapan dari algoritma random forest classifier.

- Kelebihan:
   - Mudah digunakan dan tahan terhadap overfitting karena menggabungkan banyak pohon keputusan.
   - Dapat menangani data dengan banyak fitur atau kompleks tanpa perlu penskalaan data.
   - Memberikan estimasi pentingnya fitur untuk analisis data.
- Kekurangan:
   - Konsumsi memori tinggi untuk dataset besar karena banyaknya pohon.
   - Waktu prediksi relatif lambat dibandingkan algoritma sederhana.
   - Kurang efektif untuk data dengan pola sangat non-linear atau hubungan berurutan.

```
rf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)
```

Kode tersebut membuat model Random Forest Classifier yang memiliki `n_estimators=100` (yang membangun 100 pohon keputusan), `max_depth=10` (yang menentukan kedalaman maksimum setiap pohon yang dibatasi hingga 10 level), dan `random_state=42`.  Model dilatih pada label `y_train` dan data pelatihan X_train_pca, yang merupakan data fitur yang telah direduksi dimensinya dengan PCA, oleh fungsi `rf.fit(X_train_pca, y_train)`. Ini memungkinkan model untuk memprediksi kelas berdasarkan pola dalam data.

### 2. Gradient Boosting Classifier

Untuk tugas klasifikasi, Gradient Boosting Classifier adalah algoritma pembelajaran mesin berbasis kelompok.  Algoritma ini bekerja dengan membangun serangkaian pohon keputusan secara berurutan. Setiap pohon memperbaiki kesalahan prediksi dari pohon sebelumnya dengan menggunakan teknik penurunan gradient untuk meminimalkan fungsi kerugian.  Untuk menghasilkan output akhir yang lebih akurat, GBC menggabungkan prediksi dari semua pohon dengan bobot tertentu.  Meskipun algoritma ini efektif untuk data yang kompleks, ia sensitif terhadap perubahan parameter dan dapat memakan waktu komputasi yang lebih lama daripada Random Forest.

- Kelebihan:
   -Akurasi tinggi karena membangun pohon secara berurutan untuk memperbaiki kesalahan.
   -Efektif untuk dataset kompleks dengan pola non-linear.
   -Fleksibel dengan berbagai fungsi kerugian dan optimasi parameter.
-Kekurangan:
   -Sensitif terhadap parameter (misalnya, learning rate, jumlah pohon), risiko overfitting jika tidak disetel dengan baik.
   -Waktu pelatihan lebih lama dibandingkan Random Forest.
   -Rentan terhadap noise dalam data.

```
gbc = GradientBoostingClassifier(n_estimators=300,
                                 learning_rate=0.05,
                                 random_state=42,
                                 max_features=5 )
                                 
gbc.fit(X_train_pca, y_train)
```

Kode tersebut membuat model Classifier Gradient Boosting dengan `n_estimators=30` (membuat 300 pohon keputusan), random_state=42 (membuat tanaman acak untuk konsistensi), `learning_rate=0.05` (mengatur kontribusi setiap pohon terhadap prediksi akhir), dan `max_features=5` (membatasi jumlah fitur yang dipertimbangkan untuk setiap pohon menjadi 5).  Model dilatih dengan fungsi `gbc.fit(X_train_pca, y_train)` pada data pelatihan `X_train_pca`, yang merupakan data fitur yang telah direduksi dimensinya dengan PCA, dan label `y_train`. Ini memungkinkan model untuk memprediksi kelas berdasarkan pola dalam data.

### 3. Support Vector Classifier

Untuk menangani tugas klasifikasi, Support Vector Classifier (SVC) adalah algoritma pembelajaran mesin berbasis Support Vector Machine (SVM).  Hyperplane terbaik yang memisahkan kelas-kelas dalam data dengan margin terbesar—jarak terbesar antara hyperplane dan titik data (support vectors) terdekat—adalah cara SVC bekerja.  Ketika data tidak dapat dipisahkan secara linear, SVC menggunakan fungsi kernel (seperti fungsi basis radial atau "rbf") untuk memetakan mereka ke ruang dimensi yang lebih besar agar dapat dipisahkan.  Meskipun SVC efektif untuk dataset yang memiliki fitur yang kompleks, data besar dapat membutuhkan waktu komputasi yang lebih lama.

-Kelebihan:
   - Efektif untuk dataset dengan margin pemisahan jelas atau data non-linear (dengan kernel seperti RBF).
   - Cocok untuk data berdimensi tinggi dengan generalisasi yang baik.
   - Hasil konsisten dengan margin maksimum.
-Kekurangan:
   - Skalabilitas buruk untuk dataset besar karena kompleksitas komputasi tinggi.
   - Memerlukan penskalaan data dan sensitif terhadap ketidakseimbangan kelas.
   - Sulit diinterpretasikan dibandingkan algoritma berbasis pohon.

```
svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
svm_clf.fit(X_train_pca, y_train)
```

Kode tersebut membuat model SVC dengan `kernel='rbf'`, `probability=True` (mengaktifkan estimasi probabilitas untuk prediksi kelas), dan `random_state=42`.  Model dilatih pada data pelatihan `X_train_pca` (data fitur yang telah direduksi dimensinya dengan PCA) dan label `y_train` oleh fungsi `svm_clf.fit(X_train_pca, y_train)`. Ini memungkinkan model untuk memprediksi kelas berdasarkan pola dalam data.

### Pemilihan Best Model

Setelah melakukan pemodelan dengan menggunakan algoritma machine laerning yang akan digunakan, berikut ini adalah algoritma terbaik yang akan digunakan.

![image](https://github.com/user-attachments/assets/f9b0f124-8a0e-4a55-aadc-52b428f481ab)
![image](https://github.com/user-attachments/assets/dddd17e9-bc20-46ac-85a8-dc0563f78818)
![image](https://github.com/user-attachments/assets/3326a2a6-583e-4323-aac7-6a61d57e50ce)

Dari hasil diatas, algoritma Random Forest memiliki train accuracy sebesar 1.0000 dan test accuracy sebesar 0.8756, yang menandakan bahwa model tersebut overfit. Kemudian untuk model GBC, memiliki train accuracy sebesar 1.0000 dan test accuracy 0.8373 yang menandakan bahwa model ini juga overfit. Lalu untuk model SVC memiliki train accuracy sebesar 0.9700 dan test accuracy 0.9115, yang menandakan bahwa model ini terlatih dan teruji dengan sangat baik. Model tidak mengalami overfit ataupun underfit.

![image](https://github.com/user-attachments/assets/9b0bb50a-feca-4af5-990b-f28c5e89810e)

Visualisasi diatas merupakan visualisasi dari confusion matrix dari model Support Vector Classifier. Bagian kiri atas dari visualisasi di atas menunjukkan data negatif yang diprediksi dengan benar (TN), sedangkan bagian kanan bawah menunjukkan data positif yang diprediksi dengan benar. Sebaliknya, bagian kanan atas menunjukkan data negatif yang salah diprediksi sebagai negatif, dan bagian kiri bawah menunjukkan data negatif yang salah diprediksi sebagai positif.

## Evaluation

Setelah menemukan model terbaik yang akan digunakan, yaitu Support Vector Nachine Classifier. Langkah selanjutnya adalah dengan melakukan evaluasi terhadap hasil dari accuracy, precision, recall, dan f1-score pada model tersebut.

![image](https://github.com/user-attachments/assets/3326a2a6-583e-4323-aac7-6a61d57e50ce)

### Penjelasan Matriks dan Formula Matriks

- Accuracy

Accuracy adalah ukuran yang menunjukkan seberapa banyak prediksi yang benar—baik positif maupun negatif—dari semua prediksi yang dibuat oleh model.  Dalam konteks gambar, nilai ketepatan kereta adalah 0,9700 (97%) dan ketepatan ujian adalah 0,9115 (91,15%), yang menunjukkan bahwa model SVC memiliki kinerja yang luar biasa baik pada data pelatihan maupun pengujian. Formula dari matriks accuracy adalah, `Akurasi = (TP + TN ) / (TP+FP+FN+TN)`, di mana TP adalah nilai positif asli, TN adalah nilai negatif asli, FP adalah nilai positif palsu, dan FN adalah nilai negatif palsu.  Accuracy memberi gambaran umum tentang kinerja model, tetapi kurang bermanfaat jika data menunjukkan ketidakseimbangan kelas.

- Precision

Precision mengukur seberapa akurat prediksi positif yang diberikan model, yaitu proporsi instance positif yang benar dari total instance yang diprediksi sebagai positif. Dari gambar, nilai precision keseluruhan adalah 0.9151 (91.51%), dengan detail per kelas seperti 0.96 (kelas 0), 0.76 (kelas 1), hingga 1.00 (kelas 6 dan 7), menunjukkan variasi performa antar kelas. Formula-nya adalah `Precission = (TP) / (TP+FP)` yang fokus pada proporsi positif yang benar dari total prediksi positif.

- Recall

Recall (atau sensitivity) mengukur seberapa banyak instance positif yang berhasil dideteksi oleh model dari semua instance positif yang ada. Dalam gambar, recall keseluruhan adalah 0.9115, artinya model berhasil mendeteksi 91.15% dari semua instance positif yang sebenarnya. Per kelas (misalnya, kelas 0: 0.91, kelas 1: 0.82), recall menunjukkan kemampuan model menemukan instance dari kelas tersebut. Formula: `Recall = (TP) / (TP + FN)` yang membandingkan TP dengan total positif aktual untuk meminimalkan false negatives.

- F1-Score

F1-Score adalah rata-rata harmonik dari precision dan recall, memberikan keseimbangan antara keduanya. Dalam gambar, F1-score keseluruhan adalah 0.9127, menunjukkan performa seimbang antara precision dan recall. Per kelas (misalnya, kelas 0: 0.93, kelas 1: 0.79), F1-score membantu mengevaluasi performa model untuk setiap kelas. Formula: `F1 Score = 2 * (Recall*Precission) / (Recall + Precission)` yang menggabungkan kedua metrik untuk skor tunggal, berguna saat distribusi kelas tidak merata.

### Hasil Evaluasi

Hasil evaluasi model Support Vector Classifier (SVC) dalam laporan proyek menunjukkan performa yang sangat baik dalam mengklasifikasikan tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik.

Dengan akurasi 0,9700 (97%) dan 0,9115 (91,15%), model SVC menunjukkan kemampuan untuk memprediksi dengan akurat data pelatihan dan pengujian.  Model dapat digeneralisasikan dengan baik pada data baru karena tidak mengalami overfitting atau underfitting, seperti yang ditunjukkan oleh perbedaan kecil antara ketepatan pengajaran dan pengujian.  Selain itu, precision keseluruhan sebesar 0.9151 (91.51%) menunjukkan bahwa 91.51% dari prediksi positif model benar, sedangkan recall sebesar 0.9115 (91.15%) menunjukkan bahwa model berhasil menemukan 91.15% dari semua kasus positif yang ada.  Bahkan pada dataset dengan distribusi kelas yang berbeda, model beroperasi secara konsisten, menunjukkan F1-score sebesar 0,9127 (91.27%), yang menunjukkan keseimbangan yang baik antara precision dan recall.

Model SVC terbukti lebih baik daripada Random Forest (dengan test accuracy 0.8756, overfitting) dan Gradient Boosting Classifier (dengan test accuracy 0.8373, juga overfitting).  Performa SVC yang konsisten di semua kelas menunjukkan bahwa model ini dapat diandalkan untuk mengklasifikasikan dengan akurat tingkat obesitas. Ini mendukung tujuan proyek untuk membantu merancang intervensi gaya hidup yang dapat disesuaikan untuk menurunkan risiko obesitas dengan membuat intervensi yang disesuaikan.  Performa tidak terpengaruh secara signifikan oleh distribusi kelas yang tidak merata (support berkisar dari 53 hingga 70). Ini karena metode seperti stratifikasi saat pembagian data dan PCA untuk reduksi dimensi, yang membantu model menemukan pola penting tanpa bias.




