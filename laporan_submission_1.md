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

- Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
- Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

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

Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:

- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
