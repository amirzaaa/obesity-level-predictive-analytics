# Laporan Proyek Machine Learning - Amir Hamzah

## Domain Proyek

Domain proyek yang saya pilih pada proyek ini adalah tentang tema kesehatan dengan judul proyek "Klasifikasi Tingkat Obesitas Berdasarkan Kebiasaan Makan dan Kondisi Fisik"

### Latar Belakang

Kelebihan berat badan, juga dikenal sebagai obesitas, adalah kondisi di mana seseorang mengalami sejumlah anomali dan lemak berlebih, yang merupakan salah satu faktor risiko kesehatan yang signifikan. Standar yang ditetapkan oleh Organisasi Kesehatan Dunia (WHO) untuk batas berat badan diukur dengan menggunakan indeks massa tubuh (BMI), di mana nilai 25 atau lebih dianggap kelebihan berat badan dan nilai 30 atau lebih dianggap obesitas [1](https://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2443). Menurut Fabio, ada tujuh level
dalam obesitas diantaranya Insufficient Weight,
Normal Weight, Overweight Level I, Overweight
Level II, Obesity Type I, Obesity Type II, dan
Obesity Type III [2](https://pdfs.semanticscholar.org/b5f0/8012dfee726d261da0cb0758ca29d3276111.pdf).

Menurut hasil penelitian kesehatan dasar yang dilakukan pada tahun 2018, tingkat obesitas pada individu berusia lebih dari 18 tahun meningkat dari 14,8% menjadi 21,8%. Obesitas juga dapat meningkatkan risiko penyakit jantung dan stroke [3](https://ejournal.pelitaindonesia.ac.id/ojs32/index.php/JOISIE/article/view/2467/1009). Obesitas adalah masalah kesehatan yang signifikan di seluruh dunia dengan dampak yang signifikan. Dalam beberapa dekade terakhir, obesitas menjadi masalah kesehatan masyarakat utama di Meksiko. Di Meksiko, tingkat obesitas dewasa mencapai 74,5% wanita dan 69,1% pria [4](https://proceeding.unpkediri.ac.id/index.php/inotek/article/view/5062). Data WHO terbaru menunjukkan bahwa pada tahun 2022, 2,5 miliar orang dewasa berusia 18 tahun ke atas mengalami kelebihan berat badan, dengan 890 juta termasuk dalam kategori obesitas, dan jika tingkat pertumbuhan ini terus berlanjut, proporsi orang yang masuk dalam kategori obesitas di Indonesia akan meningkat menjadi 2 miliar pada tahun 2030. Survei Kesehatan Indonesia pada tahun 2023 menunjukkan bahwa jumlah orang yang masuk dalam kategori obesitas di Indonesia akan meningkat menjadi 2 miliar [5](https://mail.ejournal.itn.ac.id/index.php/jati/article/view/13397/7526).

Oleh karena itu, memahami dan mengklasifikasikan tingkat obesitas sangat penting untuk mencegah dan menangani obesitas. Penelitian ini menggunakan algoritma machine learning untuk melakukan klasifikasi terhadap tingkat obesitas pada individu dengan menggunakan data mengenai kebiasaan makan, aktivitas fisik, dan variabel lainnya yang terkait dengan tingkat obesitas. Tujuan dari metode ini adalah untuk memahami dan mengidentifikasi faktor-faktor yang memengaruhi tingkat obesitas [6](https://ojs3.unpatti.ac.id/index.php/parameter/article/view/11875). Dataset yang akan digunakan adalah dengan menggunakan dataset yang didapat dari website UCI Machine Learning Repository terkait _Estimation of Obesity Levels Based On Eating Habits and Physical Condition_ [7](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).

## Referensi

- [1] Sitanggang, D., & Sherly, S. (2022). Model Prediksi Obesitas dengan Menggunakan Support Vector Machine. Jurnal Sistem Informasi dan Ilmu Komputer, 5(2), 172-175.
- [2] Setiyani, L., Indahsari, A. N., & Roestam, R. (2023). Analisis Prediksi Level Obesitas Menggunakan Perbandingan Algoritma Machine Learning dan Deep Learning. JTERA (Jurnal Teknol. Rekayasa), 8(1), 139.
- [3] Wie, J. V., & Siddik, M. (2023). Penerapan Metode Naïve Bayes Dalam Mengklasifikasi Tingkat Obesitas Pada Pria. JOISIE (Journal Of Information Systems And Informatics Engineering), 6(2), 69-77.
- [4] Aini, E. D. N., Khasanah, R. A., Ristyawan, A., & Diniati, E. (2024, July). Penggunaan Data Mining untuk Prediksi tingkat Obesitas di Meksiko Menggunakan Metode Random Forest. In Prosiding SEMNAS INOTEK (Seminar Nasional Inovasi Teknologi) (Vol. 8, No. 3, pp. 1256-1265).
- [5] Khikam, A., Anggadimas, N. M., & Udin, M. (2025). IMPLEMENTASI DECISION TREE UNTUK KLASIKASI OBESITAS. JATI (Jurnal Mahasiswa Teknik Informatika), 9(3), 3946-3952.
- [6] Fitriani, D. N. (2024). Prediksi PREDIKSI TINGKAT OBESITAS MENGGUNAKAN NEURAL NETWORK: PENDEKATAN KLASIFIKASI BINER. PARAMETER: Jurnal Matematika, Statistika dan Terapannya, 3(01), 85-92.
- [7] _Estimation of Obesity Levels Based On Eating Habits and Physical Condition_, UCI Machine Learning Repository.

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
| CH2O                           | Kontinu     | Berapa banyak air yang dikonsumsi setiap hari?                   |
| SCC                            | Biner       | Apakah memantau kalori yang dikonsumsi setiap hari?              |
| FAF                            | Kontinu     | Seberapa sering melakukan aktivitas fisik?                       |
| TUE                            | Integer     | Waktu yang digunakan dengan perangkat teknologi (TV, HP, dll.)   |
| CALC                           | Kategorikal | Seberapa sering mengonsumsi alkohol?                             |
| MTRANS                         | Kategorikal | Moda transportasi yang biasa digunakan                           |
| NObeyesdad                     | Kategorikal | Tingkat obesitas                                                 |

### Explanatory Data Analysis

Agar dapat lebih memahami terhadap dataset yang akan digunakan, berikut ini adalah tahapan eksplorasi data yang peneliti lakukan:

1. Memahami Informasi Dataset

berikut adalah beberapa hal yang peneliti lakukan untuk memahami dataset yang digunakan:

- df.head()

Digunakan untuk menampilkan data dalam dataframe sebanyak 5 baris. berikut adalah hasilnya:

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
