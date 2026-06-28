# 📊 SimKebijakan — UAS Edition (M16)
**Simulator Interaktif What-If | Pemodelan & Simulasi**
Mohamad Yusuf · NPM 2313020037

---

## Deskripsi Proyek
SimKebijakan adalah simulator kebijakan bisnis berbasis data yang mengintegrasikan:
- **Machine Learning** (Linear Regression) untuk prediksi profit
- **XAI** (Explainability berbasis koefisien) untuk transparansi keputusan
- **SPK Metode SAW** untuk ranking alternatif strategi
- **MLOps**: deteksi drift, serialisasi model, anonymization data PII

---

## Struktur Repositori
```
├── app.py                      # Aplikasi utama Streamlit
├── requirements.txt            # Library + versi spesifik
├── model_kebijakan_v1.joblib   # Model ML (auto-generated saat pertama run)
├── scaler_kebijakan_v1.joblib  # Scaler (auto-generated saat pertama run)
└── README.md
```

---

## Cara Menjalankan

### 1. Clone / Download repositori
```bash
git clone https://github.com/mohamadyusuf17/simulator-kebijakan.git
cd simulator-kebijakan
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi
```bash
streamlit run app.py
```

### 4. Buka di browser
Streamlit akan otomatis membuka `http://localhost:8501`

---

## Alur Data (Pipeline)
```
Slider Input (Iklan, Diskon)
        ↓
StandardScaler.transform()
        ↓
LinearRegression.predict()  →  Profit Prediksi
        ↓                              ↓
Deteksi Drift              XAI: Kontribusi Fitur
                                       ↓
                           Matriks SAW [Profit, Efisiensi, Biaya]
                                       ↓
                           Normalisasi + Pembobotan AHP
                                       ↓
                               Ranking Alternatif
```

---

## Catatan Teknis
- File `.joblib` di-generate otomatis pada **run pertama** — tidak perlu upload manual
- Jika ingin reset model, hapus kedua file `.joblib` lalu restart app
- Threshold drift: **15.0** (sesuai skala slider 0–50)

---

## Deployment
Aplikasi ini dapat diakses di:
🔗 https://mohamadyusuf0037simulator-m14.streamlit.app
