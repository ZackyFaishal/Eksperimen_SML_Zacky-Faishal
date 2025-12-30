import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def preprocess_personality_data(
    input_path: str,
    output_path: str = "personality_preprocessing_dataset.csv"
):
    """
    Melakukan preprocessing otomatis pada dataset kepribadian
    (Extrovert vs Introvert) agar siap digunakan untuk model machine learning.

    Tahapan yang dilakukan:
    1. Menghapus missing values
    2. Menghapus data duplikat
    3. Normalisasi fitur numerik (Min-Max Scaling)
    4. Deteksi dan penghapusan outlier (IQR)
    5. Encoding data kategorikal
    6. Binning fitur numerik tertentu

    Parameters
    ----------
    input_path : str
        Path ke file CSV dataset mentah.
    output_path : str, optional
        Path penyimpanan file hasil preprocessing.
    """

    # =========================
    # Load dataset
    # =========================
    df = pd.read_csv(input_path)
    print(f"📥 Dataset dimuat: {df.shape}")

    # =========================
    # 1. Hapus missing values
    # =========================
    df_clean = df.dropna()
    print(f"🧹 Setelah hapus missing values: {df_clean.shape}")

    # =========================
    # 2. Hapus data duplikat
    # =========================
    df_clean = df_clean.drop_duplicates()
    print(f"🧹 Setelah hapus duplikat: {df_clean.shape}")

    # =========================
    # 3. Normalisasi fitur numerik
    # =========================
    num_cols = [
        'Time_spent_Alone',
        'Social_event_attendance',
        'Going_outside',
        'Friends_circle_size',
        'Post_frequency'
    ]

    scaler = MinMaxScaler()
    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
    print("📊 Normalisasi fitur numerik selesai")

    # =========================
    # 4. Deteksi & hapus outlier (IQR)
    # =========================
    for col in num_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df_clean = df_clean[
            (df_clean[col] >= lower) & (df_clean[col] <= upper)
        ]

    print(f"📉 Setelah penanganan outlier: {df_clean.shape}")

    # =========================
    # 5. Encoding data kategorikal
    # =========================
    cat_cols = [
        'Stage_fear',
        'Drained_after_socializing',
        'Personality'
    ]

    label_encoders = {}

    for col in cat_cols:
        encoder = LabelEncoder()
        df_clean[col] = encoder.fit_transform(df_clean[col])
        label_encoders[col] = encoder

    print("🔤 Encoding fitur kategorikal selesai")

    # =========================
    # 6. Binning (pengelompokan data)
    # =========================
    df_clean['Time_spent_Alone_Bin'] = pd.cut(
        df_clean['Time_spent_Alone'],
        bins=3,
        labels=['Rendah', 'Sedang', 'Tinggi']
    )

    # Encode hasil binning agar numerik
    bin_encoder = LabelEncoder()
    df_clean['Time_spent_Alone_Bin'] = bin_encoder.fit_transform(
        df_clean['Time_spent_Alone_Bin']
    )

    print("📦 Binning fitur Time_spent_Alone selesai")

    # =========================
    # Simpan hasil preprocessing
    # =========================
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Data preprocessing selesai. File disimpan di: {output_path}")

    return df_clean


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated preprocessing for Extrovert vs Introvert dataset"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path ke file dataset mentah (CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="personality_preprocessing_dataset.csv",
        help="Path file output hasil preprocessing"
    )

    args = parser.parse_args()

    preprocess_personality_data(
        input_path=args.input_path,
        output_path=args.output
    )
