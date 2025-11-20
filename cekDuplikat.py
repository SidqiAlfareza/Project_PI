from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd


df = pd.read_csv("corpus_wni_kamboja_update.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['content'])

similarity_matrix = cosine_similarity(X)

# Ambil pasangan dokumen yang mirip (misalnya similarity > 0.9)
import numpy as np
pairs = []
for i in range(len(df)):
    for j in range(i+1, len(df)):
        if similarity_matrix[i, j] > 0.9:
            pairs.append((i, j, similarity_matrix[i, j]))

def hapus_duplikat(df, duplicate_pairs, threshold=0.90):
    """
    Menghapus dokumen duplikat berdasarkan pasangan yang ditemukan
    """
    # Kumpulkan indeks yang perlu dihapus
    to_remove = set()
    
    for idx1, idx2, similarity in duplicate_pairs:
        if similarity >= threshold:
            # Simpan indeks yang lebih kecil, hapus yang lebih besar
            to_remove.add(max(idx1, idx2))
    
    print(f"\n📊 Statistik Duplikasi:")
    print(f"Total pasangan duplikat: {len(duplicate_pairs)}")
    print(f"Dokumen yang akan dihapus: {len(to_remove)}")
    print(f"Indeks yang dihapus: {sorted(to_remove)}")
    
    # Buat dataframe bersih dengan menghapus indeks duplikat
    df_clean = df.drop(index=list(to_remove)).reset_index(drop=True)
    
    print(f"\n✅ Dataset Bersih:")
    print(f"Jumlah dokumen awal: {len(df)}")
    print(f"Jumlah dokumen setelah cleaning: {len(df_clean)}")
    print(f"Dokumen yang dihapus: {len(df) - len(df_clean)}")
    
    return df_clean

print(pairs)

# Setelah menemukan duplicate_pairs, tambahkan kode ini:
duplicate_pairs = pairs
# Hapus duplikat
df_clean = hapus_duplikat(df, duplicate_pairs, threshold=0.90)

# Simpan ke CSV
output_csv = 'clean_corpus_wni_kamboja_multisource.csv'
df_clean.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n💾 CSV bersih disimpan: {output_csv}")

# Simpan ke JSON
output_json = 'clean_corpus_wni_kamboja_multisource.json'
df_clean.to_json(output_json, orient='records', force_ascii=False, indent=2)
print(f"💾 JSON bersih disimpan: {output_json}")

print(f"\n🎉 Proses cleaning selesai!")
print(f"📂 File output:")
print(f"   - {output_csv}")
print(f"   - {output_json}")