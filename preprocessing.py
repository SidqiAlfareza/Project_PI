import pandas as pd
import re
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from typing import List
import math

class TextPreprocessor:
    def __init__(self):
        # Inisialisasi Sastrawi
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        
        # Stopwords tambahan untuk domain berita
        self.custom_stopwords = {
            'com', 'www', 'http', 'https', 'html', 'jpg', 'png', 'jpeg',
            'detik', 'kompas', 'tempo', 'cnn', 'liputan', 'republika'
        }
    
    def clean_text(self, text):
        """Pembersihan teks dasar"""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Hapus URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Hapus email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Hapus mention dan hashtag
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Hapus angka
        text = re.sub(r'\d+', '', text)
        
        # Hapus karakter khusus, hanya simpan huruf dan spasi
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Hapus spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenisasi sederhana"""
        return text.split()
    
    def remove_stopwords(self, tokens):
        """Hapus stopwords"""
        # Gunakan Sastrawi stopwords
        text = ' '.join(tokens)
        text = self.stopword_remover.remove(text)
        tokens = text.split()
        
        # Hapus custom stopwords
        tokens = [t for t in tokens if t not in self.custom_stopwords]
        
        # Hapus token pendek (< 3 karakter)
        tokens = [t for t in tokens if len(t) >= 3]
        
        return tokens
    
    def stem_tokens(self, tokens):
        """Stemming menggunakan Sastrawi"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text):
        """Pipeline preprocessing lengkap"""
        # 1. Clean
        text = self.clean_text(text)
        
        # 2. Tokenize
        tokens = self.tokenize(text)
        
        # 3. Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # 4. Stemming
        tokens = self.stem_tokens(tokens)
        
        return tokens

    def calculate_tfidf_score(self, query_tokens: List[str], doc_id: int) -> float:
        """
        Menghitung TF-IDF score untuk dokumen terhadap query
        """
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0)
        
        if doc_length == 0:
            return 0.0
        
        for term in query_tokens:
            if term in self.index:
                # TF: Term Frequency (normalized)
                tf = self.get_term_frequency(term, doc_id) / doc_length
                
                # IDF: Inverse Document Frequency
                df = self.get_document_frequency(term)
                idf = math.log(self.num_docs / df) if df > 0 else 0
                
                # TF-IDF
                score += tf * idf
        
        return score

    def search_tfidf(self, query_tokens: List[str], top_k: int = 10) -> List[tuple]:
        """
        Pencarian menggunakan TF-IDF
        """
        candidate_docs: Set[int] = set()
        for term in query_tokens:
            if term in self.index:
                candidate_docs.update(self.index[term].keys())
        
        scores = []
        for doc_id in candidate_docs:
            score = self.calculate_tfidf_score(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

def preprocess_corpus(input_file, output_csv, output_json):
    """
    Preprocessing corpus berita WNI Kamboja
    """
    print("🔄 Memulai preprocessing corpus...")
    
    # Load data
    df = pd.read_csv("corpus_wni_kamboja_update.csv")

    print(f"📊 Total dokumen: {len(df)}")
    
    # Inisialisasi preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocessing
    processed_data = []
    
    for idx, row in df.iterrows():
        # Gabungkan title dan content
        full_text = f"{row['title']} {row['content']}"
        
        # Preprocess
        tokens = preprocessor.preprocess(full_text)
        
        # Simpan hasil
        processed_data.append({
            'id': idx,
            'title': row['title'],
            'url': row['url'],
            'source': row['source'],
            'original_content': row['content'],
            'tokens': tokens,
            'processed_text': ' '.join(tokens),
            'token_count': len(tokens)
        })
        
        if (idx + 1) % 50 == 0:
            print(f"   ✅ Diproses: {idx + 1}/{len(df)} dokumen")
    
    # Buat DataFrame
    df_processed = pd.DataFrame(processed_data)
    
    # Statistik
    print(f"\n📈 Statistik Preprocessing:")
    print(f"Total dokumen: {len(df_processed)}")
    print(f"Rata-rata token per dokumen: {df_processed['token_count'].mean():.2f}")
    print(f"Min token: {df_processed['token_count'].min()}")
    print(f"Max token: {df_processed['token_count'].max()}")
    
    # Simpan CSV (tanpa kolom tokens yang besar)
    df_csv = df_processed.drop(columns=['tokens'])
    df_csv.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 CSV disimpan: {output_csv}")
    
    # Simpan JSON lengkap (dengan tokens)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON disimpan: {output_json}")
    
    print("\n🎉 Preprocessing selesai!")
    return df_processed

# JALANKAN PREPROCESSING
if __name__ == "__main__":
    input_file = 'corpus_wni_kamboja_update.csv'
    output_csv = 'preprocessed_corpus.csv'
    output_json = 'preprocessed_corpus.json'
    
    df_result = preprocess_corpus(input_file, output_csv, output_json)
    
    # Tampilkan contoh hasil
    print("\n📄 Contoh hasil preprocessing (3 dokumen pertama):")
    for idx in range(min(3, len(df_result))):
        print(f"\n[Dokumen {idx}]")
        print(f"Title: {df_result.iloc[idx]['title'][:80]}...")
        print(f"Token count: {df_result.iloc[idx]['token_count']}")
        print(f"First 10 tokens: {df_result.iloc[idx]['tokens'][:10]}")