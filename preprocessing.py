import pandas as pd
import re
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from typing import List
import math

class TextPreprocessor:
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        
        self.custom_stopwords = {
            'com', 'www', 'http', 'https', 'html', 'jpg', 'png', 'jpeg',
            'detik', 'kompas', 'tempo', 'cnn', 'liputan', 'republika'
        }
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'@\w+|#\w+', '', text)
        
        text = re.sub(r'\d+', '', text)
        
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        return text.split()
    
    def remove_stopwords(self, tokens):
        text = ' '.join(tokens)
        text = self.stopword_remover.remove(text)
        tokens = text.split()
        
        tokens = [t for t in tokens if t not in self.custom_stopwords]
        
        tokens = [t for t in tokens if len(t) >= 3]
        
        return tokens
    
    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text):
        text = self.clean_text(text)
        
        tokens = self.tokenize(text)
        
        tokens = self.remove_stopwords(tokens)
        
        tokens = self.stem_tokens(tokens)
        
        return tokens

    def calculate_tfidf_score(self, query_tokens: List[str], doc_id: int) -> float:
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0)
        
        if doc_length == 0:
            return 0.0
        
        for term in query_tokens:
            if term in self.index:
                tf = self.get_term_frequency(term, doc_id) / doc_length
                
                df = self.get_document_frequency(term)
                idf = math.log(self.num_docs / df) if df > 0 else 0
                
                score += tf * idf
        
        return score

    def search_tfidf(self, query_tokens: List[str], top_k: int = 10) -> List[tuple]:
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
    print("🔄 Memulai preprocessing corpus...")
    
    # Load data
    df = pd.read_csv("corpus_wni_kamboja_update.csv")

    print(f"📊 Total dokumen: {len(df)}")
    
    preprocessor = TextPreprocessor()
    
    processed_data = []
    
    for idx, row in df.iterrows():
        full_text = f"{row['title']} {row['content']}"
        
        tokens = preprocessor.preprocess(full_text)
        
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
    
    df_processed = pd.DataFrame(processed_data)
    
    print(f"\n📈 Statistik Preprocessing:")
    print(f"Total dokumen: {len(df_processed)}")
    print(f"Rata-rata token per dokumen: {df_processed['token_count'].mean():.2f}")
    print(f"Min token: {df_processed['token_count'].min()}")
    print(f"Max token: {df_processed['token_count'].max()}")
    
    df_csv = df_processed.drop(columns=['tokens'])
    df_csv.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 CSV disimpan: {output_csv}")
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON disimpan: {output_json}")
    
    print("\n🎉 Preprocessing selesai!")
    return df_processed

if __name__ == "__main__":
    input_file = 'corpus_wni_kamboja_update.csv'
    output_csv = 'preprocessed_corpus.csv'
    output_json = 'preprocessed_corpus.json'
    
    df_result = preprocess_corpus(input_file, output_csv, output_json)
    
    print("\n📄 Contoh hasil preprocessing (3 dokumen pertama):")
    for idx in range(min(3, len(df_result))):
        print(f"\n[Dokumen {idx}]")
        print(f"Title: {df_result.iloc[idx]['title'][:80]}...")
        print(f"Token count: {df_result.iloc[idx]['token_count']}")
        print(f"First 10 tokens: {df_result.iloc[idx]['tokens'][:10]}")