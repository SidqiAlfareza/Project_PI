import json
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Set
import math

class InvertedIndex:
    def __init__(self):
        self.index: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.doc_lengths: Dict[int, int] = {}
        self.num_docs = 0
        self.avg_doc_length = 0
        
    def build_index(self, json_file: str):
        """
        Membangun inverted index dari file JSON hasil preprocessing
        """
        print("📚 Membangun inverted index...")
        
        # Load data
        with open(json_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        self.num_docs = len(documents)
        total_length = 0
        
        # Build index
        for doc in documents:
            doc_id = doc['id']
            tokens = doc['tokens']
            
            # Simpan panjang dokumen
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # Build inverted index dengan term frequency
            for token in tokens:
                self.index[token][doc_id] += 1
        
        # Hitung rata-rata panjang dokumen
        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0
        
        print(f"✅ Index berhasil dibangun!")
        print(f"   Total unique terms: {len(self.index)}")
        print(f"   Total documents: {self.num_docs}")
        print(f"   Average document length: {self.avg_doc_length:.2f} tokens")
        
    def get_document_frequency(self, term: str) -> int:
        """
        Mendapatkan jumlah dokumen yang mengandung term
        """
        return len(self.index.get(term, {}))
    
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """
        Mendapatkan frekuensi term dalam dokumen tertentu
        """
        return self.index.get(term, {}).get(doc_id, 0)
    
    def get_posting_list(self, term: str) -> Dict[int, int]:
        """
        Mendapatkan posting list untuk term tertentu
        """
        return dict(self.index.get(term, {}))
    
    def calculate_idf(self, term: str) -> float:
        """
        Menghitung IDF (Inverse Document Frequency)
        IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        """
        df = self.get_document_frequency(term)
        if df == 0:
            return 0.0
        
        # BM25 IDF formula
        idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)
        return idf
    
    def calculate_bm25_score(self, query_tokens: List[str], doc_id: int, 
                            k1: float = 1.5, b: float = 0.75) -> float:
        """
        Menghitung BM25 score untuk dokumen terhadap query
        
        Parameters:
        - k1: parameter untuk term frequency saturation (default: 1.5)
        - b: parameter untuk length normalization (default: 0.75)
        """
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0)
        
        if doc_length == 0:
            return 0.0
        
        for term in query_tokens:
            if term in self.index:
                tf = self.get_term_frequency(term, doc_id)
                idf = self.calculate_idf(term)
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
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
    
    def search(self, query_tokens: List[str], top_k: int = 10) -> List[tuple]:
        """
        Melakukan pencarian menggunakan BM25
        
        Returns:
        List of tuples (doc_id, score) sorted by score descending
        """
        # Ambil dokumen yang mengandung setidaknya satu term dari query
        candidate_docs: Set[int] = set()
        for term in query_tokens:
            if term in self.index:
                candidate_docs.update(self.index[term].keys())
        
        # Hitung score untuk setiap dokumen kandidat
        scores = []
        for doc_id in candidate_docs:
            score = self.calculate_bm25_score(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort berdasarkan score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def search_tfidf(self, query_tokens: List[str], top_k: int = 10) -> List[tuple]:
        """
        Pencarian menggunakan TF-IDF
        
        Args:
            query_tokens: List token dari query
            top_k: Jumlah dokumen teratas
            
        Returns:
            List of tuples (doc_id, score) sorted by score descending
        """
        # Ambil dokumen yang mengandung setidaknya satu term dari query
        candidate_docs: Set[int] = set()
        for term in query_tokens:
            if term in self.index:
                candidate_docs.update(self.index[term].keys())
        
        # Hitung score untuk setiap dokumen kandidat
        scores = []
        for doc_id in candidate_docs:
            score = self.calculate_tfidf_score(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort berdasarkan score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def save_index(self, pkl_file: str, txt_file: str = None, json_file: str = None):
        """
        Menyimpan index dalam 3 format: PKL (utama), TXT (debug), JSON (backup)
        
        Args:
            pkl_file: Path untuk file pickle (wajib)
            txt_file: Path untuk file text (opsional)
            json_file: Path untuk file JSON (opsional)
        """
        index_data = {
            'index': dict(self.index),
            'doc_lengths': self.doc_lengths,
            'num_docs': self.num_docs,
            'avg_doc_length': self.avg_doc_length
        }
        
        # 1. SIMPAN PICKLE (Production - fastest)
        with open(pkl_file, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"💾 Pickle index disimpan: {pkl_file}")
        
        # 2. SIMPAN TEXT (Human-readable - for debugging)
        if txt_file:
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("INVERTED INDEX - TEXT FORMAT\n")
                f.write("="*80 + "\n\n")
                
                # Metadata
                f.write(f"Total Documents: {self.num_docs}\n")
                f.write(f"Total Unique Terms: {len(self.index)}\n")
                f.write(f"Average Document Length: {self.avg_doc_length:.2f} tokens\n")
                f.write("\n" + "="*80 + "\n\n")
                
                # Inverted Index
                f.write("INVERTED INDEX:\n")
                f.write("-"*80 + "\n")
                
                # Sort terms alphabetically
                sorted_terms = sorted(self.index.items())
                
                for term, postings in sorted_terms:
                    f.write(f"\nTERM: '{term}'\n")
                    f.write(f"  Document Frequency: {len(postings)}\n")
                    f.write(f"  Postings: ")
                    
                    # Sort postings by doc_id
                    sorted_postings = sorted(postings.items())
                    posting_str = ", ".join([f"(Doc{doc_id}: {tf})" for doc_id, tf in sorted_postings])
                    f.write(posting_str + "\n")
                
                # Document Lengths
                f.write("\n" + "="*80 + "\n")
                f.write("DOCUMENT LENGTHS:\n")
                f.write("-"*80 + "\n")
                
                sorted_docs = sorted(self.doc_lengths.items())
                for doc_id, length in sorted_docs:
                    f.write(f"Doc {doc_id}: {length} tokens\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("END OF INDEX\n")
                f.write("="*80 + "\n")
            
            print(f"📄 Text index disimpan: {txt_file}")
        
        # 3. SIMPAN JSON (Cross-platform - structured)
        if json_file:
            # Convert defaultdict to regular dict for JSON serialization
            json_data = {
                'metadata': {
                    'num_docs': self.num_docs,
                    'num_unique_terms': len(self.index),
                    'avg_doc_length': self.avg_doc_length
                },
                'index': {
                    term: dict(postings) for term, postings in self.index.items()
                },
                'doc_lengths': self.doc_lengths
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"📋 JSON index disimpan: {json_file}")
    
    @classmethod
    def load_index(cls, input_file: str):
        """
        Memuat index dari file pickle
        """
        with open(input_file, 'rb') as f:
            index_data = pickle.load(f)
        
        obj = cls()
        obj.index = defaultdict(lambda: defaultdict(int), index_data['index'])
        obj.doc_lengths = index_data['doc_lengths']
        obj.num_docs = index_data['num_docs']
        obj.avg_doc_length = index_data['avg_doc_length']
        
        print(f"📚 Index dimuat dari: {input_file}")
        print(f"   Total unique terms: {len(obj.index)}")
        print(f"   Total documents: {obj.num_docs}")
        
        return obj
    
    @classmethod
    def load_index_from_txt(cls, txt_file: str):
        """
        Memuat index dari file TXT (parsing manual)
        """
        print(f"📄 Memuat index dari TXT: {txt_file}")
        
        obj = cls()
        current_term = None
        section = None
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('=') or line.startswith('-'):
                    continue
                
                if line.startswith('Total Documents:'):
                    obj.num_docs = int(line.split(':')[1].strip())
                
                elif line.startswith('Average Document Length:'):
                    obj.avg_doc_length = float(line.split(':')[1].strip().split()[0])
                
                elif line == 'INVERTED INDEX:':
                    section = 'index'
                
                elif line == 'DOCUMENT LENGTHS:':
                    section = 'doc_lengths'
                
                elif line.startswith('TERM:'):
                    current_term = line.split("'")[1]
                    obj.index[current_term] = defaultdict(int)
                
                elif line.startswith('Postings:') and current_term:
                    postings_str = line.split('Postings:')[1].strip()
                    
                    import re
                    matches = re.findall(r'\(Doc(\d+):\s*(\d+)\)', postings_str)
                    
                    for doc_id, tf in matches:
                        obj.index[current_term][int(doc_id)] = int(tf)
                
                elif section == 'doc_lengths' and line.startswith('Doc '):
                    parts = line.split(':')
                    doc_id = int(parts[0].split()[1])
                    length = int(parts[1].strip().split()[0])
                    obj.doc_lengths[doc_id] = length
        
        print(f"✅ Index dimuat dari TXT")
        print(f"   Total unique terms: {len(obj.index)}")
        print(f"   Total documents: {obj.num_docs}")
        
        return obj

    @classmethod
    def load_index_from_json(cls, json_file: str):
        """
        Memuat index dari file JSON
        """
        print(f"📋 Memuat index dari JSON: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        obj = cls()
        
        obj.num_docs = json_data['metadata']['num_docs']
        obj.avg_doc_length = json_data['metadata']['avg_doc_length']
        
        for term, postings in json_data['index'].items():
            obj.index[term] = defaultdict(int, postings)
        
        obj.doc_lengths = json_data['doc_lengths']
        
        for key in obj.doc_lengths:
            obj.doc_lengths[int(key)] = obj.doc_lengths.pop(key)
        
        print(f"✅ Index dimuat dari JSON")
        print(f"   Total unique terms: {len(obj.index)}")
        print(f"   Total documents: {obj.num_docs}")
        
        return obj
    
    def get_statistics(self) -> Dict:
        """
        Mendapatkan statistik dari index
        """
        term_doc_counts = [len(posting) for posting in self.index.values()]
        
        return {
            'num_unique_terms': len(self.index),
            'num_documents': self.num_docs,
            'avg_doc_length': self.avg_doc_length,
            'min_postings': min(term_doc_counts) if term_doc_counts else 0,
            'max_postings': max(term_doc_counts) if term_doc_counts else 0,
            'avg_postings': sum(term_doc_counts) / len(term_doc_counts) if term_doc_counts else 0
        }


def main():
    """
    Main function untuk membuat index dalam 3 format
    """
    # Path file
    json_file = "preprocessed_corpus.json"
    
    # Output files (3 format)
    pkl_file = "inverted_index.pkl"
    txt_file = "inverted_index.txt"
    json_output = "inverted_index.json"
    
    # Buat index
    indexer = InvertedIndex()
    indexer.build_index(json_file)
    
    # Simpan dalam 3 format
    print("\n💾 Menyimpan index dalam 3 format...")
    indexer.save_index(
        pkl_file=pkl_file,
        txt_file=txt_file,
        json_file=json_output
    )
    
    # Tampilkan statistik
    print("\n📊 Statistik Index:")
    stats = indexer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Test load index dari pickle
    print("\n🔄 Testing load index dari pickle...")
    loaded_index = InvertedIndex.load_index(pkl_file)
    
    # Contoh pencarian
    print("\n🔍 Contoh Pencarian:")
    from preprocessing import TextPreprocessor
    
    preprocessor = TextPreprocessor()
    query = "WNI online scam Kamboja"
    query_tokens = preprocessor.preprocess(query)
    
    print(f"Query: '{query}'")
    print(f"Query tokens: {query_tokens}")
    
    # BM25
    print("\n🎯 Top 5 Hasil BM25:")
    results_bm25 = loaded_index.search(query_tokens, top_k=5)
    for rank, (doc_id, score) in enumerate(results_bm25, 1):
        print(f"   {rank}. Document ID: {doc_id}, Score: {score:.4f}")
    
    # TF-IDF
    print("\n📊 Top 5 Hasil TF-IDF:")
    results_tfidf = loaded_index.search_tfidf(query_tokens, top_k=5)
    for rank, (doc_id, score) in enumerate(results_tfidf, 1):
        print(f"   {rank}. Document ID: {doc_id}, Score: {score:.4f}")
    
    print("\n✅ Selesai! File yang dihasilkan:")
    print(f"   1. {pkl_file} (Production - untuk search engine)")
    print(f"   2. {txt_file} (Debug - human readable)")
    print(f"   3. {json_output} (Backup - structured data)")


if __name__ == "__main__":
    main()