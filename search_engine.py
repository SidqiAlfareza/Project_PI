import json
import math
from typing import List, Dict, Tuple
from preprocessing import TextPreprocessor
from indexing import InvertedIndex

class SearchEngine:
    def __init__(self, index_file: str, corpus_file: str):
        print("🔍 Memuat Search Engine...")
        
        self.index = InvertedIndex.load_index(index_file)
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        
        self.preprocessor = TextPreprocessor()
        
        print("✅ Search Engine siap digunakan!\n")
    
    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict]:
        query_tokens = self.preprocessor.preprocess(query)
        
        if not query_tokens:
            return []
        
        results = self.index.search(query_tokens, top_k)
        
        search_results = []
        for doc_id, score in results:
            doc = self.corpus[doc_id]
            search_results.append({
                'doc_id': doc_id,
                'score': score,
                'title': doc['title'],
                'url': doc['url'],
                'source': doc['source'],
                'snippet': self._create_snippet(doc['original_content'], query_tokens),
                'algorithm': 'BM25'
            })
        
        return search_results
    
    def search_tfidf(self, query: str, top_k: int = 10) -> List[Dict]:
        query_tokens = self.preprocessor.preprocess(query)
        
        if not query_tokens:
            return []
        
        results = self.index.search_tfidf(query_tokens, top_k)
        
        search_results = []
        for doc_id, score in results:
            doc = self.corpus[doc_id]
            search_results.append({
                'doc_id': doc_id,
                'score': score,
                'title': doc['title'],
                'url': doc['url'],
                'source': doc['source'],
                'snippet': self._create_snippet(doc['original_content'], query_tokens),
                'algorithm': 'TF-IDF'
            })
        
        return search_results
    
    def search_both(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        return {
            'tfidf': self.search_tfidf(query, top_k),
            'bm25': self.search_bm25(query, top_k)
        }
    
    def _create_snippet(self, content: str, query_tokens: List[str], max_length: int = 200) -> str:
        snippet = content[:max_length]
        
        if len(content) > max_length:
            snippet += "..."
        
        return snippet
    
    def print_results(self, results: List[Dict], show_snippet: bool = True):
        if not results:
            print("❌ Tidak ada hasil ditemukan")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['title']}")
            print(f"    Score: {result['score']:.4f}")
            print(f"    Sumber: {result['source']}")
            print(f"    URL: {result['url']}")
            
            if show_snippet:
                print(f"    Snippet: {result['snippet'][:150]}...")
            
            print("-"*80)
    
    def compare_algorithms(self, query: str, top_k: int = 5):
        print(f"\n{'='*80}")
        print(f"🔍 Query: '{query}'")
        print(f"{'='*80}")
        
        query_tokens = self.preprocessor.preprocess(query)
        print(f"📝 Query tokens: {query_tokens}\n")
        
        results = self.search_both(query, top_k)
        
        print("\n" + "="*80)
        print("📊 HASIL TF-IDF")
        print("="*80)
        self.print_results(results['tfidf'], show_snippet=True)
        
        print("\n" + "="*80)
        print("🎯 HASIL BM25")
        print("="*80)
        self.print_results(results['bm25'], show_snippet=True)
        
        tfidf_ids = {r['doc_id'] for r in results['tfidf']}
        bm25_ids = {r['doc_id'] for r in results['bm25']}
        overlap = tfidf_ids.intersection(bm25_ids)
        
        print(f"\n{'='*80}")
        print(f"📈 ANALISIS PERBANDINGAN")
        print(f"{'='*80}")
        print(f"Dokumen yang sama di Top-{top_k}: {len(overlap)} dari {top_k}")
        print(f"Overlap percentage: {len(overlap)/top_k*100:.1f}%")


def main():
    index_file = "inverted_index.txt"
    corpus_file = "preprocessed_corpus.json"
    
    engine = SearchEngine(index_file, corpus_file)
    
    print("\n🔍 Mode Pencarian")
    print("Ketik 'quit' untuk keluar\n")
    print("Setiap pencarian akan menampilkan hasil dari TF-IDF dan BM25")
    print("="*80 + "\n")
    
    while True:
        query = input("Masukkan query pencarian: ").strip()
        
        if query.lower() == 'quit':
            print("👋 Terima kasih!")
            break
        
        if not query:
            continue
        
        engine.compare_algorithms(query, top_k=5)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
