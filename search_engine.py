import json
import pickle
import re
from collections import defaultdict
from typing import Dict, List, Set
import math

from preprocessing import TextPreprocessor
from indexing import InvertedIndex

class SearchEngine:
    def __init__(self, index_file: str, corpus_file: str, file_type: str = 'pkl'):
        print("🔍 Memuat Search Engine...")
        
        if file_type == 'txt':
            self.index = InvertedIndex.load_index_from_txt(index_file)
        elif file_type == 'json':
            self.index = InvertedIndex.load_index_from_json(index_file)
        else:
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
        
        print(f"\n🔹 Overlap Dokumen:")
        print(f"   Dokumen yang sama: {len(overlap)} dari {top_k}")
        print(f"   Overlap percentage: {len(overlap)/top_k*100:.1f}%")
        print(f"   Unik TF-IDF: {len(tfidf_ids - bm25_ids)} dokumen")
        print(f"   Unik BM25: {len(bm25_ids - tfidf_ids)} dokumen")
        
        score_stats = self.calculate_score_statistics(query, top_k)
        print(f"\n🔹 Statistik Score:")
        print(f"   TF-IDF:")
        print(f"      Mean: {score_stats['tfidf']['mean']:.4f}")
        print(f"      Std Dev: {score_stats['tfidf']['std']:.4f}")
        print(f"      Range: {score_stats['tfidf']['min']:.4f} - {score_stats['tfidf']['max']:.4f}")
        print(f"   BM25:")
        print(f"      Mean: {score_stats['bm25']['mean']:.4f}")
        print(f"      Std Dev: {score_stats['bm25']['std']:.4f}")
        print(f"      Range: {score_stats['bm25']['min']:.4f} - {score_stats['bm25']['max']:.4f}")
        
        diversity = self.calculate_diversity_statistics(query, top_k)
        print(f"\n🔹 Keberagaman Sumber:")
        print(f"   TF-IDF: {diversity['tfidf']['unique_sources']} sumber berbeda")
        for source, count in diversity['tfidf']['source_distribution'].items():
            print(f"      - {source}: {count} artikel")
        print(f"   BM25: {diversity['bm25']['unique_sources']} sumber berbeda")
        for source, count in diversity['bm25']['source_distribution'].items():
            print(f"      - {source}: {count} artikel")
        
        if len(overlap) >= 2:
            rank_stats = self.calculate_ranking_statistics(query, top_k)
            print(f"\n🔹 Korelasi Ranking:")
            print(f"   Spearman Correlation: {rank_stats['rank_correlation']:.4f}")
            if rank_stats['rank_correlation'] > 0.7:
                print(f"   Interpretasi: Ranking sangat mirip")
            elif rank_stats['rank_correlation'] > 0.4:
                print(f"   Interpretasi: Ranking cukup mirip")
            else:
                print(f"   Interpretasi: Ranking berbeda signifikan")
    
    def calculate_diversity_statistics(self, query: str, top_k: int = 10) -> Dict:
        """
        Menghitung keberagaman sumber berita
        """
        results = self.search_both(query, top_k)
        
        tfidf_sources = [r['source'] for r in results['tfidf']]
        bm25_sources = [r['source'] for r in results['bm25']]
        
        from collections import Counter
        
        return {
            'tfidf': {
                'unique_sources': len(set(tfidf_sources)),
                'source_distribution': dict(Counter(tfidf_sources))
            },
            'bm25': {
                'unique_sources': len(set(bm25_sources)),
                'source_distribution': dict(Counter(bm25_sources))
            }
        }
    
    def calculate_score_statistics(self, query: str, top_k: int = 10) -> Dict:
        """
        Menghitung statistik distribusi score
        """
        results = self.search_both(query, top_k)
        
        tfidf_scores = [r['score'] for r in results['tfidf']]
        bm25_scores = [r['score'] for r in results['bm25']]
        
        import numpy as np
        
        return {
            'tfidf': {
                'mean': np.mean(tfidf_scores) if tfidf_scores else 0,
                'std': np.std(tfidf_scores) if tfidf_scores else 0,
                'min': min(tfidf_scores) if tfidf_scores else 0,
                'max': max(tfidf_scores) if tfidf_scores else 0,
                'range': max(tfidf_scores) - min(tfidf_scores) if tfidf_scores else 0
            },
            'bm25': {
                'mean': np.mean(bm25_scores) if bm25_scores else 0,
                'std': np.std(bm25_scores) if bm25_scores else 0,
                'min': min(bm25_scores) if bm25_scores else 0,
                'max': max(bm25_scores) if bm25_scores else 0,
                'range': max(bm25_scores) - min(bm25_scores) if bm25_scores else 0
            }
        }
    
    def calculate_ranking_statistics(self, query: str, top_k: int = 10) -> Dict:
        """
        Menghitung statistik ranking untuk perbandingan algoritma
        """
        results = self.search_both(query, top_k)
        
        tfidf_results = results['tfidf']
        bm25_results = results['bm25']
        
        tfidf_ids = [r['doc_id'] for r in tfidf_results]
        bm25_ids = [r['doc_id'] for r in bm25_results]
        
        tfidf_set = set(tfidf_ids)
        bm25_set = set(bm25_ids)
        
        overlap = tfidf_set.intersection(bm25_set)
        
        rank_correlation = 0
        if overlap:
            common_docs = list(overlap)
            tfidf_ranks = {doc_id: i for i, doc_id in enumerate(tfidf_ids)}
            bm25_ranks = {doc_id: i for i, doc_id in enumerate(bm25_ids)}
            
            from scipy.stats import spearmanr
            tfidf_rank_list = [tfidf_ranks[doc] for doc in common_docs]
            bm25_rank_list = [bm25_ranks[doc] for doc in common_docs]
            rank_correlation, _ = spearmanr(tfidf_rank_list, bm25_rank_list)
        
        return {
            'overlap_count': len(overlap),
            'overlap_percentage': len(overlap) / top_k * 100,
            'tfidf_unique': len(tfidf_set - bm25_set),
            'bm25_unique': len(bm25_set - tfidf_set),
            'rank_correlation': rank_correlation if overlap else 0,
            'total_retrieved': top_k
        }


def main():
    index_file = "inverted_index.pkl"
    corpus_file = "dataset\preprocessed_corpus.json"
    
    engine = SearchEngine(index_file, corpus_file)
    
    print("\n🔍 Mode Pencarian")
    print("Ketik 'quit' untuk keluar\n")
    
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