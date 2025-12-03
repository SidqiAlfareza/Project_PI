import json
import os
from typing import List, Dict, Tuple
from datetime import datetime
from Search_Engine import SearchEngine

class SearchEngineEvaluator:
    def __init__(self, engine: SearchEngine, ground_truth_file: str):
        self.engine = engine
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        self.k = 10
        
        self.eval_folder = os.path.join(os.path.dirname(__file__), "evaluasi")
        os.makedirs(self.eval_folder, exist_ok=True)
        print(f"📁 Folder evaluasi: {self.eval_folder}")
    
    def _load_ground_truth(self, file_path: str) -> Dict[str, List[int]]:
        ground_truth = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse format: "query" = id1, id2, id3
                if '=' in line:
                    try:
                        query_part, ids_part = line.split('=', 1)
                        query = query_part.strip().strip('"')
                        
                        doc_ids = []
                        for id_str in ids_part.split(','):
                            id_str = id_str.strip()
                            if id_str:
                                try:
                                    doc_ids.append(int(id_str))
                                except ValueError:
                                    print(f"⚠️  Warning: Invalid doc ID '{id_str}' at line {line_num}, skipping...")
                        
                        if doc_ids:
                            ground_truth[query] = doc_ids
                        else:
                            print(f"⚠️  Warning: No valid doc IDs for query '{query}' at line {line_num}")
                            
                    except Exception as e:
                        print(f"⚠️  Error parsing line {line_num}: {line}")
                        print(f"    Error: {e}")
                        continue
        
        print(f"✅ Loaded {len(ground_truth)} queries from ground truth")
        return ground_truth
    
    def precision_at_10(self, relevant_docs: List[int], retrieved_docs: List[int]) -> float:
        if self.k == 0:
            return 0.0
        retrieved_10 = retrieved_docs[:self.k]
        relevant_retrieved = len(set(relevant_docs) & set(retrieved_10))
        return relevant_retrieved / self.k
    
    def recall_at_10(self, relevant_docs: List[int], retrieved_docs: List[int]) -> float:
        if len(relevant_docs) == 0:
            return 0.0
        retrieved_10 = retrieved_docs[:self.k]
        relevant_retrieved = len(set(relevant_docs) & set(retrieved_10))
        return relevant_retrieved / len(relevant_docs)
    
    def f1_score_at_10(self, relevant_docs: List[int], retrieved_docs: List[int]) -> float:
        precision = self.precision_at_10(relevant_docs, retrieved_docs)
        recall = self.recall_at_10(relevant_docs, retrieved_docs)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, relevant_docs: List[int], retrieved_docs: List[int]) -> float:
        if len(relevant_docs) == 0:
            return 0.0
        
        score = 0.0
        num_relevant = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                score += precision_at_i
        
        return score / len(relevant_docs)
    
    def mean_average_precision(self, results: Dict[str, List[int]]) -> float:
        if len(results) == 0:
            return 0.0
        
        aps = []
        for query, retrieved_docs in results.items():
            relevant_docs = self.ground_truth.get(query, [])
            ap = self.average_precision(relevant_docs, retrieved_docs)
            aps.append(ap)
        
        return sum(aps) / len(aps)
    
    def evaluate_single_query(self, query: str, algorithm: str = 'bm25') -> Dict:
        relevant_docs = self.ground_truth.get(query, [])
        
        if not relevant_docs:
            print(f"⚠️  Query '{query}' tidak ada di ground truth")
            return None
        
        if algorithm.lower() == 'bm25':
            results = self.engine.search_bm25(query, top_k=self.k)
        else:
            results = self.engine.search_tfidf(query, top_k=self.k)
        
        retrieved_docs = [r['doc_id'] for r in results]
        
        metrics = {
            'query': query,
            'algorithm': algorithm.upper(),
            'num_relevant': len(relevant_docs),
            'num_retrieved': len(retrieved_docs),
            'precision@10': self.precision_at_10(relevant_docs, retrieved_docs),
            'recall@10': self.recall_at_10(relevant_docs, retrieved_docs),
            'f1@10': self.f1_score_at_10(relevant_docs, retrieved_docs),
            'average_precision': self.average_precision(relevant_docs, retrieved_docs)
        }
        
        return metrics
    
    def evaluate_all_queries(self, algorithm: str = 'bm25') -> Dict:
        print(f"\n{'='*80}")
        print(f"📊 EVALUASI SEARCH ENGINE @10 - Algorithm: {algorithm.upper()}")
        print(f"{'='*80}\n")
        
        all_metrics = []
        retrieved_docs_map = {}
        
        for query in self.ground_truth.keys():
            metrics = self.evaluate_single_query(query, algorithm)
            if metrics:
                all_metrics.append(metrics)
                
                if algorithm.lower() == 'bm25':
                    results = self.engine.search_bm25(query, top_k=self.k)
                else:
                    results = self.engine.search_tfidf(query, top_k=self.k)
                retrieved_docs_map[query] = [r['doc_id'] for r in results]
        
        avg_metrics = {
            'algorithm': algorithm.upper(),
            'num_queries': len(all_metrics),
            'avg_precision@10': sum(m['precision@10'] for m in all_metrics) / len(all_metrics),
            'avg_recall@10': sum(m['recall@10'] for m in all_metrics) / len(all_metrics),
            'avg_f1@10': sum(m['f1@10'] for m in all_metrics) / len(all_metrics),
            'MAP': self.mean_average_precision(retrieved_docs_map)
        }
        
        return {
            'summary': avg_metrics,
            'per_query': all_metrics
        }
    
    def save_results_to_txt(self, results: Dict, filename: str):
        filepath = os.path.join(self.eval_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            summary = results['summary']
            per_query = results['per_query']
            
            # Header
            f.write("="*80 + "\n")
            f.write(f"HASIL EVALUASI SEARCH ENGINE - {summary['algorithm']}\n")
            f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Summary
            f.write("RINGKASAN EVALUASI @10\n")
            f.write("-"*80 + "\n")
            f.write(f"Jumlah Query: {summary['num_queries']}\n\n")
            
            f.write(f"{'Metrik':<30} {'Nilai':>15}\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Average Precision@10':<30} {summary['avg_precision@10']:>15.4f}\n")
            f.write(f"{'Average Recall@10':<30} {summary['avg_recall@10']:>15.4f}\n")
            f.write(f"{'Average F1-Score@10':<30} {summary['avg_f1@10']:>15.4f}\n")
            f.write(f"{'Mean Average Precision (MAP)':<30} {summary['MAP']:>15.4f}\n")
            
            # Per Query Results
            f.write("\n" + "="*80 + "\n")
            f.write("HASIL PER QUERY @10\n")
            f.write("="*80 + "\n\n")
            
            for i, metrics in enumerate(per_query, 1):
                f.write(f"[{i}] Query: {metrics['query']}\n")
                f.write("-"*80 + "\n")
                f.write(f"  Jumlah Dokumen Relevan (Ground Truth): {metrics['num_relevant']}\n")
                f.write(f"  Jumlah Dokumen Retrieved (@10):        {metrics['num_retrieved']}\n\n")
                
                # Metrics
                f.write(f"  Performance Metrics:\n")
                f.write(f"    Precision@10:      {metrics['precision@10']:.4f}\n")
                f.write(f"    Recall@10:         {metrics['recall@10']:.4f}\n")
                f.write(f"    F1-Score@10:       {metrics['f1@10']:.4f}\n")
                f.write(f"    Average Precision: {metrics['average_precision']:.4f}\n")
                f.write("\n")
        
        print(f"✅ Hasil evaluasi TXT disimpan: {filepath}")
    
    def save_results_to_json(self, results: Dict, filename: str):
        filepath = os.path.join(self.eval_folder, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Hasil evaluasi JSON disimpan: {filepath}")
    
    def compare_algorithms(self):
        print(f"\n{'='*80}")
        print(f"🔄 EVALUASI OTOMATIS: TF-IDF & BM25 @10")
        print(f"{'='*80}\n")
        
        print("🔍 [1/2] Mengevaluasi TF-IDF...")
        tfidf_results = self.evaluate_all_queries('tfidf')
        
        print("🔍 [2/2] Mengevaluasi BM25...")
        bm25_results = self.evaluate_all_queries('bm25')
        
        print("\n💾 Menyimpan hasil evaluasi...")
        self.save_results_to_txt(tfidf_results, 'evaluation_tfidf_at10.txt')
        self.save_results_to_json(tfidf_results, 'evaluation_tfidf_at10.json')
        
        self.save_results_to_txt(bm25_results, 'evaluation_bm25_at10.txt')
        self.save_results_to_json(bm25_results, 'evaluation_bm25_at10.json')
        
        comparison = {
            'tfidf': tfidf_results,
            'bm25': bm25_results
        }
        
        self.save_comparison_to_txt(tfidf_results, bm25_results)
        self.save_results_to_json(comparison, 'evaluation_comparison_at10.json')
        
        print("\n" + "="*80)
        print("📊 PERBANDINGAN LANGSUNG @10")
        print("="*80)
        print(f"\n{'Metrik':<30} {'TF-IDF':>15} {'BM25':>15} {'Winner':>15}")
        print("-" * 78)
        
        tfidf_summary = tfidf_results['summary']
        bm25_summary = bm25_results['summary']
        
        metrics_to_compare = ['avg_precision@10', 'avg_recall@10', 'avg_f1@10', 'MAP']
        
        for key in metrics_to_compare:
            tfidf_val = tfidf_summary[key]
            bm25_val = bm25_summary[key]
            
            if abs(tfidf_val - bm25_val) < 0.0001:
                winner = "TIE"
            else:
                winner = "TF-IDF" if tfidf_val > bm25_val else "BM25"
            
            print(f"{key:<30} {tfidf_val:>15.4f} {bm25_val:>15.4f} {winner:>15}")
        
        print("\n" + "="*80)
        
        return comparison
    
    def save_comparison_to_txt(self, tfidf_results: Dict, bm25_results: Dict):
        filepath = os.path.join(self.eval_folder, 'evaluation_comparison_at10.txt')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PERBANDINGAN ALGORITMA: TF-IDF vs BM25 @10\n")
            f.write(f"Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            tfidf_summary = tfidf_results['summary']
            bm25_summary = bm25_results['summary']
            
            f.write(f"Jumlah Query: {tfidf_summary['num_queries']}\n\n")
            
            f.write(f"{'Metrik':<35} {'TF-IDF':>15} {'BM25':>15} {'Difference':>15} {'Winner':>15}\n")
            f.write("-"*98 + "\n")
            
            metrics_to_compare = ['avg_precision@10', 'avg_recall@10', 'avg_f1@10', 'MAP']
            
            for key in metrics_to_compare:
                tfidf_val = tfidf_summary[key]
                bm25_val = bm25_summary[key]
                diff = bm25_val - tfidf_val
                
                if abs(diff) < 0.0001:
                    winner = "TIE"
                else:
                    winner = "TF-IDF" if tfidf_val > bm25_val else "BM25"
                
                f.write(f"{key:<35} {tfidf_val:>15.4f} {bm25_val:>15.4f} {diff:>+15.4f} {winner:>15}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("KESIMPULAN\n")
            f.write("-"*80 + "\n")
            
            tfidf_wins = 0
            bm25_wins = 0
            
            for metric in metrics_to_compare:
                if tfidf_summary[metric] > bm25_summary[metric]:
                    tfidf_wins += 1
                elif bm25_summary[metric] > tfidf_summary[metric]:
                    bm25_wins += 1
            
            f.write(f"TF-IDF menang di {tfidf_wins} metrik\n")
            f.write(f"BM25 menang di {bm25_wins} metrik\n\n")
            
            if tfidf_wins > bm25_wins:
                f.write("ALGORITMA TERBAIK: TF-IDF\n")
            elif bm25_wins > tfidf_wins:
                f.write("ALGORITMA TERBAIK: BM25\n")
            else:
                f.write("HASIL: TIE (Performa setara)\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("PERBANDINGAN DETAIL PER QUERY\n")
            f.write("="*80 + "\n\n")
            
            for i, (tfidf_m, bm25_m) in enumerate(zip(tfidf_results['per_query'], 
                                                       bm25_results['per_query']), 1):
                f.write(f"[{i}] Query: {tfidf_m['query']}\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Metrik':<25} {'TF-IDF':>12} {'BM25':>12} {'Winner':>12}\n")
                f.write("-"*63 + "\n")
                
                metrics = ['precision@10', 'recall@10', 'f1@10', 'average_precision']
                for metric in metrics:
                    tfidf_val = tfidf_m[metric]
                    bm25_val = bm25_m[metric]
                    
                    if abs(tfidf_val - bm25_val) < 0.0001:
                        winner = "TIE"
                    else:
                        winner = "TF-IDF" if tfidf_val > bm25_val else "BM25"
                    
                    f.write(f"{metric:<25} {tfidf_val:>12.4f} {bm25_val:>12.4f} {winner:>12}\n")
                
                f.write("\n")
        
        print(f"✅ Comparison TXT disimpan: {filepath}")


def main():
    index_file = "inverted_index.txt"
    corpus_file = "dataset/preprocessed_corpus.json"
    ground_truth_file = "ground_truth.txt"
    
    print("="*80)
    print("🔍 SEARCH ENGINE EVALUATOR @10")
    print("="*80)
    
    print("\n📂 Loading Search Engine...")
    engine = SearchEngine(index_file, corpus_file, file_type='txt')
    
    print("📊 Initializing Evaluator...")
    evaluator = SearchEngineEvaluator(engine, ground_truth_file)
    
    print("\n🚀 Memulai evaluasi otomatis untuk TF-IDF dan BM25...")
    print("    (File akan diperbarui jika sudah ada)\n")
    
    comparison = evaluator.compare_algorithms()
    
    print("\n" + "="*80)
    print("✅ EVALUASI SELESAI!")
    print("="*80)
    print(f"\n📁 Semua hasil disimpan di: {evaluator.eval_folder}/")
    print("\n📄 File yang dihasilkan:")
    print("   • evaluation_tfidf_at10.txt")
    print("   • evaluation_tfidf_at10.json")
    print("   • evaluation_bm25_at10.txt")
    print("   • evaluation_bm25_at10.json")
    print("   • evaluation_comparison_at10.txt")
    print("   • evaluation_comparison_at10.json")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()