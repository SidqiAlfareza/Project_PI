import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
from datetime import datetime
from urllib.parse import urljoin, quote

class MultiSourceKambojaCrawler:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.articles = []
        self.urls_visited = set()
        
        # Konfigurasi untuk berbagai sumber berita
        self.sources = {
            'detik': {
                'base': 'https://www.detik.com',
                'search': 'https://www.detik.com/search/searchall?query={}&sortby=time',
                'selectors': {
                    'title': {'tag': 'h1', 'class': 'detail__title'},
                    'content': {'tag': 'div', 'class': 'detail__body-text'}
                }
            },
            'kompas': {
                'base': 'https://www.kompas.com',
                'search': 'https://www.kompas.com/search/{term}',
                'selectors': {
                    'title': {'tag': 'h1', 'class': 'read__title'},
                    'content': {'tag': 'div', 'class': 'read__content'}
                }
            },
            'tempo': {
                'base': 'https://www.tempo.co',
                'search': 'https://www.tempo.co/search?q={}',
                'selectors': {
                    'title': {'tag': 'h1', 'class': 'title'},
                    'content': {'tag': 'div', 'class': 'detail-content'}
                }
            },
            'cnnindonesia': {
                'base': 'https://www.cnnindonesia.com',
                'search': 'https://www.cnnindonesia.com/search/?query={}',
                'selectors': {
                    'title': {'tag': 'h1', 'class': 'title'},
                    'content': {'tag': 'div', 'class': 'detail-text'}
                }
            }, 
            'liputan6': {
                'base': 'https://www.liputan6.com',
                'search': 'https://www.liputan6.com/search/{}',
                'selectors': {
                    'title': {'tag': 'h1', 'class': 'title-detail'},
                    'content': {'tag': 'div', 'class': 'article-content-body'}
                }
            }
        }
    
    def is_relevant(self, title, content):
        """Validasi relevansi artikel dengan WNI di Kamboja"""
        text = (title + " " + content).lower()
        
        # Harus ada keyword Kamboja
        kamboja_kw = ['kamboja', 'cambodia', 'phnom penh', 'sihanoukville']
        if not any(k in text for k in kamboja_kw):
            return False
        
        # Harus ada keyword Indonesia/WNI
        indo_kw = ['wni', 'indonesia', 'tki', 'pekerja migran', 'korban', 
                   'warga negara indonesia', 'warga indonesia']
        if not any(k in text for k in indo_kw):
            return False
        
        # Bonus: ada keyword topik relevan
        topik_kw = [
            'penipuan', 'scam', 'cyber', 'online', 'judi', 'gambling',
            'human trafficking', 'perdagangan', 'eksploitasi', 'terjerat',
            'repatriasi', 'pemulangan', 'sindikat', 'mafia', 'deportasi',
            'kriminal', 'kejahatan', 'paspor', 'visa', 'imigrasi'
        ]
        
        return True
    
    def extract_content_generic(self, soup, url):
        """Ekstraksi konten secara generic (fallback)"""
        title = None
        content_parts = []
        
        # Coba berbagai selector untuk judul
        for selector in ['h1', 'h2']:
            title_tag = soup.find(selector)
            if title_tag:
                title = title_tag.get_text(strip=True)
                if len(title) > 20:
                    break
        
        # Ambil semua paragraf
        for p in soup.find_all('p'):
            text = p.get_text(strip=True)
            if len(text) > 50:
                content_parts.append(text)
        
        content = ' '.join(content_parts)
        return title, content
    
    def scrape_article(self, url, source_name='unknown'):
        """Scrape artikel dari URL"""
        if url in self.urls_visited or len(url) > 500:
            return False
        
        try:
            # Deteksi sumber dari URL dengan lebih akurat
            if source_name == 'unknown':
                source_name = self.detect_source(url)
            
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Ekstraksi konten
            title, content = self.extract_content_generic(soup, url)
            
            if not title or len(content) < 500:
                return False
            
            # Validasi relevansi
            if not self.is_relevant(title, content):
                return False
            
            # Simpan artikel
            article = {
                'id': len(self.articles) + 1,
                'url': url,
                'source': source_name,
                'title': title,
                'content': content,
                'word_count': len(content.split()),
                'crawled_at': datetime.now().isoformat()
            }
            
            self.articles.append(article)
            self.urls_visited.add(url)
            
            print(f"✅ [{len(self.articles)}] [{source_name.upper()}] {title[:60]}...")
            return True
            
        except Exception as e:
            return False
    
    def detect_source(self, url):
        """Deteksi sumber berita dari URL dengan lebih akurat"""
        url_lower = url.lower()
        
        # Mapping domain ke nama sumber
        source_mapping = {
            'detik.com': 'detik',
            'kompas.com': 'kompas',
            'tempo.co': 'tempo',
            'cnnindonesia.com': 'cnnindonesia',
            'liputan6.com': 'liputan6',
            'tribunnews.com': 'tribun',
            'tribunmedan.com': 'tribun',
            'tribunjakarta.com': 'tribun',
            'okezone.com': 'okezone',
            'sindonews.com': 'sindonews',
            'antaranews.com': 'antara',
            'merdeka.com': 'merdeka',
            'viva.co.id': 'viva',
            'suara.com': 'suara',
            'jpnn.com': 'jpnn',
            'beritasatu.com': 'beritasatu',
            'republika.co.id': 'republika',
            'medcom.id': 'medcom',
            'bisnis.com': 'bisnis',
            'idntimes.com': 'idntimes',
            'grid.id': 'grid',
            'kumparan.com': 'kumparan',
            'tirto.id': 'tirto',
            'vice.com': 'vice',
            'bbc.com/indonesia': 'bbc',
            'voaindonesia.com': 'voa'
        }
        
        # Cek setiap domain
        for domain, source in source_mapping.items():
            if domain in url_lower:
                return source
        
        # Jika masih tidak ketemu, ekstrak domain dari URL
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            
            # Ambil nama utama domain (tanpa .com/.co.id)
            if domain:
                source_name = domain.split('.')[0]
                return source_name
        except:
            pass
        
        return 'other'
    
    def search_google_news(self, query, num_results=100):
        """Gunakan Google untuk mencari berita dari berbagai sumber"""
        print(f"\n🔍 Google Search: '{query}'")
        
        search_query = f"{query} WNI Kamboja"
        encoded = quote(search_query)
        
        # Cari di Google News
        urls_to_check = []
        
        for start in range(0, num_results, 10):
            try:
                google_url = f"https://www.google.com/search?q={encoded}&start={start}&num=10&tbm=nws"
                response = requests.get(google_url, headers=self.headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Ekstrak URL
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    
                    # Filter domain berita Indonesia (diperluas)
                    indo_news = [
                        'detik.com', 'kompas.com', 'tempo.co', 
                        'cnnindonesia.com', 'liputan6.com', 
                        'tribunnews.com', 'okezone.com', 'sindonews.com',
                        'antaranews.com', 'merdeka.com', 'viva.co.id',
                        'suara.com', 'jpnn.com', 'beritasatu.com',
                        'republika.co.id', 'medcom.id', 'bisnis.com',
                        'idntimes.com', 'grid.id', 'kumparan.com',
                        'tirto.id', 'vice.com/id', 'bbc.com/indonesia',
                        'voaindonesia.com', 'tribun', 'jawapos.com'
                    ]
                    
                    if any(domain in href for domain in indo_news):
                        # Bersihkan URL
                        if 'url?q=' in href:
                            clean_url = href.split('url?q=')[1].split('&')[0]
                        else:
                            clean_url = href
                        
                        if clean_url.startswith('http') and clean_url not in urls_to_check:
                            urls_to_check.append(clean_url)
                
                time.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                continue
        
        print(f"   📊 Ditemukan {len(urls_to_check)} URL potensial")
        
        # Scrape setiap URL
        success_count = 0
        for i, url in enumerate(urls_to_check, 1):
            if len(self.articles) >= 500:
                break
            
            # Scrape dengan auto-detect source
            if self.scrape_article(url):
                success_count += 1
            
            if i % 10 == 0:
                print(f"   📈 Progress: {i}/{len(urls_to_check)} URL | {success_count} relevan")
            
            time.sleep(1.5)
        
        print(f"   ✅ Berhasil: {success_count} artikel relevan dari {len(urls_to_check)} URL")
        return success_count
    
    def crawl_all(self):
        """Crawl dari berbagai keyword dan sumber"""
        queries = [
            # Query utama
            "WNI Kamboja penipuan",
            "penipuan online Kamboja",
            "WNI korban Kamboja",
            "cyber crime Kamboja",
            "human trafficking Kamboja Indonesia",
            "TKI Kamboja terjerat",
            "repatriasi WNI Kamboja",
            "judi online Kamboja Indonesia",
            "scam Kamboja",
            "perdagangan manusia Kamboja",
            "WNI dipulangkan Kamboja",
            
            # Query tambahan untuk mencapai 500
            "sindikat penipuan Kamboja",
            "WNI tertipu Kamboja",
            "Indonesia Kamboja kejahatan",
            "eksploitasi TKI Kamboja",
            "mafia Kamboja Indonesia",
            "pekerja migran Kamboja",
            "WNI terjebak Kamboja",
            "kasus WNI Kamboja",
            "deportasi WNI Kamboja",
            "imigrasi Kamboja Indonesia",
            "paspor palsu Kamboja",
            "visa overstay Kamboja",
            "gambling Kamboja Indonesia",
            "kasino online Kamboja",
            "call center penipuan Kamboja",
            "cyber slavery Kamboja",
            "pinjol ilegal Kamboja",
            "romance scam Kamboja",
            "job scam Kamboja Indonesia",
            "lowongan kerja palsu Kamboja",
            "WNI hilang Kamboja",
            "pencarian WNI Kamboja",
            "keluarga korban Kamboja",
            "Polri Kamboja",
            "Kemlu Kamboja WNI",
            "KBRI Phnom Penh",
            "embassy Indonesia Cambodia",
            "pemerintah Indonesia Kamboja",
            "kerjasama Indonesia Kamboja kejahatan"
        ]
        
        for i, query in enumerate(queries, 1):
            if len(self.articles) >= 500:
                print(f"\n🎉 Target 500 artikel tercapai!")
                break
            
            print(f"\n[{i}/{len(queries)}] Memproses query...")
            self.search_google_news(query, num_results=50)
            time.sleep(3)
    
    def save_results(self):
        """Simpan hasil crawling"""
        if not self.articles:
            print("\n❌ Tidak ada artikel yang berhasil di-crawl")
            return
        
        # JSON
        with open('corpus_wni_kamboja_update.json', 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=2)
        
        # CSV
        df = pd.DataFrame(self.articles)
        df.to_csv('corpus_wni_kamboja_update.csv', index=False, encoding='utf-8-sig')
        
        # Statistik
        print("\n" + "="*70)
        print("📊 STATISTIK HASIL CRAWLING")
        print("="*70)
        print(f"✅ Total Artikel: {len(self.articles)}")
        print(f"📝 Total Kata: {df['word_count'].sum():,}")
        print(f"📊 Rata-rata Kata/Artikel: {df['word_count'].mean():.0f}")
        print(f"\n📰 Sumber Artikel:")
        print(df['source'].value_counts().to_string())
        print("="*70)
        print("\n💾 File tersimpan:")
        print("   - corpus_wni_kamboja_multisource.json")
        print("   - corpus_wni_kamboja_multisource.csv")

# JALANKAN
if __name__ == "__main__":
    print("🚀 MULTI-SOURCE CRAWLER - WNI DI KAMBOJA")
    print("="*70)
    print("📌 Mengambil dari: Detik, Kompas, Tempo, CNN, Liputan6, dll")
    print("📌 Dengan filter relevansi otomatis")
    print("="*70)
    
    crawler = MultiSourceKambojaCrawler()
    crawler.crawl_all()
    crawler.save_results()
    
    print("\n✅ CRAWLING SELESAI!")