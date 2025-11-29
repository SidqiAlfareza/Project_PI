import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import json
from urllib.parse import urljoin
import time

class ImageExtractor:
    def __init__(self, csv_file, preprocessed_json, preprocessed_csv):
        self.csv_file = csv_file
        self.preprocessed_json = preprocessed_json
        self.preprocessed_csv = preprocessed_csv
        
        self.df = pd.read_csv(csv_file)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.image_folder = 'artikel_images'
        os.makedirs(self.image_folder, exist_ok=True)
    
    def extract_image_from_url(self, url, article_id):
        """
        Ekstrak gambar utama dari URL artikel
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            image_url = None
            
            # 1. Cari Open Graph image (paling reliable)
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                image_url = og_image['content']
            
            # 2. Cari Twitter Card image
            if not image_url:
                twitter_image = soup.find('meta', attrs={'name': 'twitter:image'})
                if twitter_image and twitter_image.get('content'):
                    image_url = twitter_image['content']
            
            # 3. Cari di article/content div
            if not image_url:
                content_divs = soup.find_all(['article', 'div'], class_=['content', 'article', 'detail'])
                for div in content_divs:
                    img = div.find('img')
                    if img and img.get('src'):
                        image_url = img['src']
                        break
            
            # 4. Fallback: ambil img pertama yang besar
            if not image_url:
                for img in soup.find_all('img'):
                    src = img.get('src') or img.get('data-src')
                    if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png']):
                        image_url = src
                        break
            
            if not image_url:
                return None
            
            # Pastikan URL lengkap
            if not image_url.startswith('http'):
                image_url = urljoin(url, image_url)
            
            # Download gambar
            img_response = requests.get(image_url, headers=self.headers, timeout=10)
            img_response.raise_for_status()
            
            # Simpan gambar
            ext = image_url.split('.')[-1].split('?')[0]
            if ext not in ['jpg', 'jpeg', 'png', 'webp']:
                ext = 'jpg'
            
            filename = f"article_{article_id}.{ext}"
            filepath = os.path.join(self.image_folder, filename)
            
            with open(filepath, 'wb') as f:
                f.write(img_response.content)
            
            return filename
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")
            return None
    
    def extract_all_images(self):
        """
        Ekstrak gambar untuk semua artikel
        """
        print(f"🖼️  Memulai ekstraksi gambar dari {len(self.df)} artikel...")
        print("="*70)
        
        image_filenames = []
        success_count = 0
        
        for idx, row in self.df.iterrows():
            article_id = idx
            url = row['url']
            title = row['title'][:50]
            
            print(f"[{idx+1}/{len(self.df)}] {title}...")
            
            filename = self.extract_image_from_url(url, article_id)
            
            if filename:
                image_filenames.append(filename)
                success_count += 1
                print(f"   ✅ Gambar disimpan: {filename}")
            else:
                image_filenames.append(None)
            
            time.sleep(1)
        
        # Update DataFrame
        self.df['image'] = image_filenames
        
        print(f"\n✅ Berhasil: {success_count}/{len(self.df)} artikel")
        
        return image_filenames
    
    def update_preprocessed_files(self, image_filenames):
        """
        Update preprocessed JSON dan CSV dengan kolom image
        """
        print("\n📝 Update file preprocessed...")
        
        with open(self.preprocessed_json, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        for idx, item in enumerate(json_data):
            if idx < len(image_filenames):
                item['image'] = image_filenames[idx]
            else:
                item['image'] = None
        
        with open(self.preprocessed_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ {self.preprocessed_json} diupdate")
        
        if os.path.exists(self.preprocessed_csv):
            df_preprocessed = pd.read_csv(self.preprocessed_csv)
            
            if len(df_preprocessed) == len(image_filenames):
                df_preprocessed['image'] = image_filenames
                df_preprocessed.to_csv(self.preprocessed_csv, index=False, encoding='utf-8-sig')
                print(f"   ✅ {self.preprocessed_csv} diupdate")
            else:
                print(f"   ⚠️  Jumlah baris tidak cocok, skip update CSV")
        
        output_csv = 'corpus_wni_kamboja_with_images.csv'
        self.df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"   ✅ {output_csv} dibuat")
    
    def run(self):
        """
        Jalankan semua proses
        """
        image_filenames = self.extract_all_images()
        
        self.update_preprocessed_files(image_filenames)
        
        print("\n" + "="*70)
        print("📊 RINGKASAN")
        print("="*70)
        print(f"📁 Gambar tersimpan di: {self.image_folder}/")
        print(f"📄 File diupdate:")
        print(f"   - {self.preprocessed_json}")
        print(f"   - {self.preprocessed_csv}")
        print(f"   - corpus_wni_kamboja_with_images.csv")
        print("="*70)


if __name__ == "__main__":
    csv_file = 'corpus_wni_kamboja_update.csv'
    preprocessed_json = 'preprocessed_corpus.json'
    preprocessed_csv = 'corpus_preprocessed.csv'
    
    extractor = ImageExtractor(csv_file, preprocessed_json, preprocessed_csv)
    extractor.run()
    
    print("\n✅ EKSTRAKSI DAN UPDATE SELESAI!")