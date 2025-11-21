import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from dateutil.parser import parse
import re
from collections import Counter
import time

def load_urls(file_path):
    try:
        df = pd.read_excel(file_path)
        if 'lien' in df.columns:
            return df['lien'].tolist()
        else:
            return df.iloc[:, 1].tolist()
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return []

def generate_summary(text, word_limit):
    words = text.split()
    if len(words) <= word_limit:
        return text
    
    word_counts = Counter(words)
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    sentence_scores = {}
    for sentence in sentences:
        for word in sentence.split():
            if word in word_counts:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = 0
                sentence_scores[sentence] += word_counts[word]
    
    sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    
    summary = []
    current_word_count = 0
    for sentence in sorted_sentences:
        if current_word_count + len(sentence.split()) <= word_limit:
            summary.append(sentence)
            current_word_count += len(sentence.split())
        else:
            break
            
    summary_indices = [sentences.index(s) for s in summary]
    summary_indices.sort()
    final_summary = " ".join([sentences[i] for i in summary_indices])
    return final_summary

def extract_info(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.title.string if soup.title else ""
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
            
        paragraphs = soup.find_all('p')
        content = " ".join([p.get_text(strip=True) for p in paragraphs])
        
        try:
            lang = detect(content)
        except:
            lang = "unknown"
            
        date_str = ""
        meta_date = soup.find('meta', property='article:published_time') or \
                    soup.find('meta', property='og:published_time') or \
                    soup.find('meta', attrs={'name': 'date'})
        if meta_date:
            date_str = meta_date.get('content', '')
        
        if not date_str:
            date_match = re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', content)
            if date_match:
                date_str = date_match.group(0)
                
        try:
            if date_str:
                dt = parse(date_str, fuzzy=True)
                date_pub = dt.strftime("%d-%m-%Y")
            else:
                date_pub = "Unknown"
        except:
            date_pub = "Unknown"

        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        domain = domain_match.group(1) if domain_match else "Unknown"
        
        source = "Media"
        social_domains = ["facebook.com", "twitter.com", "x.com", "youtube.com", "instagram.com", "linkedin.com"]
        official_domains = ["who.int", "woah.org", "fao.org", "gov", "gouv.fr", "europa.eu", "agriculture.dz"]
        
        if any(sd in domain for sd in social_domains):
            source = "Social Media"
        elif any(od in domain for od in official_domains):
            source = "Official"
            
        diseases_list = [
            "fièvre aphteuse", "grippe aviaire", "influenza", "peste porcine", "peste des petits ruminants",
            "rage", "brucellose", "tuberculose", "charbon", "anthrax", "clavelée", "newcastle",
            "vache folle", "blue tongue", "fièvre catarrhale", "dermatose nodulaire", "fco", "bovine",
            "strangles", "gourme", "mhd", "epizootic"
        ]
        
        locations_list = [
            "algérie", "tunisie", "maroc", "france", "egypte", "tchad", "afrique", "europe", "asie",
            "alger", "oran", "constantine", "tunis", "rabat", "paris", "ontario", "michigan", "benzie"
        ]
        
        disease = "Unknown"
        content_lower = content.lower()
        title_lower = title.lower()
        
        for d in diseases_list:
            if d in title_lower or d in content_lower:
                disease = d.title()
                break
                
        location = "Unknown"
        for l in locations_list:
            if l in title_lower or l in content_lower:
                location = l.title()
                break
        
        return {
            "URL": url,
            "Titre": title,
            "Contenu": content,
            "Langue": lang,
            "Nb Caractères": len(content),
            "Nb Mots": len(content.split()),
            "Date Publication": date_pub,
            "Lieu": location,
            "Maladie": disease,
            "Source": source,
            "Domaine": domain,
            "Résumé 50": generate_summary(content, 50),
            "Résumé 100": generate_summary(content, 100),
            "Résumé 150": generate_summary(content, 150),
            "Entités": ""
        }

    except Exception as e:
        print(f"Failed to process {url}: {e}")
        return None

def main():
    urls = load_urls('groupe14.xlsx')
    print(f"Found {len(urls)} URLs.")
    
    data = []
    for i, url in enumerate(urls):
        print(f"Processing {i+1}/{len(urls)}: {url}")
        info = extract_info(url)
        if info:
            data.append(info)
        time.sleep(1)
        
    df_result = pd.DataFrame(data)
    df_result.to_csv('dataset.csv', index=False, encoding='utf-8-sig')
    print("Dataset saved to dataset.csv")

if __name__ == "__main__":
    main()
