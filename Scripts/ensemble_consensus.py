import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def read_text_files(folder_path):
    """Read all txt files from the folder"""
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                texts[filename] = f.read()
    return texts

def calculate_consensus_similarity(texts):
    """Calculate average cosine similarity of each text against all others"""
    text_list = list(texts.values())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_list)
    
    similarity_scores = {}
    for i, filename in enumerate(texts.keys()):
        similarities = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix)[0]
        avg_similarity = (np.sum(similarities) - 1) / (len(similarities) - 1) if len(similarities) > 1 else 0
        similarity_scores[filename] = avg_similarity
    
    return similarity_scores

def calculate_prompt_similarity(prompt, texts):
    """Calculate cosine similarity between prompt and each text"""
    all_texts = [prompt] + list(texts.values())
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Prompt is at index 0
    prompt_vector = tfidf_matrix[0:1]
    
    similarity_scores = {}
    for i, filename in enumerate(texts.keys(), start=1):
        text_vector = tfidf_matrix[i:i+1]
        similarity = cosine_similarity(prompt_vector, text_vector)[0][0]
        similarity_scores[filename] = similarity
    
    return similarity_scores

def calculate_repetition_rate(text):
    """Calculate repetition rate in the text"""
    words = text.lower().split()
    if len(words) < 2:
        return 0.0
    
    bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)] # Count repeated bigrams
    if len(bigrams) == 0:
        return 0.0
    
    unique_bigrams = len(set(bigrams))
    repetition_rate = 1 - (unique_bigrams / len(bigrams))
    
    return repetition_rate

def calculate_metrics(prompt, texts):
    """Calculate all metrics for each text"""
    consensus_scores = calculate_consensus_similarity(texts)
    prompt_scores = calculate_prompt_similarity(prompt, texts)
    
    results = []
    for filename, text in texts.items():
        repetition_rate = calculate_repetition_rate(text)
        w_rep_inv = 1 - repetition_rate # inverse repetition rate
        
        consensus_score = (
            0.4 * consensus_scores[filename] + 
            0.3 * prompt_scores[filename] + 
            0.3 * w_rep_inv
        )
        
        results.append({
            'model_name': filename,
            'consensus_similarity': round(consensus_scores[filename], 4),
            'prompt_similarity': round(prompt_scores[filename], 4),
            'w_rep_inv': round(w_rep_inv, 4),
            'final_consensus_score': round(consensus_score, 4)
        })
    
    return results

def main():
    # Configuration
    folder_path = input("Enter the folder path containing txt files: ").strip()
    prompt_path = input("Enter the path to the prompt file (txt): ").strip()
    output_csv = input("Enter output CSV filename (default: ensemble_results.csv): ").strip()
    
    if not output_csv:
        output_csv = "ensemble_results.csv"
    
    # Read prompt
    print("\nReading prompt...")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # Read texts
    print("Reading text files...")
    texts = read_text_files(folder_path)
    
    if not texts:
        print("No txt files found in the specified folder!")
        return
    
    print(f"Found {len(texts)} text files")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    results = calculate_metrics(prompt, texts)
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values('final_consensus_score', ascending=False)
    df.to_csv(output_csv, index=False)
    
if __name__ == "__main__":
    main()