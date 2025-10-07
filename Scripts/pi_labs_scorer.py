import os
import pandas as pd
from withpi import PiClient

def read_text_files(folder_path):
    """Read all txt files from the folder"""
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                texts[filename] = f.read()
    return texts

def score_with_pi(pi_client, prompt, text, observations):
    """Score a single text using PI scorer"""
    
    scoring_params = {
        "llm_input": prompt,
        "llm_output": text,
        "scoring_spec": [
            {
                "label": "Realism",
                "question": f"How realistic does the generated output look based on actual changelog/version file patterns? Consider these observations: {observations}"
            },
            {
                "label": "Prompt Adherence",
                "question": "How much does the generated text answer the prompt?"
            },
            {
                "label": "Clarity",
                "question": "How well can the content be understood? How much of the things are clarified in the generated text?"
            },
            {
                "label": "Factual Consistency",
                "question": "Does the generated text contain any contradictions or inconsistent information?"
            },
            {
                "label": "Completeness",
                "question": "Does the output cover all key aspects typically expected in a changelog/version file?"
            },
            {
                "label": "Technical Accuracy",
                "question": "Does the technical terminology and syntax appear correct and appropriate?"
            }
        ],
    }
    
    response = pi_client.scoring_system.score(**scoring_params)
    return response

def main():
    # Configuration
    PI_API_KEY = input("Enter your PI API key: ").strip()
    folder_path = input("Enter the folder path containing txt files: ").strip()
    prompt_path = input("Enter the path to the prompt file (txt): ").strip()
    output_csv = input("Enter output CSV filename (default: pi_scores.csv): ").strip()
    
    if not output_csv:
        output_csv = "pi_scores.csv"
    
    # Your observations for the realism question
    observations = """
    Technical jargon/syntax used (output syntax), numbers, unique ticket numbers.
    Occasionally sounds like code output, especially with errors.
    Pattern: <Error fixed> <What error was filed>.
    Sometimes: <Error fixed>, <What error was fixed>, <Why was it fixed?>.
    Around 7-15 tickets max per version release.
    Grammatical mistakes/typos/improper punctuation.
    Clear distinction in updated categories (Documentation, security, etc).
    No consistency in action verbs (Update/Updated).
    Each ticket is 3 sentences at max.
    Sprinkled with name drops: <Full Name> + <email id>.
    Version files are formal with occasional hype, exceptionally longer than changelog.
    Sentence pattern: <What the new thing does> <How it does it>.
    """
    
    # Initialize PI client
    print("\nInitializing PI client...")
    pi = PiClient(api_key=PI_API_KEY)
    
    # Read prompt
    print("Reading prompt...")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    # Read texts
    print("Reading text files...")
    texts = read_text_files(folder_path)
    
    if not texts:
        print("No txt files found in the specified folder!")
        return
    
    print(f"Found {len(texts)} text files")
    
    # Score each text
    results = []
    print("\nScoring texts with PI judge...")
    
    for i, (filename, text) in enumerate(texts.items(), 1):
        print(f"Scoring {i}/{len(texts)}: {filename}")
        
        try:
            response = score_with_pi(pi, prompt, text, observations)
            question_scores = response.question_scores
            
            results.append({
                'model_name': filename,
                'total_score': round(response.total_score, 4),
                'realism': round(question_scores.get('Realism', 0), 4),
                'prompt_adherence': round(question_scores.get('Prompt Adherence', 0), 4),
                'clarity': round(question_scores.get('Clarity', 0), 4),
                'factual_consistency': round(question_scores.get('Factual Consistency', 0), 4),
                'completeness': round(question_scores.get('Completeness', 0), 4),
                'technical_accuracy': round(question_scores.get('Technical Accuracy', 0), 4)
            })
            
        except Exception as e:
            print(f"Error scoring {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create DataFrame and save
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('total_score', ascending=False)
        df.to_csv(output_csv, index=False)
        
        print(f"\n✓ Results saved to {output_csv}")
        print("\nTop 3 models:")
        print(df.head(3).to_string(index=False))
    else:
        print("\nNo results to save!")

if __name__ == "__main__":
    main()