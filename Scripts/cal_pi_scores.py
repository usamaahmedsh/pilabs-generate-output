import os
import pandas as pd
from withpi import PiClient
from withpi_utils import stream

def read_text_files(folder_path):
    """Read all txt files from the folder"""
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                texts[filename] = f.read()
    return texts

def create_scoring_spec(observations):
    """Create the scoring specification"""
    return [
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
    ]

def create_examples(prompt, texts, original_scores_df, num_examples=10):
    """Create examples for calibration from existing texts with scores"""
    examples = []
    
    # Select a subset of texts for calibration examples
    text_items = list(texts.items())[:num_examples]
    
    for filename, text in text_items:
        # Get the original score for this model if it exists
        score_row = original_scores_df[original_scores_df['model_name'] == filename]
        
        if not score_row.empty:
            score = float(score_row['total_score'].iloc[0])
        else:
            # Default score if not found
            score = 0.5
        
        examples.append({
            "llm_input": prompt,
            "llm_output": text,
            "score": score  # Required for calibration
        })
    
    return examples

def calibrate_scoring_system(pi, scoring_spec, examples):
    """Calibrate the scoring system"""
    print("\nStarting calibration job...")
    
    try:
        calibration_status = pi.scoring_system.calibrate.start_job(
            scoring_spec=scoring_spec,
            examples=examples,
            preference_examples=[]
        )
        
        print(f"Calibration job started with ID: {calibration_status.job_id}")
        print(f"Status: {calibration_status}")
        print("Waiting for calibration to complete...")
        
        # Stream calibration progress
        for update in stream(pi.scoring_system.calibrate, calibration_status):
            if update:
                print(f"Progress update: {update}")
        
        print("Calibration complete! Retrieving calibrated scoring spec...")
        
        # Retrieve calibrated scoring spec
        calibrated_result = pi.scoring_system.calibrate.retrieve(calibration_status.job_id)
        
        print(f"Calibration result: {calibrated_result}")
        
        if hasattr(calibrated_result, 'calibrated_scoring_spec'):
            scoring_spec_calibrated = calibrated_result.calibrated_scoring_spec
        else:
            print("Error: No calibrated_scoring_spec found in result")
            print(f"Available attributes: {dir(calibrated_result)}")
            return None
        
        return scoring_spec_calibrated
        
    except Exception as e:
        print(f"Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return None

def score_with_calibrated_spec(pi, prompt, text, scoring_spec_calibrated):
    """Score a text using calibrated scoring spec"""
    response = pi.scoring_system.score(
        llm_input=prompt,
        llm_output=text,
        scoring_spec=scoring_spec_calibrated,
    )
    return response

def main():
    # Configuration
    PI_API_KEY = input("Enter your PI API key: ").strip()
    folder_path = input("Enter the folder path containing txt files: ").strip()
    prompt_path = input("Enter the path to the prompt file (txt): ").strip()
    original_scores_csv = input("Enter path to original PI scores CSV: ").strip()
    num_calibration_examples = int(input("Number of examples to use for calibration (default 10): ").strip() or "10")
    output_csv = input("Enter output CSV filename (default: pi_scores_calibrated.csv): ").strip()
    
    if not output_csv:
        output_csv = "pi_scores_calibrated.csv"
    
    # Observations for realism question
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
    
    # Load original scores first (needed for calibration examples)
    print("Loading original scores...")
    df_original = pd.read_csv(original_scores_csv)
    
    # Create scoring spec
    print("Creating scoring spec...")
    spec = create_scoring_spec(observations)
    
    # Create calibration examples with scores
    print(f"Creating {num_calibration_examples} calibration examples with scores...")
    calib_examples = create_examples(prompt, texts, df_original, num_calibration_examples)
    
    # Calibrate
    print("Starting calibration...")
    calibrated_spec = calibrate_scoring_system(pi, spec, calib_examples)
    
    if calibrated_spec is None:
        print("\nCalibration failed! Cannot continue.")
        return
    
    # Score all texts with calibrated spec
    print("\nScoring all texts with calibrated scoring spec...")
    calibrated_results = []
    
    for i, (filename, text) in enumerate(texts.items(), 1):
        print(f"Scoring {i}/{len(texts)}: {filename}")
        
        try:
            response = score_with_calibrated_spec(pi, prompt, text, calibrated_spec)
            question_scores = response.question_scores
            
            calibrated_results.append({
                'model_name': filename,
                'calibrated_total_score': round(response.total_score, 4),
                'calibrated_realism': round(question_scores.get('Realism', 0), 4),
                'calibrated_prompt_adherence': round(question_scores.get('Prompt Adherence', 0), 4),
                'calibrated_clarity': round(question_scores.get('Clarity', 0), 4),
                'calibrated_factual_consistency': round(question_scores.get('Factual Consistency', 0), 4),
                'calibrated_completeness': round(question_scores.get('Completeness', 0), 4),
                'calibrated_technical_accuracy': round(question_scores.get('Technical Accuracy', 0), 4)
            })
            
        except Exception as e:
            print(f"Error scoring {filename}: {e}")
            continue
    
    # Merge with original scores
    print("\nMerging with original scores...")
    df_calibrated = pd.DataFrame(calibrated_results)
    
    comparison_df = pd.merge(df_original, df_calibrated, on='model_name', how='inner')
    
    # Calculate score differences
    comparison_df['score_difference'] = comparison_df['calibrated_total_score'] - comparison_df['total_score']
    
    # Save results
    comparison_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to {output_csv}")
    
    # Print comparison summary
    print("\n" + "="*70)
    print("CALIBRATION COMPARISON SUMMARY")
    print("="*70)
    print(f"Average original score: {comparison_df['total_score'].mean():.4f}")
    print(f"Average calibrated score: {comparison_df['calibrated_total_score'].mean():.4f}")
    print(f"Average score difference: {comparison_df['score_difference'].mean():.4f}")
    print(f"Max improvement: {comparison_df['score_difference'].max():.4f}")
    print(f"Max decrease: {comparison_df['score_difference'].min():.4f}")
    
    print("\nTop 5 models by calibrated score:")
    print(comparison_df.nlargest(5, 'calibrated_total_score')[['model_name', 'total_score', 'calibrated_total_score']].to_string(index=False))

if __name__ == "__main__":
    main()