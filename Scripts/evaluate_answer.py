import os
import pandas as pd
from withpi import PiClient
from withpi_utils import stream

def create_evaluation_scoring_spec():
    """Create scoring specification for version 1.3 evaluation"""
    return [
        {
            "label": "Rule Adherence - No Invention",
            "question": "Does the document only include features and changes explicitly listed in the change log, without inventing additional features? Rate strictly: any invented feature = low score."
        },
        {
            "label": "Completeness - Changelog Coverage", 
            "question": "Are ALL items from the change log (Added: 3, Changed: 3, Fixed: 3, Security: 2, Performance: 3, Deprecated: 1, Documentation: 3) represented in both the document body AND revision table?"
        },
        {
            "label": "Terminology Consistency",
            "question": "Does the document maintain consistent terminology with v1.2 without introducing undefined terms?"
        },
        {
            "label": "Version Update Accuracy",
            "question": "Are all version references correctly updated (1.2→1.3, dates, changelog refs, migration paths)?"
        },
        {
            "label": "Revision Table Quality",
            "question": "Does the revision table exist and correctly map all changes to their source changelog items with proper format (Section | Change | Source)?"
        },
        {
            "label": "Technical Accuracy",
            "question": "Are technical descriptions accurate and coherent without technical errors or misrepresentations?"
        },
        {
            "label": "Breaking Changes Handling",
            "question": "Are breaking changes prominently highlighted with appropriate warnings and migration guidance?"
        },
        {
            "label": "Professional Quality",
            "question": "Is the document written in professional technical writing style appropriate for release notes, with proper formatting and structure?"
        }
    ]

def score_generated_output(pi_client, prompt, generated_output, scoring_spec):
    """Score the generated output using PI scorer"""
    print("\nScoring generated output with PI scorer...")
    
    try:
        response = pi_client.scoring_system.score(
            llm_input=prompt,
            llm_output=generated_output,
            scoring_spec=scoring_spec
        )
        
        return {
            'total_score': response.total_score,
            'question_scores': response.question_scores
        }
    except Exception as e:
        print(f"Error scoring output: {e}")
        return None

def calibrate_and_score(pi_client, prompt, generated_output, scoring_spec):
    """Calibrate scoring system and score output"""
    print("\nStarting calibration process...")
    
    # First, get initial score to use as calibration example
    initial_score = score_generated_output(pi_client, prompt, generated_output, scoring_spec)
    
    if not initial_score:
        print("Failed to get initial score for calibration")
        return None
    
    # Create calibration example
    calibration_example = {
        "llm_input": prompt,
        "llm_output": generated_output,
        "score": initial_score['total_score']
    }
    
    print(f"Using initial score {initial_score['total_score']:.4f} for calibration...")
    
    try:
        # Start calibration
        calibration_status = pi_client.scoring_system.calibrate.start_job(
            scoring_spec=scoring_spec,
            examples=[calibration_example],
            preference_examples=[]
        )
        
        
        # Stream progress
        for update in stream(pi_client.scoring_system.calibrate, calibration_status):
            if update:
                print(f"Progress: {update}")
        
        # Retrieve calibrated spec
        calibrated_result = pi_client.scoring_system.calibrate.retrieve(calibration_status.job_id)
        
        if not hasattr(calibrated_result, 'calibrated_scoring_spec'):
            print("Calibration failed - no calibrated spec returned")
            return None
        
        calibrated_spec = calibrated_result.calibrated_scoring_spec
        print("Calibration complete!")
        
        # Score with calibrated spec
        print("\nScoring with calibrated spec...")
        calibrated_response = pi_client.scoring_system.score(
            llm_input=prompt,
            llm_output=generated_output,
            scoring_spec=calibrated_spec
        )
        
        return {
            'total_score': calibrated_response.total_score,
            'question_scores': calibrated_response.question_scores
        }
        
    except Exception as e:
        print(f"Calibration error: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_scores(label, scores):
    """Print scores in a formatted way"""
    print("\n" + "="*70)
    print(f"{label}")
    print("="*70)
    print(f"Total Score: {scores['total_score']:.4f}")
    print("\nIndividual Scores:")
    for question, score in scores['question_scores'].items():
        print(f"  {question}: {score:.4f}")

def main():
    print("="*70)
    print("VERSION 1.3 AGENT EVALUATION")
    print("="*70)
    
    # Get inputs
    PI_API_KEY = input("\nEnter your PI API key: ").strip()
    prompt_file = input("Enter path to prompt file (txt): ").strip()
    generated_file = input("Enter path to generated v1.3 file (txt): ").strip()
    output_csv = input("Enter output CSV filename (default: v1.3_evaluation.csv): ").strip()
    
    if not output_csv:
        output_csv = "v1.3_evaluation.csv"
    
    # Initialize PI client
    print("\nInitializing PI client...")
    pi = PiClient(api_key=PI_API_KEY)
    
    # Read files
    print("Reading prompt...")
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    print("Reading generated output...")
    with open(generated_file, 'r', encoding='utf-8') as f:
        generated_output = f.read()
    
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Generated output length: {len(generated_output)} characters")
    
    # Create scoring spec
    scoring_spec = create_evaluation_scoring_spec()
    print(f"\nCreated {len(scoring_spec)} evaluation criteria")
    
    # Score with standard PI scorer
    standard_scores = score_generated_output(pi, prompt, generated_output, scoring_spec)
    
    if standard_scores:
        print_scores("STANDARD PI SCORES", standard_scores)
    else:
        print("Standard scoring failed!")
        return
    
    calibrated_scores = calibrate_and_score(pi, prompt, generated_output, scoring_spec)
    
    if calibrated_scores:
        print_scores("CALIBRATED PI SCORES", calibrated_scores)
    else:
        print("Calibrated scoring failed - using standard scores only")
        calibrated_scores = None
    
    
    results = []
    
    for question_label in standard_scores['question_scores'].keys():
        standard_score = standard_scores['question_scores'][question_label]
        calibrated_score = calibrated_scores['question_scores'][question_label] if calibrated_scores else None
        
        result = {
            'criterion': question_label,
            'standard_score': round(standard_score, 4),
            'calibrated_score': round(calibrated_score, 4) if calibrated_score else None,
            'score_difference': round(calibrated_score - standard_score, 4) if calibrated_score else None
        }
        results.append(result)
    
    # Add total scores
    results.append({
        'criterion': 'TOTAL SCORE',
        'standard_score': round(standard_scores['total_score'], 4),
        'calibrated_score': round(calibrated_scores['total_score'], 4) if calibrated_scores else None,
        'score_difference': round(calibrated_scores['total_score'] - standard_scores['total_score'], 4) if calibrated_scores else None
    })
    
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    

if __name__ == "__main__":
    main()
