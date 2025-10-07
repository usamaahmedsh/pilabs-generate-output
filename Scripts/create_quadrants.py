import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_merge_scores(pi_csv, ensemble_csv):
    """Load and merge PI and ensemble scores"""
    df_pi = pd.read_csv(pi_csv)
    df_ensemble = pd.read_csv(ensemble_csv)
    
    # Merge on model_name
    merged = pd.merge(df_pi, df_ensemble, on='model_name', how='inner')
    
    return merged

def classify_quadrant(row, pi_median, consensus_median):
    """Classify each model into a quadrant"""
    pi_score = row['calibrated_total_score'] if 'calibrated_total_score' in row else row['total_score']
    consensus_score = row['final_consensus_score']
    
    if pi_score >= pi_median and consensus_score >= consensus_median:
        return "Goldilocks (High PI + High Consensus)"
    elif pi_score >= pi_median and consensus_score < consensus_median:
        return "Creative Excellence (High PI + Low Consensus)"
    elif pi_score < pi_median and consensus_score >= consensus_median:
        return "Safe Consensus (Low PI + High Consensus)"
    else:
        return "Avoid (Low PI + Low Consensus)"

def analyze_quadrants(df):
    """Analyze and categorize models into quadrants"""
    
    # Determine which PI score column to use
    pi_col = 'calibrated_total_score' if 'calibrated_total_score' in df.columns else 'total_score'
    
    # Calculate medians
    pi_median = df[pi_col].median()
    consensus_median = df['final_consensus_score'].median()
    
    print("\n" + "="*70)
    print("THRESHOLD VALUES")
    print("="*70)
    print(f"PI Score Median: {pi_median:.4f}")
    print(f"Consensus Score Median: {consensus_median:.4f}")
    
    # Classify each model
    df['quadrant'] = df.apply(lambda row: classify_quadrant(row, pi_median, consensus_median), axis=1)
    
    # Count models in each quadrant
    quadrant_counts = df['quadrant'].value_counts()
    
    print("\n" + "="*70)
    print("QUADRANT DISTRIBUTION")
    print("="*70)
    for quadrant, count in quadrant_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{quadrant}: {count} models ({percentage:.1f}%)")
    
    return df, pi_col, pi_median, consensus_median

def print_quadrant_details(df, pi_col):
    """Print top models in each quadrant"""
    
    print("\n" + "="*70)
    print("TOP MODELS BY QUADRANT")
    print("="*70)
    
    quadrants = [
        "Goldilocks (High PI + High Consensus)",
        "Creative Excellence (High PI + Low Consensus)",
        "Safe Consensus (Low PI + High Consensus)",
        "Avoid (Low PI + Low Consensus)"
    ]
    
    for quadrant in quadrants:
        quadrant_df = df[df['quadrant'] == quadrant]
        
        if len(quadrant_df) == 0:
            print(f"\n{quadrant}: No models")
            continue
        
        print(f"\n{quadrant}: {len(quadrant_df)} models")
        print("-" * 70)
        
        # Sort by PI score within quadrant
        top_models = quadrant_df.nlargest(5, pi_col)
        
        for idx, row in top_models.iterrows():
            print(f"  {row['model_name']}")
            print(f"    PI Score: {row[pi_col]:.4f} | Consensus Score: {row['final_consensus_score']:.4f}")

def plot_quadrants(df, pi_col, pi_median, consensus_median, output_file='quadrant_plot.png'):
    """Create scatter plot with quadrants"""
    
    plt.figure(figsize=(12, 8))
    
    # Define colors for each quadrant
    colors = {
        "Goldilocks (High PI + High Consensus)": '#2ecc71',  # Green
        "Creative Excellence (High PI + Low Consensus)": '#3498db',  # Blue
        "Safe Consensus (Low PI + High Consensus)": '#f39c12',  # Orange
        "Avoid (Low PI + Low Consensus)": '#e74c3c'  # Red
    }
    
    # Plot each quadrant
    for quadrant in df['quadrant'].unique():
        quadrant_data = df[df['quadrant'] == quadrant]
        plt.scatter(quadrant_data['final_consensus_score'], 
                   quadrant_data[pi_col],
                   c=colors[quadrant],
                   label=quadrant,
                   alpha=0.6,
                   s=100,
                   edgecolors='black',
                   linewidth=0.5)
    
    # Add median lines
    plt.axhline(y=pi_median, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='PI Median')
    plt.axvline(x=consensus_median, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Consensus Median')
    
    # Add quadrant labels
    y_range = df[pi_col].max() - df[pi_col].min()
    x_range = df['final_consensus_score'].max() - df['final_consensus_score'].min()
    
    plt.text(consensus_median + x_range*0.25, pi_median + y_range*0.25, 
             'GOLDILOCKS\n(Best)', 
             fontsize=12, ha='center', va='center', weight='bold', alpha=0.5)
    
    plt.text(consensus_median - x_range*0.25, pi_median + y_range*0.25, 
             'CREATIVE\nEXCELLENCE', 
             fontsize=12, ha='center', va='center', weight='bold', alpha=0.5)
    
    plt.text(consensus_median + x_range*0.25, pi_median - y_range*0.25, 
             'SAFE\nCONSENSUS', 
             fontsize=12, ha='center', va='center', weight='bold', alpha=0.5)
    
    plt.text(consensus_median - x_range*0.25, pi_median - y_range*0.25, 
             'AVOID', 
             fontsize=12, ha='center', va='center', weight='bold', alpha=0.5)
    
    plt.xlabel('Final Consensus Score', fontsize=12)
    plt.ylabel('PI Score (Quality)', fontsize=12)
    plt.title('Model Quality Analysis: PI Score vs Consensus Score', fontsize=14, weight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Quadrant plot saved to {output_file}")
    
    plt.show()

def save_quadrant_results(df, output_csv='quadrant_analysis.csv'):
    """Save results with quadrant classifications"""
    df_sorted = df.sort_values('quadrant')
    df_sorted.to_csv(output_csv, index=False)
    print(f"✓ Quadrant analysis saved to {output_csv}")

def main():
    print("="*70)
    print("QUADRANT ANALYSIS: PI SCORE VS CONSENSUS SCORE")
    print("="*70)
    
    # Get input files
    pi_csv = input("\nEnter path to PI scores CSV (calibrated or original): ").strip()
    ensemble_csv = input("Enter path to ensemble scores CSV: ").strip()
    output_csv = input("Enter output CSV filename (default: quadrant_analysis.csv): ").strip()
    output_plot = input("Enter output plot filename (default: quadrant_plot.png): ").strip()
    
    if not output_csv:
        output_csv = "quadrant_analysis.csv"
    if not output_plot:
        output_plot = "quadrant_plot.png"
    
    # Load and merge data
    print("\nLoading data...")
    df = load_and_merge_scores(pi_csv, ensemble_csv)
    print(f"Loaded {len(df)} models")
    
    # Analyze quadrants
    df, pi_col, pi_median, consensus_median = analyze_quadrants(df)
    
    # Print details
    print_quadrant_details(df, pi_col)
    
    # Save results
    print("\nSaving results...")
    save_quadrant_results(df, output_csv)
    

if __name__ == "__main__":
    main()