import os
from anthropic import Anthropic

def query_claude(prompt, temperature=0.7, top_p=0.9, max_tokens=2000):
    """Generate using Claude 3.5 Sonnet via Anthropic API"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set it with: export ANTHROPIC_API_KEY=your_key_here")
        return None

    client = Anthropic(api_key=api_key)

    try:
        print("\nGenerating with Claude 3.5 Sonnet...")
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Claude request failed: {e}")
        return None

def main():
    MODEL = "claude-3-7-sonnet-20250219"
    TEMPERATURE = 0.7
    TOP_P = 0.9
    MAX_TOKENS = 20000
    
    
    # Get inputs
    prompt_file = input("\nEnter path to prompt file (txt): ").strip()
    output_file = input("Enter output filename (default: version_1.3_generated.txt): ").strip()
    
    if not output_file:
        output_file = "version_1.3_generated.txt"
    
    # Read prompt
    print("\nReading prompt...")
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    print(f"Prompt length: {len(prompt)} characters")
    
    # Generate
    generated_text = query_claude(prompt, TEMPERATURE, TOP_P, MAX_TOKENS)
    
    if not generated_text:
        print("\nGeneration failed!")
        return
    
    # Save output
    print(f"\nSaving generated text to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    

if __name__ == "__main__":
    main()