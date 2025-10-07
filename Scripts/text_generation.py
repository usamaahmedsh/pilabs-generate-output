import os
from openai import OpenAI
import anthropic
import requests

# Models    

# OpenAI GPT-4o
def query_openai(prompt, temperature=0.7, top_p=1.0, max_tokens=20000):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


# Anthropic Claude 3.5
def query_claude(prompt, temperature=0.7, top_p=1.0, max_tokens=20000):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    kwargs = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    # Add top_p if not default
    if top_p != 1.0:
        kwargs["top_p"] = top_p
    
    response = client.messages.create(**kwargs)
    return response.content[0].text


# Llama-3.3 70B (Hugging Face API)
def query_llama(prompt, temperature=0.7, top_p=1.0, max_tokens=20000):
    hf_token = os.getenv("HF_API_KEY")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct:fireworks-ai",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Llama request failed: {e}")
        return None


def print_and_save(model_name, params, output, directory="outputs"):
    """
    Save output with detailed parameter naming
    params: dict with keys like 'temperature', 'top_p', 'top_k', 'max_tokens'
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Build filename from parameters
    param_str = "_".join([f"{k}-{v}" for k, v in params.items() if v is not None])
    filename = os.path.join(directory, f"{model_name}_{param_str}.txt")
    
    print(f"\n--- {model_name} ({param_str}) ---")
    print(output[:200] + "..." if output and len(output) > 200 else (output if output else "No output returned"))
    
    if output:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Saved to {filename}")


# Prompts
best_prompt = '''You are tasked with creating two realistic documentation files for version 1.2 of a hypothetical programming language called Brush. 
Generate the text for these files:

VERSION_1.2 - A formal version release document
CHANGELOG_1.2 - A technical changelog with bug fixes and updates

CHANGELOG_1.2.md Requirements
Structure & Format

Use semantic versioning: ## [1.2.0] - YYYY-MM-DD format at the top
Organize changes into categories (not all must be present):

Added - New features
Changed - Changes to existing functionality
Fixed - Bug fixes
Security - Security patches
Deprecated - Soon-to-be removed features
Performance - Optimizations
Documentation - Doc updates



Content Guidelines (7-15 tickets total)

Each ticket format:

Start with inconsistent action verbs: "Fixed", "Fix", "Resolved", "Updated", "Update", "Patched", etc.
Maximum 3 sentences per ticket
Pattern: <What was fixed/changed> + <Technical detail> + <Optional: why/impact>


Technical authenticity:

Include issue/ticket references: (#1847), [BRH-302], (fixes #203)
Sprinkle function names: parseExpression(), Token.IDENTIFIER, gc_collect()
Include error codes: E0308, SEGFAULT_0x7f, ERR_INVALID_TOKEN
Use stack trace snippets occasionally: at brush.runtime.eval:142
Reference file paths: src/compiler/lexer.br, stdlib/io.brush


Attribution (20-30 percent of tickets):

Format: Reported by John Smith <jsmith@email.com>
Or: Fixed by Sarah Chen <s.chen@brushlang.org>
Or: Thanks to Mike Torres for reporting


Style characteristics:

Include 1-2 minor typos or grammatical inconsistencies (missing period, inconsistent capitalization)
No consistency in verb tense or format between tickets
Some tickets very technical, others more user-friendly
Occasional passive voice: "Memory leak was addressed"



Example Ticket Styles
- Fixed compiler crash when using nested lambda expressions with closure capture. The issue occurred in `eval_context()` when scope resolution failed for shadowed variables (fixes #1847).

- Update documentation for async/await syntax, previous examples was using deprecated callback style

- Resolved SEGFAULT in garbage collector during concurrent object allocation. Reported by Emma Rodriguez <e.rodriguez@techcorp.io>

- Performance improvement in string interpolation - now 23 percent faster for strings with 10+ variables (#2301)

VERSION_1.2.md Requirements
Structure & Format

Start with: # Brush Language - Version 1.2.0 Release Notes
Include release date: Released: January 15, 2025
Length: 2-3x longer than changelog (more verbose, explanatory)

Content Guidelines

Tone: Formal and professional with occasional enthusiasm for major features

Example: "We're excited to introduce..." (use sparingly, 1-2 times max)
Mostly stick to: "This release includes...", "Version 1.2 brings...", "Developers can now..."


Section structure:

Overview - Brief summary of the release (2-3 paragraphs)
Highlights - Major features (3-5 items, more detailed than changelog)
Breaking Changes - If any (mark with WARNING)
Bug Fixes - Summary reference to changelog
Performance Improvements - With metrics if applicable
Deprecation Notices - What will be removed in future versions
Installation & Upgrade - Brief migration notes
Dependencies - Updated library versions


Writing pattern (per feature):

Sentence 1: What the feature does (user-facing benefit)
Sentence 2: How it works (technical mechanism)
Sentence 3: Example or edge case detail (optional)
Maximum 3 sentences per point


Technical details to include:

Function signatures: brush.async.timeout(duration: int, callback: fn)
Code examples in fenced blocks (2-3 examples throughout)
Performance metrics: "15 percent reduction in compile time", "40 percent smaller binary size"
Version dependencies: "Requires LLVM 15.0+", "Compatible with GCC 11+"
File size/memory stats: "Runtime footprint reduced from 8MB to 5.2MB"


Cross-referencing:

Reference the changelog: "For a complete list of fixes, see CHANGELOG_1.2.md"
Reference issue tracker: "See issues #1840-#1891 for details"


Migration guidance (if breaking changes exist):

"Developers upgrading from 1.1.x should note..."
Code snippets showing before/after syntax



Example Content Style
## Improved Type Inference System

Version 1.2 introduces enhanced type inference for generic functions, significantly reducing the need for explicit type annotations. The compiler now performs bidirectional type checking during the constraint resolution phase, allowing it to infer complex generic types from context. This improvement is particularly noticeable in functional programming patterns involving higher-order functions and closures.

## Breaking Changes

 **String Encoding Change**: The default string encoding has changed from UTF-8 to UTF-16 for Windows compatibility. Existing code that relies on byte-level string operations may need adjustment. See migration guide in docs/migration/1.1-to-1.2.md.

General Authenticity Guidelines

Be inconsistent in minor ways - Real docs have small irregularities
Use realistic timestamps - Recent dates, weekday releases common
Include version compatibility notes - "Brush 1.2 is compatible with 1.1.x modules"
Reference realistic tooling - Git commits, CI/CD pipelines, test coverage
Vary sentence length - Mix short and long sentences naturally
Include 1-2 typos maximum - Too many hurt credibility
Use domain-appropriate jargon - Compile-time, runtime, heap, stack, parser, lexer, AST
Make up realistic issue numbers - Use ranges like #1840-#1950 for v1.2


Output Format
Generate both files with clear markdown formatting. Use proper headers, code blocks, and lists. Make the version file noticeably more polished and formal than the changelog, which should feel more like developer notes.  
'''

medium_prompt = '''
Create a VERSION_1.2.md and CHANGELOG_1.2.md for a programming language called "Brush".
CHANGELOG Requirements:

Use semantic versioning format: [1.2.0] - Date
Include 7-15 bug fixes/updates organized by categories like: Added, Fixed, Changed, Security, Performance, Documentation
Each entry should be 1-3 sentences
Use inconsistent action verbs (Fixed, Fix, Update, Updated, Resolved, etc.)
Include technical details like function names, error codes, and file paths
Add issue numbers like (#1234) or [BRH-123]
About 25 percent of entries should credit contributors with names and emails
Include 1-2 minor typos to make it realistic
Make some entries very technical and others more user-friendly

VERSION FILE Requirements:

Title: "Brush Language - Version 1.2.0 Release Notes"
Include release date
Should be 2-3x longer than the changelog
Formal tone with occasional enthusiasm for big features
Include these sections: Overview, Highlights, Breaking Changes (if any), Bug Fixes, Performance Improvements, Dependencies
Each feature description should explain what it does and how it works (1-3 sentences each)
Include technical details like function signatures, performance metrics (percentages), and version requirements
Add code examples in markdown code blocks
Reference the changelog file
Include migration notes if there are breaking changes

Make both files look like real open-source project documentation with appropriate technical jargon and realistic formatting inconsistencies.
'''

worst_prompt = '''
Write a version 1.2 file and changelog for a programming language called Brush. Make it look realistic with bug fixes and new features. Include some technical stuff and make the version file longer than the changelog.
'''


if __name__ == "__main__":
    # Grid search parameters
    temperatures = [0.5, 0.7, 1.0, 1.5]
    top_p_values = [0.5, 0.7, 0.9, 1.0]
    max_tokens_values = [2000, 4000, 6000, 8000, 10000]
    
    # Choose which prompt to use
    prompt = best_prompt
    
    total_runs = len(temperatures) * len(top_p_values) * len(max_tokens_values) * 3
    current_run = 0
    
    # Grid search
    for temp in temperatures:
        for top_p in top_p_values:
            for max_tok in max_tokens_values:
                current_run += 1
                
                # OpenAI GPT-4o
                print(f"\n[{current_run}/{total_runs}] Running OpenAI GPT-4o...")
                params_openai = {
                    "temp": temp,
                    "top_p": top_p,
                    "max_tok": max_tok
                }
                output_openai = query_openai(prompt, temperature=temp, top_p=top_p, max_tokens=max_tok)
                print_and_save("OpenAI_GPT-4o", params_openai, output_openai)
                
                current_run += 1
                
                # Anthropic Claude 3.5
                print(f"\n[{current_run}/{total_runs}] Running Claude 3.5...")
                params_claude = {
                    "temp": temp,
                    "top_p": top_p,
                    "max_tok": max_tok
                }
                output_claude = query_claude(prompt, temperature=temp, top_p=top_p, max_tokens=max_tok)
                print_and_save("Claude_3.5", params_claude, output_claude)
                
                current_run += 1
                
                # Llama 3.3
                print(f"\n[{current_run}/{total_runs}] Running Llama 3.3...")
                params_llama = {
                    "temp": temp,
                    "top_p": top_p,
                    "max_tok": max_tok
                }
                output_llama = query_llama(prompt, temperature=temp, top_p=top_p, max_tokens=max_tok)
                print_and_save("Llama-3.3_70B", params_llama, output_llama)
    
