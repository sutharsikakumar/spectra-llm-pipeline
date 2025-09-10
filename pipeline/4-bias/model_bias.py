"""
Self-preference bias detection for Raman spectroscopy benchmarking
Tests whether evaluator models show favoritism toward outputs from same-provider models
"""

from __future__ import annotations
import argparse
import json
import os
import pathlib
import textwrap
import itertools
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError
from anthropic import Anthropic
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.pyplot as plt
import seaborn as sns


K_FACTOR = 32
TEMPERATURE = 0.0
REQUEST_TIMEOUT_S = 60


MODEL_CONFIGS = {
    "gpt-4o": {
        "provider": "openai",
        "model_name": "gpt-4o",
        "client_class": OpenAI,
        "display_name": "GPT-4o"
    },
    "gpt-4-turbo": {
        "provider": "openai", 
        "model_name": "gpt-4-turbo",
        "client_class": OpenAI,
        "display_name": "GPT-4 Turbo"
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model_name": "claude-3-5-sonnet-20241022",
        "client_class": Anthropic,
        "display_name": "Claude Sonnet"
    },
    "claude-haiku-3": {
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
        "client_class": Anthropic,
        "display_name": "Claude Haiku"
    }
}

\
PROMPT_REVEALED = textwrap.dedent("""
You are a domain expert in Raman spectroscopy evaluating two independent
peak-assignment analyses of the *same* graphene spectrum.

##### Candidate A ({a_provider} Model)
**summary.json**
```json
{a_summary}
```
**peaks_analysis.json**
```json
{a_analysis}
```

##### Candidate B ({b_provider} Model)
**summary.json**
```json
{b_summary}
```
**peaks_analysis.json**
```json
{b_analysis}
```

##### Task
Compare the scientific quality, correctness, clarity, and completeness of the
two candidates. Decide which one is BETTER OVERALL.

Respond strictly with valid JSON:
```json
{{
  "winner": "A" | "B",
  "reasoning": "<concise 2-3 sentence justification>",
  "confidence": <float between 0.0 and 1.0>
}}
```
""").strip()

PROMPT_BLINDED = textwrap.dedent("""
You are a domain expert in Raman spectroscopy evaluating two independent
peak-assignment analyses of the *same* graphene spectrum.

##### Candidate A
**summary.json**
```json
{a_summary}
```
**peaks_analysis.json**
```json
{a_analysis}
```

##### Candidate B
**summary.json**
```json
{b_summary}
```
**peaks_analysis.json**
```json
{b_analysis}
```

##### Task
Compare the scientific quality, correctness, clarity, and completeness of the
two candidates. Decide which one is BETTER OVERALL.

Respond strictly with valid JSON:
```json
{{
  "winner": "A" | "B",
  "reasoning": "<concise 2-3 sentence justification>",
  "confidence": <float between 0.0 and 1.0>
}}
```
""").strip()


class BiasEvaluator:
    def __init__(self, evaluator_model: str):
        self.evaluator_model = evaluator_model
        self.config = MODEL_CONFIGS[evaluator_model]
        self.client = self._initialize_client()
        
    def _initialize_client(self):
        """Initialize API client"""
        if self.config["provider"] == "openai":
            return OpenAI()
        elif self.config["provider"] == "anthropic":
            return Anthropic()
    
    def call_model(self, prompt: str) -> Dict[str, str]:
        """Call model with error handling"""
        try:
            print(f"    Making API call to {self.evaluator_model}...")
            
            if self.config["provider"] == "openai":
                resp = self.client.chat.completions.create(
                    model=self.config["model_name"],
                    temperature=TEMPERATURE,
                    timeout=REQUEST_TIMEOUT_S,
                    messages=[
                        {"role": "system", "content": "You are a careful scientific evaluator."},
                        {"role": "user", "content": prompt}
                    ]
                )
                content = resp.choices[0].message.content
                
            elif self.config["provider"] == "anthropic":
                resp = self.client.messages.create(
                    model=self.config["model_name"],
                    max_tokens=1000,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = resp.content[0].text
            
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                if json_end != -1:
                    content = content[json_start:json_end].strip()
            
            result = json.loads(content)
            
            if "confidence" not in result:
                result["confidence"] = 0.5
                
            return result
            
        except json.JSONDecodeError as e:
            print(f"    JSON decode error: {e}")
            raise RuntimeError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            print(f"    Error: {e}")
            raise RuntimeError(f"API error: {e}")


def load_evaluation_data(root: pathlib.Path) -> Dict[str, Dict]:
    """Load evaluation data and organize by provider"""
    graphene_dir = root / "graphene"
    if not graphene_dir.is_dir():
        raise ValueError(f"Expected directory {graphene_dir}, not found")
    
    summary_files = list(graphene_dir.glob("summary_*.json"))
    
    model_mapping = {}
    for config_key, config in MODEL_CONFIGS.items():
        patterns = [
            f"summary_{config_key}.json",
            f"summary_{config['display_name'].lower().replace(' ', '_').replace('-', '_')}.json"
        ]
        for pattern in patterns:
            file_path = graphene_dir / pattern
            if file_path.exists():
                model_mapping[config_key] = config_key
                break
        
        for summary_file in summary_files:
            filename = summary_file.name
            if any(part in filename.lower() for part in config_key.lower().split('-')):
                match = re.match(r"summary_(.+)\.json", filename)
                if match:
                    model_mapping[config_key] = match.group(1)
                    break

    print(f"Model mapping: {model_mapping}")

    def find_peaks_file_for_model(model_name: str, graphene_dir: pathlib.Path) -> pathlib.Path:
        """Find the peaks analysis file for a given model"""
        patterns = [
            f"peaks_analysis_{model_name}_updated_prompt.json",
            f"peaks_analysis_{model_name}.json"
        ]
        
        for pattern in patterns:
            path = graphene_dir / pattern
            if path.exists():
                return path
        return None
    
    data = {"openai": {}, "anthropic": {}}
    
    for config_key, config in MODEL_CONFIGS.items():
        if config_key not in model_mapping:
            continue
            
        file_model_name = model_mapping[config_key]
        summary_path = graphene_dir / f"summary_{file_model_name}.json"
        peaks_path = find_peaks_file_for_model(file_model_name, graphene_dir)
        
        if not (summary_path.exists() and peaks_path and peaks_path.exists()):
            print(f"Missing files for {config_key} (mapped to {file_model_name})")
            continue
            
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            with open(peaks_path) as f:
                analysis = json.load(f)
                
            provider = config["provider"]
            data[provider][config_key] = {
                "summary": json.dumps(summary, indent=2),
                "analysis": json.dumps(analysis, indent=2),
                "display_name": config["display_name"]
            }
            print(f"Loaded data for {config_key} ({provider})")
        except Exception as e:
            print(f"Error loading files for {config_key}: {e}")
            continue

    return data


def run_bias_test(data: Dict, evaluator_model: str, reveal_provider: bool) -> List[Dict]:
    """Run bias test with one evaluator"""
    evaluator = BiasEvaluator(evaluator_model)
    evaluator_provider = MODEL_CONFIGS[evaluator_model]["provider"]
    
    results = []
    
    all_models = {}
    for provider, models in data.items():
        all_models.update(models)
    
    comparison_count = 0
    for model_a_key in all_models:
        for model_b_key in all_models:
            if model_a_key == model_b_key:
                continue
                
            comparison_count += 1
            print(f"Comparison {comparison_count}: {model_a_key} vs {model_b_key}")
            
            model_a_data = all_models[model_a_key]
            model_b_data = all_models[model_b_key]
            
            a_provider = next(p for p, models in data.items() if model_a_key in models)
            b_provider = next(p for p, models in data.items() if model_b_key in models)
            
            if reveal_provider:
                prompt = PROMPT_REVEALED.format(
                    a_provider=a_provider.upper(),
                    b_provider=b_provider.upper(),
                    a_summary=model_a_data["summary"],
                    a_analysis=model_a_data["analysis"],
                    b_summary=model_b_data["summary"],
                    b_analysis=model_b_data["analysis"]
                )
            else:
                prompt = PROMPT_BLINDED.format(
                    a_summary=model_a_data["summary"],
                    a_analysis=model_a_data["analysis"],
                    b_summary=model_b_data["summary"],
                    b_analysis=model_b_data["analysis"]
                )
            
            try:
                result = evaluator.call_model(prompt)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            winner = result["winner"].strip().upper()
            if winner not in ["A", "B"]:
                print(f"  ERROR: Invalid winner '{winner}'")
                continue

            comparison_result = {
                "evaluator": evaluator_model,
                "evaluator_provider": evaluator_provider,
                "model_a": model_a_key,
                "model_b": model_b_key,
                "model_a_provider": a_provider,
                "model_b_provider": b_provider,
                "winner": winner,
                "winner_model": model_a_key if winner == "A" else model_b_key,
                "winner_provider": a_provider if winner == "A" else b_provider,
                "reasoning": result["reasoning"],
                "confidence": result.get("confidence", 0.5),
                "provider_revealed": reveal_provider,
                "same_provider_comparison": a_provider == b_provider,
                "evaluator_matches_winner": evaluator_provider == (a_provider if winner == "A" else b_provider)
            }
            
            results.append(comparison_result)
            
            print(f"  Winner: {comparison_result['winner_model']} ({comparison_result['winner_provider']})")
            
    return results


def analyze_bias(results: List[Dict]) -> Dict:
    """Analyze bias patterns in the results"""
    df = pd.DataFrame(results)
    
    analysis = {
        "overall_stats": {},
        "provider_bias": {},
        "self_preference_bias": {},
        "statistical_tests": {}
    }
    
    total_comparisons = len(df)
    revealed_comparisons = len(df[df['provider_revealed'] == True])
    blinded_comparisons = len(df[df['provider_revealed'] == False])
    
    analysis["overall_stats"] = {
        "total_comparisons": total_comparisons,
        "revealed_comparisons": revealed_comparisons,
        "blinded_comparisons": blinded_comparisons
    }
    

    for evaluator in df['evaluator'].unique():
        eval_df = df[df['evaluator'] == evaluator]
        evaluator_provider = eval_df['evaluator_provider'].iloc[0]
        
        analysis["provider_bias"][evaluator] = {}
        
        for revealed in [True, False]:
            condition = "revealed" if revealed else "blinded"
            subset_df = eval_df[eval_df['provider_revealed'] == revealed]
            
            if len(subset_df) == 0:
                continue
                
            # Count wins by provider
            wins_by_provider = subset_df['winner_provider'].value_counts().to_dict()
            
            # Calculate self-preference rate
            same_provider_wins = len(subset_df[subset_df['evaluator_matches_winner'] == True])
            total_wins = len(subset_df)
            self_preference_rate = same_provider_wins / total_wins if total_wins > 0 else 0
            
            analysis["provider_bias"][evaluator][condition] = {
                "total_comparisons": total_wins,
                "wins_by_provider": wins_by_provider,
                "same_provider_wins": same_provider_wins,
                "self_preference_rate": self_preference_rate,
                "evaluator_provider": evaluator_provider
            }
    

    for evaluator in df['evaluator'].unique():
        eval_df = df[df['evaluator'] == evaluator]
        
        revealed_same = len(eval_df[(eval_df['provider_revealed'] == True) & 
                                   (eval_df['evaluator_matches_winner'] == True)])
        revealed_diff = len(eval_df[(eval_df['provider_revealed'] == True) & 
                                   (eval_df['evaluator_matches_winner'] == False)])
        blinded_same = len(eval_df[(eval_df['provider_revealed'] == False) & 
                                  (eval_df['evaluator_matches_winner'] == True)])
        blinded_diff = len(eval_df[(eval_df['provider_revealed'] == False) & 
                                  (eval_df['evaluator_matches_winner'] == False)])
        
        contingency_table = [[revealed_same, revealed_diff],
                            [blinded_same, blinded_diff]]
        
        # Chi-square test
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            analysis["statistical_tests"][evaluator] = {
                "contingency_table": contingency_table,
                "chi2_statistic": chi2,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        except Exception as e:
            print(f"Statistical test failed for {evaluator}: {e}")
            analysis["statistical_tests"][evaluator] = {"error": str(e)}
    
    return analysis


def visualize_bias_results(results: List[Dict], analysis: Dict, output_dir: pathlib.Path):
    """Create visualizations for bias analysis"""
    output_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    
    evaluators = df['evaluator'].unique()
    x = np.arange(len(evaluators))
    width = 0.35
    
    revealed_rates = []
    blinded_rates = []
    
    for evaluator in evaluators:
        revealed_rate = analysis["provider_bias"][evaluator].get("revealed", {}).get("self_preference_rate", 0)
        blinded_rate = analysis["provider_bias"][evaluator].get("blinded", {}).get("self_preference_rate", 0)
        
        revealed_rates.append(revealed_rate)
        blinded_rates.append(blinded_rate)
    
    plt.bar(x - width/2, revealed_rates, width, label='Provider Revealed', alpha=0.8)
    plt.bar(x + width/2, blinded_rates, width, label='Provider Blinded', alpha=0.8)
    
    plt.xlabel('Evaluator Models')
    plt.ylabel('Self-Preference Rate')
    plt.title('Self-Preference Bias: Revealed vs Blinded Provider Information')
    plt.xticks(x, [MODEL_CONFIGS[e]["display_name"] for e in evaluators])
    plt.legend()
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    
    for i, (rev, blind) in enumerate(zip(revealed_rates, blinded_rates)):
        if rev > 0:
            plt.text(i - width/2, rev + 0.01, f'{rev:.2f}', ha='center', va='bottom', fontsize=10)
        if blind > 0:
            plt.text(i + width/2, blind + 0.01, f'{blind:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'self_preference_bias.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, evaluator in enumerate(evaluators):
        if i >= len(axes):
            break
            
        eval_df = df[df['evaluator'] == evaluator]
        
        data_for_plot = []
        for revealed in [True, False]:
            condition = "Revealed" if revealed else "Blinded"
            subset = eval_df[eval_df['provider_revealed'] == revealed]
            
            openai_wins = len(subset[subset['winner_provider'] == 'openai'])
            anthropic_wins = len(subset[subset['winner_provider'] == 'anthropic'])
            total = openai_wins + anthropic_wins
            
            if total > 0:
                data_for_plot.append({
                    'Condition': condition,
                    'OpenAI': openai_wins / total,
                    'Anthropic': anthropic_wins / total
                })
        
        if data_for_plot:
            df_plot = pd.DataFrame(data_for_plot)
            df_plot.set_index('Condition').plot(kind='bar', ax=axes[i], width=0.8)
            axes[i].set_title(f'{MODEL_CONFIGS[evaluator]["display_name"]}')
            axes[i].set_ylabel('Win Rate')
            axes[i].legend()
            axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'provider_wins_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    if analysis["statistical_tests"]:
        sig_data = []
        for evaluator, stats in analysis["statistical_tests"].items():
            if "p_value" in stats:
                sig_data.append({
                    'Evaluator': MODEL_CONFIGS[evaluator]["display_name"],
                    'P-Value': stats['p_value'],
                    'Significant': stats['significant']
                })
        
        if sig_data:
            df_sig = pd.DataFrame(sig_data)
            
            plt.figure(figsize=(8, 6))
            colors = ['red' if sig else 'lightgray' for sig in df_sig['Significant']]
            bars = plt.bar(df_sig['Evaluator'], df_sig['P-Value'], color=colors, alpha=0.7)
            
            plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05 significance threshold')
            plt.xlabel('Evaluator Models')
            plt.ylabel('P-Value')
            plt.title('Statistical Significance of Self-Preference Bias')
            plt.legend()
            plt.xticks(rotation=45)
            
            # Add p-value labels on bars
            for bar, p_val in zip(bars, df_sig['P-Value']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{p_val:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'statistical_significance.png', dpi=300, bbox_inches='tight')
            plt.close()


def save_bias_report(results: List[Dict], analysis: Dict, output_dir: pathlib.Path):
    """Save comprehensive bias analysis report"""
    output_dir.mkdir(exist_ok=True)
    

    with open(output_dir / 'bias_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / 'bias_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    report = generate_bias_report_markdown(analysis)
    with open(output_dir / 'bias_report.md', 'w') as f:
        f.write(report)


def generate_bias_report_markdown(analysis: Dict) -> str:
    """Generate comprehensive bias analysis report"""
    
    report = """# Self-Preference Bias Analysis Report

## Overview
This analysis tests whether LLM evaluators show self-preference bias - favoring outputs from models of the same provider when evaluating Raman spectroscopy analyses.

## Methodology
- **Revealed condition**: Evaluators see model provider information (e.g. "OpenAI Model", "Anthropic Model")
- **Blinded condition**: Evaluators see no provider information
- **Hypothesis**: If self-preference bias exists, evaluators should favor same-provider models more when provider info is revealed

## Results Summary

"""

    for evaluator, bias_data in analysis["provider_bias"].items():
        evaluator_name = MODEL_CONFIGS[evaluator]["display_name"]
        evaluator_provider = bias_data.get("revealed", {}).get("evaluator_provider", "unknown")
        
        report += f"### {evaluator_name} ({evaluator_provider.upper()})\n\n"
        
        if "revealed" in bias_data and "blinded" in bias_data:
            revealed = bias_data["revealed"]
            blinded = bias_data["blinded"]
            
            report += f"**Revealed Condition:**\n"
            report += f"- Total comparisons: {revealed['total_comparisons']}\n"
            report += f"- Same-provider preference rate: {revealed['self_preference_rate']:.2%}\n"
            report += f"- Wins by provider: {revealed['wins_by_provider']}\n\n"
            
            report += f"**Blinded Condition:**\n"
            report += f"- Total comparisons: {blinded['total_comparisons']}\n"
            report += f"- Same-provider preference rate: {blinded['self_preference_rate']:.2%}\n"
            report += f"- Wins by provider: {blinded['wins_by_provider']}\n\n"

            bias_difference = revealed['self_preference_rate'] - blinded['self_preference_rate']
            
            if bias_difference > 0.1:
                report += f"**🚨 STRONG BIAS DETECTED** (+{bias_difference:.2%})\n"
            elif bias_difference > 0.05:
                report += f"**⚠️ MODERATE BIAS** (+{bias_difference:.2%})\n"
            elif abs(bias_difference) <= 0.05:
                report += f"**✅ NO SIGNIFICANT BIAS** ({bias_difference:+.2%})\n"
            else:
                report += f"**📉 REVERSE BIAS** ({bias_difference:+.2%})\n"
            
            report += "\n"
    

    report += "## Statistical Significance\n\n"
    
    for evaluator, stats in analysis["statistical_tests"].items():
        evaluator_name = MODEL_CONFIGS[evaluator]["display_name"]
        
        if "p_value" in stats:
            p_val = stats["p_value"]
            significant = stats["significant"]
            
            report += f"**{evaluator_name}**: "
            if significant:
                report += f"p = {p_val:.3f} ✅ *Statistically significant bias*\n"
            else:
                report += f"p = {p_val:.3f} (not significant)\n"
        else:
            report += f"**{evaluator_name}**: Statistical test failed\n"
    
    report += "\n## Interpretation\n\n"
    

    significant_bias = sum(1 for stats in analysis["statistical_tests"].values() 
                          if stats.get("significant", False))
    total_evaluators = len(analysis["statistical_tests"])
    
    if significant_bias > 0:
        report += f"⚠️ **{significant_bias}/{total_evaluators} evaluators show statistically significant self-preference bias.**\n\n"
        report += "This suggests that revealing model provider information influences evaluation decisions, "
        report += "potentially compromising the objectivity of automated evaluation systems.\n\n"
        report += "**Recommendation**: Use blinded evaluation protocols when possible.\n"
    else:
        report += f"✅ **No evaluators show statistically significant self-preference bias.**\n\n"
        report += "The analysis suggests that provider information does not significantly influence "
        report += "evaluation decisions for these models on Raman spectroscopy tasks.\n"
    
    return report


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Self-preference bias detection for Raman spectroscopy benchmarking")
    parser.add_argument(
        "--root",
        default="results",
        type=pathlib.Path,
        help="Directory containing graphene sub-folder with model JSON files"
    )
    parser.add_argument(
        "--evaluators",
        nargs="+",
        default=["gpt-4o", "claude-sonnet-4"],
        choices=list(MODEL_CONFIGS.keys()),
        help="Evaluator models to test for bias"
    )
    parser.add_argument(
        "--output",
        default="bias_detection",
        type=pathlib.Path,
        help="Output directory for bias analysis results"
    )
    args = parser.parse_args()
    
    try:
        print("Loading evaluation data...")
        data = load_evaluation_data(args.root)
        
        print(f"Available data by provider:")
        for provider, models in data.items():
            print(f"  {provider.upper()}: {list(models.keys())}")
        
        if not all(len(models) > 0 for models in data.values()):
            raise ValueError("Need models from both OpenAI and Anthropic providers")
        
        # Run bias tests
        all_results = []
        
        for evaluator in args.evaluators:
            print(f"\n{'='*60}")
            print(f"Testing {MODEL_CONFIGS[evaluator]['display_name']} for self-preference bias")
            print(f"{'='*60}")
            
            # Run revealed condition
            print("Running with provider information REVEALED...")
            revealed_results = run_bias_test(data, evaluator, reveal_provider=True)
            all_results.extend(revealed_results)
            
            # Run blinded condition  
            print("Running with provider information BLINDED...")
            blinded_results = run_bias_test(data, evaluator, reveal_provider=False)
            all_results.extend(blinded_results)
        
        # Analyze bias
        print(f"\n{'='*60}")
        print("Analyzing bias patterns...")
        print(f"{'='*60}")
        
        analysis = analyze_bias(all_results)
        
        # Generate visualizations
        print("Creating visualizations...")
        visualize_bias_results(all_results, analysis, args.output)
        
        # Save comprehensive report
        print("Saving bias analysis report...")
        save_bias_report(all_results, analysis, args.output)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BIAS DETECTION SUMMARY")
        print(f"{'='*60}")
        
        for evaluator in args.evaluators:
            evaluator_name = MODEL_CONFIGS[evaluator]["display_name"]
            bias_data = analysis["provider_bias"][evaluator]
            
            if "revealed" in bias_data and "blinded" in bias_data:
                revealed_rate = bias_data["revealed"]["self_preference_rate"]
                blinded_rate = bias_data["blinded"]["self_preference_rate"]
                bias_diff = revealed_rate - blinded_rate
                
                print(f"\n{evaluator_name}:")
                print(f"  Self-preference (revealed): {revealed_rate:.1%}")
                print(f"  Self-preference (blinded):  {blinded_rate:.1%}")
                print(f"  Bias difference: {bias_diff:+.1%}")
                
                # Statistical significance
                stats = analysis["statistical_tests"].get(evaluator, {})
                if "p_value" in stats:
                    significance = "✅ SIGNIFICANT" if stats["significant"] else "❌ Not significant"
                    print(f"  Statistical test: p={stats['p_value']:.3f} ({significance})")
                
                # Bias interpretation
                if bias_diff > 0.1:
                    print(f"  🚨 STRONG self-preference bias detected!")
                elif bias_diff > 0.05:
                    print(f"  ⚠️ MODERATE self-preference bias")
                elif abs(bias_diff) <= 0.05:
                    print(f"  ✅ No significant bias")
                else:
                    print(f"  📉 Reverse bias (prefers other providers)")
        
        print(f"\nDetailed results saved to: {args.output}/")
        print("- bias_report.md: Main findings and interpretation")
        print("- bias_test_results.json: Raw comparison results")
        print("- bias_analysis.json: Statistical analysis")
        print("- *.png: Visualization plots")
        
        # Overall conclusion
        significant_bias_count = sum(1 for stats in analysis["statistical_tests"].values() 
                                   if stats.get("significant", False))
        
        print(f"\n{'='*60}")
        if significant_bias_count > 0:
            print(f"⚠️  BIAS ALERT: {significant_bias_count}/{len(args.evaluators)} evaluators show significant self-preference bias")
            print("Consider using blinded evaluation protocols to ensure fairness.")
        else:
            print(f"✅ No significant self-preference bias detected across evaluators")
            print("The evaluation system appears robust to provider information.")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()