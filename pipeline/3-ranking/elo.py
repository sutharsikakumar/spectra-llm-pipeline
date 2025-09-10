"""
Multi-model bias assessment for Raman spectroscopy benchmarking
Compares rankings from different LLM evaluators to detect model bias
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
from scipy.stats import spearmanr, kendalltau
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
    },
    "gpt-4-turbo": {
        "provider": "openai", 
        "model_name": "gpt-4-turbo",
        "client_class": OpenAI,
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "model_name": "claude-3-5-sonnet-20241022", 
        "client_class": Anthropic,
    },
    "claude-haiku-3": {
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
        "client_class": Anthropic,
    }
}

PROMPT_TMPL = textwrap.dedent("""
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


class EloPlayer:
    def __init__(self, name: str):
        self.name = name
        self.rating = 1000
        self.wins: List[str] = []
        self.losses: List[str] = []
        self.reasons: List[str] = []
        self.match_details: List[Dict] = []

    def expected(self, opp: 'EloPlayer') -> float:
        return 1 / (1 + 10 ** ((opp.rating - self.rating) / 400))

    def update(self, opp: 'EloPlayer', score: int) -> None:
        exp = self.expected(opp)
        delta = K_FACTOR * (score - exp)
        self.rating += delta

    def reset(self):
        """Reset player stats for new evaluation"""
        self.rating = 1000
        self.wins.clear()
        self.losses.clear()
        self.reasons.clear()
        self.match_details.clear()


class MultiModelEvaluator:
    def __init__(self, models_to_test: List[str]):
        self.models = models_to_test
        self.clients = {}
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize API clients for each model"""
        for model_key in self.models:
            config = MODEL_CONFIGS[model_key]
            if config["provider"] == "openai":
                self.clients[model_key] = OpenAI()
            elif config["provider"] == "anthropic":
                self.clients[model_key] = Anthropic()
    
    def call_model(self, prompt: str, model_key: str) -> Dict[str, str]:
        """Call specific model with error handling"""
        config = MODEL_CONFIGS[model_key]
        client = self.clients[model_key]
        
        try:
            print(f"    Making API call to {model_key}...")
            
            if config["provider"] == "openai":
                resp = client.chat.completions.create(
                    model=config["model_name"],
                    temperature=TEMPERATURE,
                    timeout=REQUEST_TIMEOUT_S,
                    messages=[
                        {"role": "system", "content": "You are a careful scientific evaluator."},
                        {"role": "user", "content": prompt}
                    ]
                )
                content = resp.choices[0].message.content
                
            elif config["provider"] == "anthropic":
                resp = client.messages.create(
                    model=config["model_name"],
                    max_tokens=1000,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = resp.content[0].text
            
            print(f"    Raw response: {content[:200]}...")
            

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
            
            print(f"    Parsed JSON content: {content}")
            result = json.loads(content)
            print(f"    Parsed result: {result}")
            

            if "confidence" not in result:
                result["confidence"] = 0.5 
                
            return result
            
        except json.JSONDecodeError as e:
            print(f"    JSON decode error: {e}")
            print(f"    Raw content: {content}")
            raise RuntimeError(f"Failed to parse JSON response: {e}")
        except Exception as e:
            print(f"    Error with {model_key}: {e}")
            raise RuntimeError(f"{model_key} error: {e}")


def extract_model_name_from_summary(filename: str) -> str:
    """Extract model name from summary file like 'summary_claude_haiku35.json'"""
    match = re.match(r"summary_(.+)\.json", filename)
    if match:
        return match.group(1)
    return None


def extract_model_name_from_peaks(filename: str) -> str:
    """Extract model name from peaks analysis file with various patterns"""
    match = re.match(r"peaks_analysis_(.+)_updated_prompt\.json", filename)
    if match:
        return match.group(1)
    
    match = re.match(r"peaks_analysis_(.+)\.json", filename)
    if match:
        return match.group(1)
    
    return None


def find_peaks_file_for_model(model_name: str, graphene_dir: pathlib.Path) -> pathlib.Path:
    """Find the peaks analysis file for a given model, trying both naming patterns"""
    updated_path = graphene_dir / f"peaks_analysis_{model_name}_updated_prompt.json"
    if updated_path.exists():
        return updated_path
    regular_path = graphene_dir / f"peaks_analysis_{model_name}.json"
    if regular_path.exists():
        return regular_path
    
    return None


def load_evaluation_data(root: pathlib.Path) -> Dict[str, Dict]:
    """Load all evaluation data for models"""
    graphene_dir = root / "graphene"
    if not graphene_dir.is_dir():
        raise ValueError(f"Expected directory {graphene_dir}, not found")
    
    summary_files = list(graphene_dir.glob("summary_*.json"))
    peaks_files = list(graphene_dir.glob("peaks_analysis_*.json"))

    print(f"Found {len(summary_files)} summary files and {len(peaks_files)} peaks analysis files")

    summary_models = set()
    for f in summary_files:
        model_name = extract_model_name_from_summary(f.name)
        if model_name:
            summary_models.add(model_name)

    peaks_models = set()
    for f in peaks_files:
        model_name = extract_model_name_from_peaks(f.name)
        if model_name:
            peaks_models.add(model_name)

    model_names = summary_models.intersection(peaks_models)
    print(f"Detected {len(model_names)} complete models: {sorted(model_names)}")

    if len(model_names) < 2:
        raise ValueError(f"Need at least 2 complete models, found {len(model_names)}")


    data = {}
    for name in model_names:
        summary_path = graphene_dir / f"summary_{name}.json"
        peaks_path = find_peaks_file_for_model(name, graphene_dir)
        
        if not (summary_path.exists() and peaks_path and peaks_path.exists()):
            continue
            
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            with open(peaks_path) as f:
                analysis = json.load(f)
            data[name] = {
                "summary": json.dumps(summary, indent=2),
                "analysis": json.dumps(analysis, indent=2)
            }
            print(f"Loaded data for {name}")
        except Exception as e:
            print(f"Error loading files for {name}: {e}")
            continue

    return data


def run_single_evaluator_benchmark(data: Dict, evaluator_model: str) -> Dict[str, EloPlayer]:
    """Run benchmark with single evaluator model"""
    evaluator = MultiModelEvaluator([evaluator_model])
    
    players = {name: EloPlayer(name) for name in data.keys()}
    
    print(f"\nRunning comparisons with evaluator: {evaluator_model}")
    print(f"Comparing {len(players)} models: {sorted(players.keys())}")

    comparison_count = 0
    total_comparisons = len(list(itertools.combinations(players.keys(), 2)))
    
    for a, b in itertools.combinations(players.keys(), 2):
        comparison_count += 1
        print(f"Comparison {comparison_count}/{total_comparisons}: {a} vs {b}")
        
        a_data, b_data = data[a], data[b]
        prompt = PROMPT_TMPL.format(
            a_summary=a_data["summary"],
            a_analysis=a_data["analysis"],
            b_summary=b_data["summary"],
            b_analysis=b_data["analysis"]
        )
        
        try:
            result = evaluator.call_model(prompt, evaluator_model)
        except Exception as e:
            print(f"  ERROR calling {evaluator_model} for {a} vs {b}: {e}")
            continue

        if not all(key in result for key in ["winner", "reasoning"]):
            print(f"  ERROR: Missing required keys in result: {result}")
            continue

        winner = result["winner"].strip().upper()
        reasoning = result["reasoning"]
        confidence = result.get("confidence", 0.5)
        
        if winner == "A":
            old_a_rating = players[a].rating
            old_b_rating = players[b].rating
            
            players[a].update(players[b], 1)
            players[b].update(players[a], 0)
            players[a].wins.append(b)
            players[b].losses.append(a)
            
            print(f"  Winner: {a}")
            print(f"    {a}: {old_a_rating:.1f} -> {players[a].rating:.1f}")
            print(f"    {b}: {old_b_rating:.1f} -> {players[b].rating:.1f}")
            
        elif winner == "B":
            old_a_rating = players[a].rating
            old_b_rating = players[b].rating
            
            players[a].update(players[b], 0)
            players[b].update(players[a], 1)
            players[b].wins.append(a)
            players[a].losses.append(b)
            
            print(f"  Winner: {b}")
            print(f"    {a}: {old_a_rating:.1f} -> {players[a].rating:.1f}")
            print(f"    {b}: {old_b_rating:.1f} -> {players[b].rating:.1f}")
        else:
            print(f"  ERROR: Invalid winner value: '{winner}'")
            continue

        match_detail = {
            "opponent": b if winner == "A" else a,
            "won": winner == "A" if a in players else winner == "B",
            "reasoning": reasoning,
            "confidence": confidence,
            "evaluator": evaluator_model
        }
        players[a].match_details.append(match_detail.copy())
        players[b].match_details.append({
            **match_detail,
            "opponent": a if winner == "A" else b,
            "won": not match_detail["won"]
        })

    return players


def calculate_bias_metrics(results: Dict[str, Dict[str, EloPlayer]]) -> Dict:
    """Calculate various bias metrics across evaluators"""
    evaluators = list(results.keys())
    model_names = list(results[evaluators[0]].keys())

    rankings = {}
    ratings = {}
    
    for evaluator in evaluators:
        evaluator_results = results[evaluator]
        sorted_models = sorted(evaluator_results.keys(), 
                              key=lambda x: evaluator_results[x].rating, reverse=True)
        
        rankings[evaluator] = {model: rank+1 for rank, model in enumerate(sorted_models)}
        ratings[evaluator] = {model: evaluator_results[model].rating for model in model_names}
    

    bias_metrics = {
        "ranking_correlations": {},
        "rating_correlations": {},
        "rank_disagreement": {},
        "confidence_analysis": {},
        "evaluator_agreement": {}
    }
    

    for i, eval1 in enumerate(evaluators):
        for eval2 in evaluators[i+1:]:

            ranks1 = [rankings[eval1][model] for model in model_names]
            ranks2 = [rankings[eval2][model] for model in model_names]
            
            spearman_corr, spearman_p = spearmanr(ranks1, ranks2)
            kendall_corr, kendall_p = kendalltau(ranks1, ranks2)
            
            bias_metrics["ranking_correlations"][f"{eval1}_vs_{eval2}"] = {
                "spearman": {"correlation": spearman_corr, "p_value": spearman_p},
                "kendall": {"correlation": kendall_corr, "p_value": kendall_p}
            }
            

            ratings1 = [ratings[eval1][model] for model in model_names]
            ratings2 = [ratings[eval2][model] for model in model_names]
            
            rating_corr = np.corrcoef(ratings1, ratings2)[0, 1]
            bias_metrics["rating_correlations"][f"{eval1}_vs_{eval2}"] = rating_corr
            

            rank_diff = np.mean([abs(rankings[eval1][model] - rankings[eval2][model]) 
                               for model in model_names])
            bias_metrics["rank_disagreement"][f"{eval1}_vs_{eval2}"] = rank_diff
    
    return bias_metrics


def visualize_bias_analysis(results: Dict, bias_metrics: Dict, output_dir: pathlib.Path):
    """Create visualizations for bias analysis"""
    output_dir.mkdir(exist_ok=True)
    
    evaluators = list(results.keys())
    model_names = list(results[evaluators[0]].keys())
    

    plt.figure(figsize=(12, 8))
    
    for i, evaluator in enumerate(evaluators):
        ratings = [results[evaluator][model].rating for model in model_names]
        plt.subplot(2, 2, i+1)
        plt.bar(range(len(model_names)), ratings, alpha=0.7)
        plt.title(f'ELO Ratings - {evaluator}')
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.ylabel('ELO Rating')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'elo_ratings_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    ranking_matrix = np.zeros((len(evaluators), len(model_names)))
    for i, evaluator in enumerate(evaluators):
        sorted_models = sorted(results[evaluator].keys(), 
                              key=lambda x: results[evaluator][x].rating, reverse=True)
        for j, model in enumerate(sorted_models):
            model_idx = model_names.index(model)
            ranking_matrix[i, model_idx] = j + 1
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(ranking_matrix, 
                xticklabels=model_names, 
                yticklabels=evaluators,
                annot=True, 
                fmt='.0f', 
                cmap='viridis_r',
                cbar_kws={'label': 'Rank (1=best)'})
    plt.title('Model Rankings Across Evaluators')
    plt.xlabel('Models')
    plt.ylabel('Evaluators')
    plt.tight_layout()
    plt.savefig(output_dir / 'ranking_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    

    corr_data = []
    for comparison, corr_info in bias_metrics["ranking_correlations"].items():
        corr_data.append({
            "Comparison": comparison,
            "Spearman": corr_info["spearman"]["correlation"],
            "Kendall": corr_info["kendall"]["correlation"]
        })
    
    if corr_data:
        df_corr = pd.DataFrame(corr_data)
        plt.figure(figsize=(10, 6))
        x = np.arange(len(df_corr))
        width = 0.35
        
        plt.bar(x - width/2, df_corr["Spearman"], width, label='Spearman', alpha=0.7)
        plt.bar(x + width/2, df_corr["Kendall"], width, label='Kendall', alpha=0.7)
        
        plt.xlabel('Evaluator Pairs')
        plt.ylabel('Correlation')
        plt.title('Ranking Correlations Between Evaluators')
        plt.xticks(x, df_corr["Comparison"], rotation=45)
        plt.legend()
        plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='High correlation threshold')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def save_bias_report(results: Dict, bias_metrics: Dict, output_dir: pathlib.Path):
    """Save comprehensive bias analysis report"""
    output_dir.mkdir(exist_ok=True)
    

    detailed_results = {}
    for evaluator, players in results.items():
        detailed_results[evaluator] = {
            model: {
                "rating": player.rating,
                "wins": len(player.wins),
                "losses": len(player.losses),
                "win_rate": len(player.wins) / (len(player.wins) + len(player.losses)) if (len(player.wins) + len(player.losses)) > 0 else 0,
                "match_details": player.match_details
            }
            for model, player in players.items()
        }
    
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    

    with open(output_dir / 'bias_metrics.json', 'w') as f:
        json.dump(bias_metrics, f, indent=2)
    

    report_content = generate_bias_report_markdown(results, bias_metrics)
    with open(output_dir / 'bias_analysis_report.md', 'w') as f:
        f.write(report_content)


def generate_bias_report_markdown(results: Dict, bias_metrics: Dict) -> str:
    """Generate comprehensive markdown report"""
    evaluators = list(results.keys())
    model_names = list(results[evaluators[0]].keys())
    
    report = f"""# Model Bias Analysis Report

## Overview
This report analyzes potential bias in Raman spectroscopy evaluation across {len(evaluators)} different evaluator models:
{chr(10).join(f'- {eval_model}' for eval_model in evaluators)}

Evaluated on {len(model_names)} candidate models:
{chr(10).join(f'- {model}' for model in sorted(model_names))}

## Rankings Summary

"""
    

    report += "| Model |"
    for evaluator in evaluators:
        report += f" {evaluator} |"
    report += "\n|-------|"
    for _ in evaluators:
        report += "----------|"
    report += "\n"
    
    for model in sorted(model_names):
        report += f"| {model} |"
        for evaluator in evaluators:
            sorted_models = sorted(results[evaluator].keys(), 
                                  key=lambda x: results[evaluator][x].rating, reverse=True)
            rank = sorted_models.index(model) + 1
            rating = results[evaluator][model].rating
            report += f" #{rank} ({rating:.1f}) |"
        report += "\n"
    

    report += "\n## Bias Analysis\n\n### Ranking Correlations\n\n"
    
    for comparison, corr_info in bias_metrics["ranking_correlations"].items():
        spearman = corr_info["spearman"]["correlation"]
        kendall = corr_info["kendall"]["correlation"]
        report += f"**{comparison}**:\n"
        report += f"- Spearman correlation: {spearman:.3f}\n"
        report += f"- Kendall correlation: {kendall:.3f}\n\n"
    

    report += "### Interpretation\n\n"
    avg_spearman = np.mean([corr_info["spearman"]["correlation"] 
                           for corr_info in bias_metrics["ranking_correlations"].values()])
    
    if avg_spearman > 0.8:
        report += "**Low Bias**: High correlation between evaluators suggests consistent rankings.\n"
    elif avg_spearman > 0.6:
        report += "**Moderate Bias**: Moderate correlation indicates some evaluator disagreement.\n"
    else:
        report += "**High Bias**: Low correlation suggests significant evaluator disagreement.\n"
    
    report += f"\nAverage Spearman correlation: {avg_spearman:.3f}\n"
    
    return report


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Multi-model bias assessment for Raman spectroscopy benchmarking")
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
        help="Evaluator models to use for bias assessment"
    )
    parser.add_argument(
        "--output",
        default="bias_analysis",
        type=pathlib.Path,
        help="Output directory for bias analysis results"
    )
    args = parser.parse_args()
    
    try:
        print("Loading evaluation data...")
        data = load_evaluation_data(args.root)
        
        print(f"Running bias assessment with evaluators: {args.evaluators}")
        results = {}
        
        for evaluator in args.evaluators:
            print(f"\n{'='*50}")
            print(f"Running evaluation with {evaluator}")
            print(f"{'='*50}")
            
            results[evaluator] = run_single_evaluator_benchmark(data, evaluator)
            

            ranking = sorted(results[evaluator].values(), key=lambda p: p.rating, reverse=True)
            print(f"\n=== RANKING FROM {evaluator.upper()} ===")
            for i, p in enumerate(ranking, 1):
                mark = "🏆" if i == 1 else "⭐" if i <= 3 else ""
                print(f"{i:2d}. {p.name:<20} {p.rating:7.1f} ({len(p.wins)}W-{len(p.losses)}L) {mark}")
        

        print(f"\n{'='*50}")
        print("Calculating bias metrics...")
        print(f"{'='*50}")
        
        bias_metrics = calculate_bias_metrics(results)
        

        print("Generating visualizations...")
        visualize_bias_analysis(results, bias_metrics, args.output)
        

        print("Saving bias analysis report...")
        save_bias_report(results, bias_metrics, args.output)
        

        print(f"\n{'='*50}")
        print("BIAS ANALYSIS SUMMARY")
        print(f"{'='*50}")
        
        avg_spearman = np.mean([corr_info["spearman"]["correlation"] 
                               for corr_info in bias_metrics["ranking_correlations"].values()])
        
        print(f"Average ranking correlation (Spearman): {avg_spearman:.3f}")
        
        if avg_spearman > 0.8:
            print("Low bias detected - evaluators show high agreement")
        elif avg_spearman > 0.6:
            print("Moderate bias detected - evaluators show some disagreement")  
        else:
            print("High bias detected - significant evaluator disagreement")
            
        print(f"\nDetailed results saved to: {args.output}/")
        print("- bias_analysis_report.md: Main findings")
        print("- detailed_results.json: Complete evaluation data")
        print("- *.png: Visualization plots")
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()