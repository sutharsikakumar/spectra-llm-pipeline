"""OpenAI's GPT"""

from __future__ import annotations

import json
import os
import pathlib
import textwrap
import time
from typing import Dict, List, Counter

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError

# Fix: Load .env from the root directory
ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

PEAK_FILE = ROOT / "results" / "mos2" / "mos2_peaks.json"
OUT_FILE  = ROOT / "results" / "mos2_peaks_analysis_gpt35turbo_updated_prompt.json"
SUMMARY_FILE = ROOT / "results" / "summary_gpt35turbo_mos2.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is required")


OPENAI_MODEL = "gpt-3.5-turbo"
TEMPERATURE  = 0.3 
MAX_RETRIES  = 5   
MAX_TOKENS   = 5000 


SYSTEM_PROMPT = (
    "You are an expert Raman spectroscopy analyst with access to comprehensive "
    "spectroscopic databases and peer-reviewed literature. For each Raman peak provided, "
    "perform the following analysis:\n\n"
    
    "PEAK ANALYSIS:\n"
    "1. Provide exactly 3 possible molecular assignments ranked by likelihood "
    "(1=most likely, 2=moderately likely, 3=least likely)\n"
    "2. For each assignment, include:\n"
    "   - Vibrational mode description \n"
    "   - Brief scientific rationale based on molecular structure\n"
    "   - Literature reference supporting the assignment\n"
    "3. Select the most probable assignment as your final conclusion\n\n"
    
    "LITERATURE REQUIREMENTS:\n"
    "- Use peer-reviewed sources (journals, established spectroscopic databases)\n"
    "- Cite specific papers with author names and publication years\n"
    "- Prefer recent publications (last 20 years) when available\n"
    "- Include reference to established Raman databases (e.g., RRUFF, NIST)\n\n"
    
    "OUTPUT FORMAT:\n"
    "Respond with valid JSON containing:\n"
    "{\n"
    '  "peak_frequency": [wavenumber],\n'
    '  "assignments": [\n'
    '    {\n'
    '      "rank": 1,\n'
    '      "compound": "compound/material name",\n'
    '      "vibrational_mode": "specific mode description",\n'
    '      "rationale": "scientific explanation",\n'
    '      "literature_reference": "Author et al. (Year). Journal. DOI/Citation"\n'
    '    },\n'
    '    {\n'
    '      "rank": 2,\n'
    '      "compound": "compound/material name",\n'
    '      "vibrational_mode": "specific mode description",\n'
    '      "rationale": "scientific explanation",\n'
    '      "literature_reference": "Author et al. (Year). Journal. DOI/Citation"\n'
    '    },\n'
    '    {\n'
    '      "rank": 3,\n'
    '      "compound": "compound/material name",\n'
    '      "vibrational_mode": "specific mode description",\n'
    '      "rationale": "scientific explanation",\n'
    '      "literature_reference": "Author et al. (Year). Journal. DOI/Citation"\n'
    '    }\n'
    '  ],\n'
    '  "final_assignment": "most probable compound/material",\n'
    '  "confidence_level": "high/medium/low",\n'
    '  "summary": {\n'
    '    "analysis_overview": "brief summary of peak characteristics and assignment rationale",\n'
    '    "key_literature_findings": "summary of supporting literature evidence",\n'
    '    "alternative_considerations": "discussion of other possible assignments and why they were ranked lower"\n'
    '  },\n'
    '  "references": [\n'
    '    "Complete citation 1",\n'
    '    "Complete citation 2",\n'
    '    "Complete citation 3"\n'
    '  ]\n'
    "}\n\n"
    
    "CRITICAL: Respond ONLY with the requested JSON format. Ensure all literature "
    "references are real and verifiable. Base assignments on established spectroscopic "
    "knowledge and published research."
)



def build_prompt(peak: dict) -> str:
    """Format the user prompt for a single Raman peak (updated for enhanced format)."""
    return textwrap.dedent(
        f"""
        RAMAN PEAK ANALYSIS REQUEST

        Peak Data:
        - Wavenumber: {peak['position']:.2f} cm⁻¹
        - Intensity: {peak['intensity']:.1f} arbitrary units
        - FWHM: {peak['whm']:.2f} cm⁻¹

        Please analyze this peak and provide 3 ranked assignments with literature support
        in the JSON format specified in the system prompt.
        """
    ).strip()


def ask_openai(prompt: str, client: OpenAI) -> Dict:
    """Send a prompt to GPT-3.5, return parsed JSON (with enhanced retry logic)."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            
            content = resp.choices[0].message.content.strip()
            
            try:
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    content = content[start:end].strip()
                elif "```" in content:
                    start = content.find("```") + 3
                    end = content.find("```", start)
                    content = content[start:end].strip()
                
                return json.loads(content)
                
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                raise

        except (APIError, RateLimitError) as err:
            last_error = err
            if attempt < MAX_RETRIES:
                wait_time = min(2 ** attempt, 60)
                print(f"API error (attempt {attempt}/{MAX_RETRIES}), retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
                
        except json.JSONDecodeError as err:
            print(f"Invalid JSON response (attempt {attempt}/{MAX_RETRIES}):")
            print(f"Response: {content[:200]}...")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise ValueError(f"OpenAI returned invalid JSON after {MAX_RETRIES} attempts:\n{content}") from err

    raise RuntimeError(f"OpenAI request failed after {MAX_RETRIES} attempts") from last_error


def validate_response(result: Dict, peak: dict) -> Dict:
    """Validate and clean up the API response for enhanced format."""
    if "assignments" not in result:
        result["assignments"] = []
    if "final_assignment" not in result:
        result["final_assignment"] = "Unknown"
    if "confidence_level" not in result:
        result["confidence_level"] = "low"
    if "summary" not in result:
        result["summary"] = {
            "analysis_overview": "Analysis incomplete",
            "key_literature_findings": "No literature findings",
            "alternative_considerations": "No alternative considerations"
        }
    if "references" not in result:
        result["references"] = []
    
    assignments = result["assignments"]
    if not isinstance(assignments, list) or len(assignments) != 3:
        print(f"Warning: Invalid assignments structure for peak at {peak['position']:.2f} cm⁻¹")
        result["assignments"] = [
            {
                "rank": i+1,
                "compound": "Unassigned",
                "vibrational_mode": "Unknown",
                "rationale": "Analysis incomplete",
                "literature_reference": "No reference available"
            } 
            for i in range(3)
        ]
    
    for i, assignment in enumerate(result["assignments"]):
        if not isinstance(assignment, dict):
            result["assignments"][i] = {
                "rank": i+1,
                "compound": "Unknown",
                "vibrational_mode": "Unknown",
                "rationale": "Invalid response",
                "literature_reference": "No reference available"
            }
        else:
            assignment.setdefault("rank", i+1)
            assignment.setdefault("compound", "Unknown")
            assignment.setdefault("vibrational_mode", "Unknown")
            assignment.setdefault("rationale", "No rationale provided")
            assignment.setdefault("literature_reference", "No reference available")
    
    return result


def analyze(peaks: List[dict]) -> List[dict]:
    """Run the analysis for every peak with progress tracking."""
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=60) 
    analyses: List[dict] = []
    
    print(f"Analyzing {len(peaks)} peaks with GPT-3.5 Turbo...")
    
    for i, peak in enumerate(peaks, 1):
        print(f"Processing peak {i}/{len(peaks)} at {peak['position']:.2f} cm⁻¹...")
        
        try:
            result = ask_openai(build_prompt(peak), client)
            result = validate_response(result, peak)
            
            result.update(
                position=peak["position"],
                intensity=peak["intensity"],
                whm=peak["whm"],
                peak_index=i-1,
            )
            analyses.append(result)
            
            time.sleep(1.0) 
            
        except Exception as e:
            print(f"Error analyzing peak {i}: {e}")
            fallback = {
                "assignments": [
                    {
                        "rank": j+1,
                        "compound": "Analysis failed",
                        "vibrational_mode": "Unknown",
                        "rationale": str(e)[:100],
                        "literature_reference": "No reference available"
                    } 
                    for j in range(3)
                ],
                "final_assignment": "Analysis failed",
                "confidence_level": "low",
                "summary": {
                    "analysis_overview": f"Analysis failed: {str(e)[:100]}",
                    "key_literature_findings": "No findings due to error",
                    "alternative_considerations": "No considerations due to error"
                },
                "references": [],
                "position": peak["position"],
                "intensity": peak["intensity"],
                "whm": peak["whm"],
                "peak_index": i-1,
                "error": str(e)
            }
            analyses.append(fallback)
    
    return analyses


def analyze_material_composition(analyses: List[dict]) -> Dict:
    """Analyze the overall material composition from peak assignments."""
    from collections import Counter
    
    final_assignments = []
    confidence_levels = []
    all_compounds = []
    
    for analysis in analyses:
        if "error" not in analysis: 
            final_assignment = analysis.get("final_assignment", "Unknown")
            if final_assignment != "Unknown" and final_assignment != "Analysis failed":
                final_assignments.append(final_assignment)
            
            confidence = analysis.get("confidence_level", "low")
            confidence_levels.append(confidence)
            
            for assignment in analysis.get("assignments", []):
                compound = assignment.get("compound", "Unknown")
                if compound not in ["Unknown", "Unassigned", "Analysis failed"]:
                    all_compounds.append(compound)
    
    final_counts = Counter(final_assignments)
    compound_counts = Counter(all_compounds)
    confidence_counts = Counter(confidence_levels)
    
    most_common_material = final_counts.most_common(1)[0][0] if final_counts else "Unknown"
    most_common_count = final_counts.most_common(1)[0][1] if final_counts else 0
    
    total_peaks = len([a for a in analyses if "error" not in a])
    high_confidence_peaks = confidence_counts.get("high", 0)
    medium_confidence_peaks = confidence_counts.get("medium", 0)
    
    if total_peaks == 0:
        overall_confidence = "low"
    elif high_confidence_peaks / total_peaks >= 0.6:
        overall_confidence = "high"
    elif (high_confidence_peaks + medium_confidence_peaks) / total_peaks >= 0.5:
        overall_confidence = "medium"
    else:
        overall_confidence = "low"
    
    material_summary = {
        "identified_material": most_common_material,
        "confidence": overall_confidence,
        "supporting_peaks": most_common_count,
        "total_analyzed_peaks": total_peaks,
        "agreement_percentage": round((most_common_count / total_peaks * 100), 1) if total_peaks > 0 else 0,
        
        "detailed_analysis": {
            "primary_assignments": dict(final_counts.most_common(5)), 
            "all_detected_compounds": dict(compound_counts.most_common(10)), 
            "confidence_distribution": dict(confidence_counts),
        },
        
        "summary_text": generate_material_summary_text(
            most_common_material, 
            overall_confidence, 
            most_common_count, 
            total_peaks,
            final_counts
        ),
        
        "analysis_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "successful_peak_analyses": total_peaks,
            "failed_analyses": len(analyses) - total_peaks,
        }
    }
    
    return material_summary


def generate_material_summary_text(material: str, confidence: str, supporting_peaks: int, 
                                 total_peaks: int, all_assignments: Counter) -> str:
    """Generate a human-readable summary of the material identification."""
    
    if material == "Unknown" or total_peaks == 0:
        return ("Material identification was unsuccessful. "
                "The Raman spectrum could not be reliably matched to known materials.")
    
    agreement_pct = round((supporting_peaks / total_peaks * 100), 1)
    
    summary = f"Based on Raman spectroscopic analysis of {total_peaks} peaks, "
    
    if confidence == "high" and agreement_pct >= 70:
        summary += f"the sample is identified as **{material}** with high confidence. "
        summary += f"{supporting_peaks} out of {total_peaks} peaks ({agreement_pct}%) "
        summary += "strongly support this identification."
    
    elif confidence == "medium" or agreement_pct >= 50:
        summary += f"the sample is most likely **{material}** with moderate confidence. "
        summary += f"{supporting_peaks} out of {total_peaks} peaks ({agreement_pct}%) "
        summary += "support this identification."
    
    else:
        summary += f"the sample may be **{material}**, but confidence is low. "
        summary += f"Only {supporting_peaks} out of {total_peaks} peaks ({agreement_pct}%) "
        summary += "support this identification."
    

    alternatives = list(all_assignments.most_common(3))[1:]  
    if alternatives:
        alt_text = ", ".join([f"{mat} ({count} peaks)" for mat, count in alternatives])
        summary += f" Alternative possibilities include: {alt_text}."
    
    return summary


def main() -> None:
    print("Raman Peak Analysis with GPT-4o Turbo (Enhanced)")
    print("=" * 50)
    
    if not PEAK_FILE.exists():
        raise FileNotFoundError(f"Peak file not found: {PEAK_FILE}")

    print(f"Reading peaks from: {PEAK_FILE.relative_to(ROOT)}")
    peaks: List[dict] = json.loads(PEAK_FILE.read_text())
    print(f"Found {len(peaks)} peaks to analyze")
    

    results = {"peaks_analysis": analyze(peaks)}
    

    material_summary = analyze_material_composition(results["peaks_analysis"])
    

    results["metadata"] = {
        "model": OPENAI_MODEL,
        "temperature": TEMPERATURE,
        "total_peaks": len(peaks),
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_version": "enhanced_with_literature",
    }


    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(results, indent=2))
    

    SUMMARY_FILE.write_text(json.dumps(material_summary, indent=2))
    
    print(f"✓ Analysis complete!")
    print(f"✓ Detailed results written to: {OUT_FILE.relative_to(ROOT)}")
    print(f"✓ Material summary written to: {SUMMARY_FILE.relative_to(ROOT)}")
    print(f"✓ Analyzed {len(results['peaks_analysis'])} peaks")
    

    print("\n" + "="*60)
    print("MATERIAL IDENTIFICATION SUMMARY")
    print("="*60)
    print(f"Identified Material: {material_summary['identified_material']}")
    print(f"Confidence Level: {material_summary['confidence'].upper()}")
    print(f"Supporting Evidence: {material_summary['supporting_peaks']}/{material_summary['total_analyzed_peaks']} peaks ({material_summary['agreement_percentage']}%)")
    print(f"\nSummary: {material_summary['summary_text']}")
    print("="*60)


if __name__ == "__main__":
    main()