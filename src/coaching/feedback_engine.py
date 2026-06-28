"""
Feedback engine for iRacing telemetry coaching.

Hybrid architecture:
    Layer 1. Statistical core (deterministic, reproducible)
    Layer 2. Template rendering fallback (deterministic)
    Layer 3. LLM (non-deterministic):
        • Takes the structured analysis and generates natural-language coaching
        • If LLM unavailable, step 2 output is used

Inputs:
    amateur_lap  : (N_POINTS, N_FEATURES) — single amateur lap (raw, NOT normalised)
    expert_recon : (N_POINTS, N_FEATURES) — autoencoder reconstruction of that lap (denormalised)
    mean, std    : scaler params for denormalisation
    
Outputs:
    FeedbackReport with zones, causal chains, severity, and text feedback
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv
from utils.config import N_POINTS, CHANNEL_DISPLAY_SCALE, CHANNEL_UNITS
from coaching.statistical_analysis import (compute_signed_error, detect_zones, analyse_zone_channels,
                                   detect_causal_chains, compute_zone_severity, estimate_time_loss,
                                   ChannelDeviation, CausalChain)
from coaching.template_feedback_fallback import render_zone_feedback, render_summary

load_dotenv()


#* ----------------------------- Constants ----------------------------- #

LAP_DIST = np.linspace(0, 100, N_POINTS)


# ----------------------------- Data Classes ---------------------------- #
@dataclass
class CoachingZone:
    """A segment of the lap with significant deviation from expert driving."""
    zone_id: int
    lap_pct_start: float
    lap_pct_end: float
    idx_start: int
    idx_end: int
    severity_score: float           # 0-1 overall zone severity
    dominant_channels: list         # List of ChannelDeviation, sorted by severity
    causal_chains: list             # List of CausalChain
    template_feedback: str          # Layer 2: template-based text
    llm_feedback: Optional[str] = None  # Layer 3: LLM-polished text (if available)


@dataclass
class FeedbackReport:
    """Complete coaching report for a single amateur lap."""
    n_zones: int
    overall_severity: float         # 0-1 
    estimated_time_loss_s: float    # Rough estimate of total time lost
    zones: list                     # List of CoachingZone
    summary_template: str           # Layer 2 summary
    summary_llm: Optional[str] = None  # Layer 3 summary

    def to_dict(self):
        return asdict(self)

    def to_json(self, path=None):
        d = self.to_dict()
        text = json.dumps(d, indent=2, ensure_ascii=False)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        return text


#* --------------------------- LLM-POWERED NATURAL LANGUAGE COACHING ------------------------- #

#! LLM provider configurations - GROQ DEFAULT
LLM_PROVIDERS = {
    "gemini": {
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.1-8b-instant",
    }
}

DEFAULT_PROVIDER = "groq"  

def _build_llm_prompt(zone, deviations, causal_chains):
    """
    Build a structured prompt for the LLM to generate natural coaching feedback.
    The prompt includes ALL the statistical analysis so the LLM only needs to 
    rephrase — not invent technical content.
    """
    dev_text = "\n".join([
        f"  - {d.channel}: {d.signed_mean:+.1f} {d.unit} vs expert "
        f"({'amateur exceeds expert' if d.direction == 'over' else 'amateur below expert'}), "
        f"severity {d.severity:.0%}"
        for d in deviations[:4]
    ])

    chain_text = "\n".join([
        f"  - {c.description} (confidence {c.confidence:.0%})"
        for c in causal_chains[:3]
    ]) if causal_chains else "  No clear causal pattern detected."

    prompt = f"""You are an expert racing engineer providing coaching feedback to an amateur sim-racer on iRacing.
Based on the following telemetry analysis for a segment of the lap ({zone.lap_pct_start:.1f}% to {zone.lap_pct_end:.1f}%),
write a concise, actionable coaching paragraph (3-5 sentences max).

CHANNEL DEVIATIONS (amateur vs expert reconstruction):
{dev_text}

CAUSAL ANALYSIS:
{chain_text}

RULES:
- Use concrete numbers from the data (e.g. "you are 8 km/h faster than optimal")
- Focus on what the driver should DO differently, not just what is wrong
- Use sim-racing terminology the driver would understand (braking point, apex, trail-braking, corner exit, etc.)
- Do NOT invent data not provided above — only use the numbers given
- Be encouraging but direct
- Write in second person ("you")
"""
    return prompt


def _build_summary_prompt(zones, overall_severity, total_time_loss):
    """Build a prompt for an overall lap summary."""
    zone_briefs = []
    for z in zones[:5]:
        top_ch = z.dominant_channels[0] if z.dominant_channels else None
        chain = z.causal_chains[0].description if z.causal_chains else "no clear cause"
        brief = (f"Zone {z.zone_id} ({z.lap_pct_start:.1f}%-{z.lap_pct_end:.1f}%): "
                 f"severity {z.severity_score:.0%}")
        if top_ch:
            brief += f", main issue: {top_ch.channel} {top_ch.direction} by {abs(top_ch.signed_mean):.1f} {top_ch.unit}"
        brief += f", cause: {chain}"
        zone_briefs.append(brief)

    zones_text = "\n".join(f"  - {b}" for b in zone_briefs)

    prompt = f"""You are an expert racing engineer writing a brief overall coaching summary for an amateur sim-racer.

LAP ANALYSIS:
  Overall severity: {overall_severity:.0%}
  Estimated total time loss: ~{total_time_loss:.3f}s
  
ZONES:
{zones_text}

Write a 3-4 sentence summary that:
1. Highlights the single biggest area for improvement
2. Gives one concrete action the driver should focus on in their next lap
3. Mentions something positive or encouraging
4. Use concrete numbers from the data — do NOT invent numbers
"""
    return prompt


def _create_llm_client(provider, api_key=None):
    """
    Create an LLM client for the specified provider.
    
    All providers use the OpenAI-compatible chat API format:
      - Gemini via google-genai
      - Groq via groq SDK (OpenAI-compatible)
      - OpenAI via openai SDK
    
    """
    config = LLM_PROVIDERS.get(provider, LLM_PROVIDERS[DEFAULT_PROVIDER])
    key = api_key or os.environ.get(config["env_key"])
    model = config["default_model"]

    if not key:
        return None, None

    if provider == "gemini":
        try:
            from google import genai
            client = genai.Client(api_key=key)
            return client, model
        except (ImportError, Exception):
            return None, None

    elif provider == "groq":
        try:
            from groq import Groq
            client = Groq(api_key=key)
            return client, model
        except (ImportError, Exception):
            return None, None

    return None, None # (client, model_name) or (None, None) if unavailable


def _llm_generate(client, provider, model, prompt, max_tokens=300):
    """
    Send a prompt to the LLM and return the response text.
    Handles the API differences between Gemini and OpenAI-compatible providers.
    """
    if provider == "gemini":
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "max_output_tokens": max_tokens,
                "temperature": 0.3,
            },
        )
        return response.text.strip()
    else:
        # OpenAI-compatible API (Groq, OpenAI)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()


def generate_llm_feedback(zones, overall_severity, total_time_loss,
                           provider=DEFAULT_PROVIDER, api_key=None, model=None):
    """
    Generate LLM coaching feedback for all zones IN PARALLEL.
    
    All zone prompts + the summary prompt are submitted concurrently via
    a ThreadPoolExecutor, reducing total wall-clock time from N×latency
    to ~1×latency.
    """
    client, default_model = _create_llm_client(provider, api_key)
    if client is None:
        return None, None

    use_model = model or default_model

    # Build all prompts upfront
    zone_prompts = {}
    for zone in zones:
        zone_prompts[zone.zone_id] = _build_llm_prompt(zone, zone.dominant_channels, zone.causal_chains)
    summary_prompt = _build_summary_prompt(zones, overall_severity, total_time_loss)

    zone_feedbacks = {}
    summary = None

    def _call_zone(zone_id, prompt):
        return ("zone", zone_id, _llm_generate(client, provider, use_model, prompt, max_tokens=300))

    def _call_summary(prompt):
        return ("summary", None, _llm_generate(client, provider, use_model, prompt, max_tokens=250))

    # Launch all calls in parallel (zones + summary)
    with ThreadPoolExecutor(max_workers=len(zones) + 1) as executor:
        futures = []
        for zone_id, prompt in zone_prompts.items():
            futures.append(executor.submit(_call_zone, zone_id, prompt))
        futures.append(executor.submit(_call_summary, summary_prompt))

        for future in as_completed(futures):
            try:
                call_type, zone_id, result = future.result()
                if call_type == "zone":
                    zone_feedbacks[zone_id] = result
                else:
                    summary = result
            except Exception as e:
                print(f"  LLM parallel call error: {e}")

    return zone_feedbacks, summary


#* --------------------------- MAIN: ANALYSE → LLM (primary) → TEMPLATE (fallback) ------------------------- #

def generate_feedback(amateur_raw, expert_recon_raw, expert_baseline_sq_error,
                      lap_time_s=100.0, top_k=5,
                      llm_provider=DEFAULT_PROVIDER, llm_api_key=None, llm_model=None):
    """
    Full feedback pipeline for a single amateur lap.
    
    Flow:
        1. Statistical analysis (always runs — deterministic core)
        2. Template fallback (if LLM unavailable or fails)
        3. LLM coaching (attempts Gemini/Groq)    
    Args:
        amateur_raw:       (N_POINTS, N_FEATURES) — raw amateur telemetry (denormalised)
        expert_recon_raw:  (N_POINTS, N_FEATURES) — autoencoder reconstruction (denormalised)
        expert_baseline_sq_error: (N_POINTS, N_FEATURES) — mean expert squared error (used as baseline for severity)
        lap_time_s:        lap time in seconds (for time loss estimation)
        top_k:             max coaching zones to return
        llm_provider:      "gemini" , "groq"
        llm_api_key:       API key 
        llm_model:         override default model name
        
    Returns:
        FeedbackReport
    """
    #* Statistical analysis 
    signed_error = compute_signed_error(amateur_raw, expert_recon_raw)
    sq_error = signed_error ** 2

    raw_zones = detect_zones(sq_error, window=15, top_k=top_k)
    # detect_zones returns zones sorted by track position for consistent numbering

    coaching_zones = []
    total_time_loss = 0.0

    for rank, (idx_s, idx_e, _) in enumerate(raw_zones, start=1):
        deviations = analyse_zone_channels(signed_error, sq_error, idx_s, idx_e)
        causal_chains = detect_causal_chains(signed_error, idx_s, idx_e)
        severity = compute_zone_severity(sq_error, expert_baseline_sq_error, idx_s, idx_e)
        time_loss = estimate_time_loss(signed_error, lap_time_s, idx_s, idx_e)
        total_time_loss += time_loss

        # Template fallback
        template_text = render_zone_feedback(
            zone_id=rank,
            lap_pct_start=LAP_DIST[idx_s],
            lap_pct_end=LAP_DIST[idx_e],
            deviations=deviations,
            causal_chains=causal_chains,
            severity_score=severity,
            time_loss_s=time_loss,
        )

        coaching_zones.append(CoachingZone(
            zone_id=rank,
            lap_pct_start=round(float(LAP_DIST[idx_s]), 2),
            lap_pct_end=round(float(LAP_DIST[idx_e]), 2),
            idx_start=int(idx_s),
            idx_end=int(idx_e),
            severity_score=round(severity, 3),
            dominant_channels=deviations,
            causal_chains=causal_chains,
            template_feedback=template_text,
        ))

    overall_severity = np.mean([z.severity_score for z in coaching_zones]) if coaching_zones else 0.0

    #* Template summary (always generated as fallback)
    summary_template = render_summary(coaching_zones, overall_severity, total_time_loss)

    #* LLM coaching
    llm_summary = None
    if coaching_zones:
        print(f"  Generating LLM feedback via {llm_provider}...")
        llm_zone_feedbacks, llm_summary = generate_llm_feedback(
            coaching_zones, overall_severity, total_time_loss,
            provider=llm_provider, api_key=llm_api_key, model=llm_model,
        )
        if llm_zone_feedbacks:
            for zone in coaching_zones:
                zone.llm_feedback = llm_zone_feedbacks.get(zone.zone_id)
            print(f"  ✓ LLM feedback generated successfully ({llm_provider})")
        else:
            print(f"  ✗ LLM unavailable — falling back to template engine")

    return FeedbackReport(
        n_zones=len(coaching_zones),
        overall_severity=round(float(overall_severity), 3),
        estimated_time_loss_s=round(total_time_loss, 3),
        zones=coaching_zones,
        summary_template=summary_template,
        summary_llm=llm_summary,
    )


