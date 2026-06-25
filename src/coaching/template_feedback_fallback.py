"""
Template-based feedback rendering for iRacing telemetry coaching.
This module provides functions to generate structured coaching feedback based on the statistical analysis of telemetry deviations. 
It serves as a fallback mechanism when LLM-based feedback is unavailable, ensuring that users still receive actionable insights.
"""

# Direction descriptions for each channel
CHANNEL_FEEDBACK = {
    ("Brake", "under"): "You are braking too late or with insufficient pressure",
    ("Brake", "over"):  "You are braking too early or with excessive pressure",
    ("Throttle", "under"): "You are applying throttle too late on corner exit",
    ("Throttle", "over"):  "You are applying throttle too aggressively or too early",
    ("Speed", "under"): "You are carrying less speed than the experts in this section",
    ("Speed", "over"):  "You are carrying more speed than optimal in this section",
    ("SteeringWheelAngle", "under"): "You could use more steering input through this section",
    ("SteeringWheelAngle", "over"):  "You are oversteering — using more lock than needed",
    ("RPM", "under"): "Engine RPM is lower than expected — check your gear selection",
    ("RPM", "over"):  "Engine RPM is higher than expected — consider upshifting earlier",
    ("Gear", "under"): "You may be in a higher gear than optimal",
    ("Gear", "over"):  "You may be in a lower gear than optimal",
    ("LatAccel", "under"): "Lateral load is below the expected level — you may be under-driving the corner",
    ("LatAccel", "over"):  "Lateral load exceeds the expected level — the car is being pushed beyond the grip limit",
    ("LongAccel", "under"): "Longitudinal acceleration is below optimal — you are not maximising traction",
    ("LongAccel", "over"):  "Longitudinal forces are higher than expected — potential lock-up or wheelspin",
    ("VertAccel", "under"): "Vertical load is lower than expected in this section",
    ("VertAccel", "over"):  "Vertical load is higher than expected — possible kerb impact or bump",
    ("YawRate", "under"): "The car is rotating less than expected — possible understeer",
    ("YawRate", "over"):  "The car is rotating more than expected — possible oversteer or snap",
}

# Actionable tips mapped to causal chain patterns
CAUSAL_TIPS = {
    ("Brake", "under", "Speed", "over"):
        "Try initiating your braking earlier and with more initial pressure to scrub speed before the turn-in point.",
    ("Speed", "over", "LatAccel", "over"):
        "The excess entry speed is overloading the front tyres. Carry less speed in to maintain grip.",
    ("Throttle", "over", "YawRate", "over"):
        "Apply throttle more progressively on exit to avoid destabilising the rear.",
    ("Throttle", "over", "YawRate", "under"):
        "Your aggressive throttle is pushing the car wide. Be smoother on application.",
    ("Brake", "over", "Speed", "under"):
        "You can brake later and carry more speed into the corner — trust the car's braking capability.",
    ("Speed", "under", "Throttle", "over"):
        "Low mid-corner speed is tempting you to get on throttle early. Focus on carrying more apex speed instead.",
    ("SteeringWheelAngle", "over", "Speed", "under"):
        "Reduce steering angle — a straighter wheel lets the car roll faster. Work on your turn-in point.",
    ("Throttle", "under", "Speed", "under"):
        "Get on the throttle earlier and more decisively on corner exit to maximise straight-line speed.",
}


def render_zone_feedback(zone_id, lap_pct_start, lap_pct_end, deviations, causal_chains, 
                          severity_score, time_loss_s):
    """
    Render template-based feedback text for a single coaching zone.
    """
    lines = []
    lines.append(f"━━━ Zone {zone_id} ({lap_pct_start:.1f}% – {lap_pct_end:.1f}% of lap) ━━━")
    lines.append(f"  Severity: {'█' * int(severity_score * 10)}{'░' * (10 - int(severity_score * 10))} ({severity_score:.0%})")
    if time_loss_s > 0.01:
        lines.append(f"  Estimated time impact: ~{time_loss_s:.3f}s")
    lines.append("")

    # Top 3 channel deviations (skip negligible ones)
    significant = [d for d in deviations if d.severity > 0.15][:3]
    if significant:
        lines.append("  Key deviations:")
        for d in significant:
            arrow = "▲" if d.direction == "over" else "▼"
            feedback_key = (d.channel, d.direction)
            desc = CHANNEL_FEEDBACK.get(feedback_key, f"{d.channel} deviates from expert pattern")
            lines.append(f"    {arrow} {d.channel}: {desc}")
            lines.append(f"      Deviation: {abs(d.signed_mean):.1f} {d.unit} {'above' if d.direction == 'over' else 'below'} expert level")
        lines.append("")

    # Causal chains
    if causal_chains:
        lines.append("  Root cause analysis:")
        for chain in causal_chains[:2]:  # Top 2 chains
            lines.append(f"    → {chain.description} (confidence: {chain.confidence:.0%})")
            tip_key = (chain.cause_channel, chain.cause_direction,
                       chain.effect_channel, chain.effect_direction)
            tip = CAUSAL_TIPS.get(tip_key)
            if tip:
                lines.append(f"    💡 Tip: {tip}")
        lines.append("")

    return "\n".join(lines)


def render_summary(zones, overall_severity, total_time_loss):
    """Render a summary of all coaching zones."""
    lines = []
    lines.append("═══════════════════════════════════════════════════")
    lines.append("           COACHING REPORT SUMMARY")
    lines.append("═══════════════════════════════════════════════════")
    lines.append(f"  Zones analysed: {len(zones)}")
    lines.append(f"  Overall severity: {overall_severity:.0%}")
    lines.append(f"  Total estimated time impact: ~{total_time_loss:.3f}s")
    lines.append("")

    if zones:
        # Find the single biggest improvement opportunity (highest severity)
        worst_zone = max(zones, key=lambda z: z.severity_score)
        if worst_zone.dominant_channels:
            top_ch = worst_zone.dominant_channels[0]
            lines.append(f"  🎯 Priority focus: Zone {worst_zone.zone_id} "
                          f"({worst_zone.lap_pct_start:.1f}%-{worst_zone.lap_pct_end:.1f}%)")
            lines.append(f"     Main issue: {top_ch.channel} "
                          f"({'too much' if top_ch.direction == 'over' else 'insufficient'}) "
                          f"— {abs(top_ch.signed_mean):.1f} {top_ch.unit} off expert level")
            if worst_zone.causal_chains:
                lines.append(f"     Root cause: {worst_zone.causal_chains[0].description}")
    lines.append("")
    lines.append("═══════════════════════════════════════════════════")
    return "\n".join(lines)
