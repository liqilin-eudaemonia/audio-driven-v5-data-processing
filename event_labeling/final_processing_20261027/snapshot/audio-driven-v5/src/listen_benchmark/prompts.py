"""Versioned experimental prompts; production labeling prompts stay unchanged."""
from listen_labeling.client import prompt


VARIANTS = {"baseline", "backchannel_occurrences_v1", "backchannel_simple_v2", "backchannel_visual_simple_v2", "backchannel_audio_simple_v2"}


def annotation_prompt(system, duration, stage):
    variant = system.get("prompt_variant", "baseline")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown experimental prompt variant: {variant}")
    if variant == "baseline":
        return prompt(system["branch"], duration, stage)
    if variant == "backchannel_audio_simple_v2":
        if stage != "discovery" or system["branch"] != "audio":
            raise ValueError("Audio timeline prompt requires audio discovery")
        return f'''Listen to this {duration:.9g}-second listener clip. List all audible backchannel behaviors and sounds you recognize. Give each occurrence its own start time, end time (seconds from clip start), and a brief description.
Return JSON with an events array. Each event has start, end, modality ("audio"), and behavior (description).'''
    if variant == "backchannel_visual_simple_v2":
        if stage != "discovery" or system["branch"] != "visual":
            raise ValueError("Visual timeline prompt requires visual discovery")
        return f'''Watch this {duration:.9g}-second listener clip. List all visible backchannel behaviors you recognize. Give each occurrence its own start time, end time (seconds from clip start), and a brief description.
Return JSON with an events array. Each event has start, end, modality ("visual"), and behavior (description).'''
    if stage != "discovery" or system["branch"] != "omni":
        raise ValueError("Occurrence prompt requires audiovisual discovery")
    if variant == "backchannel_simple_v2":
        return f'''Watch and listen to this {duration:.9g}-second listener clip. List all backchannel behaviors and sounds you recognize. Give each occurrence its own start time, end time (seconds from clip start), and a brief description.
Return JSON with an events array. Each event has start, end, modality ("visual" or "audio"), and behavior (description).'''
    return f'''Task: occurrence_level_backchannel_discovery
Inspect the entire {duration:.9g}-second listener clip with synchronized video and independent listener audio. Produce an event timeline, not a caption or whole-clip summary. Find every directly observable occurrence of possible listening feedback, including brief and subtle occurrences. Do not invent an event to satisfy this instruction; an empty result is allowed.

PRIORITY: individual occurrences and their own onset/offset.
1. Examine visible behavior and audible behavior separately, then consider their temporal correspondence. Examples to look for include nods, head shakes, smiles, laughter, facial reactions, gestures and short utterances such as "mm", "uh-huh", "yeah", "okay", "ok" and "haha". These examples are not an exhaustive taxonomy and are not evidence that any of them occur in this clip. Keep unfamiliar observations in free text. Ordinary posture changes and blinks are not automatically communicative feedback; describe only what is supported and keep interpretation uncertain.
2. Output one event for EACH distinguishable occurrence. If a word such as "okay" occurs twice, output two audio events with independently estimated start and end times, not one event saying "says okay twice". Also split repeated intelligible feedback words when their individual boundaries are distinguishable. A continuous laugh is one bout, not an invented event for each "ha". Split bouts when actual pauses separate them.
3. A nod is a visible temporal head movement, not a still downward pose. When individual down-and-return or up-and-return cycles are distinguishable, give each cycle its own interval and motion description. Do not convert sideways turns, leaning or body translation into nods. If a continuous repeated sequence cannot be reliably split, report the observed sequence with uncertainty about cycle boundaries; do not manufacture equally spaced cycles.
4. Give each visual event its actual visible onset and offset. For a sustained smile, include the supported duration of the smile, not just the instant of onset. Split a new occurrence after a real cessation. Do not divide one continuous state into repeated 0.5-second or 1-second descriptions. Separate distinct actions (for example a hand moving up and a hand covering the mouth) when their changes can be distinguished, without adding redundant copies of the same action.
5. For audio, use the actual audible beginning and ending of the utterance or vocalization. Preserve the original spoken word in behavior when intelligible; otherwise describe the sound and explicitly state uncertainty. Do not guess a word from lip motion. Do not infer audible laughter merely from a visible open mouth or smile.
6. Keep visual and audio events separate even when they co-occur: a nod plus "yeah" requires a visual event and an audio event, potentially with different intervals. Joint evidence may help identify correspondence or ownership, but must not cause a sound's interval to be copied onto an entire visible gesture or vice versa.
7. Times are local seconds from clip start. Do not use [0, clip duration] merely because you observed an event somewhere in the clip. Use a whole-clip event only if that specific behavior is continuously supported throughout it. Locate boundaries from the media, not by evenly dividing the clip or distributing events uniformly. Video is sampled at 8 fps (approximately 0.125-second spacing); additional decimals do not establish greater accuracy. When timing or occurrence identity is ambiguous, explicitly state that uncertainty. Never fabricate precise timestamps.
8. Before emitting the JSON, recheck the timeline for omitted head movements, omitted short sounds, repeated occurrences incorrectly merged, duplicate descriptions of one continuous event, and overlong intervals. This is an instruction to re-examine the supplied evidence, not permission to add unsupported positives. Keep notes short; any event reported in notes must also have its own supported entry in events. Sort events by onset, retaining overlaps.

OUTPUT CONTRACT (same fields as the baseline):
''' + prompt("omni", duration, "discovery")
