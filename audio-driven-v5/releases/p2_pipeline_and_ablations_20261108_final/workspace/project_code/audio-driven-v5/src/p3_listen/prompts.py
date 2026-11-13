from .common import digest

OMNI = """You model natural audiovisual responses for a listener in a conversation.

The reference image shows the listener. The input audio belongs to
the other participant, who holds the conversational floor. The
listener's response spans the same time interval as the input audio.

Use facial expressions, head movements, and brief vocal feedback
appropriate to the speaker's meaning and tone, while remaining in
the listening role.

Preserve the listener's identity. Align mouth movements with the
listener's own vocalizations. The output audio contains only the
listener's vocalizations and natural silent intervals."""

GEMMA = """A continuous close-up video of the person in the reference image
listening to another person speaking off-camera. The camera remains
fixed, and the person's identity and background remain consistent.
Facial expressions, head movements, and brief vocal feedback, when
appropriate, form a natural listening response. The person remains
in the listening role throughout. The soundtrack contains only the
listener's own vocalizations and natural silent intervals. Mouth
movements match the listener's own vocalizations."""


def observation_messages(instruction=None):
    """Put the shared listening task before media in the causal Qwen prefix."""
    content = [dict(type="image"), dict(type="audio")]
    if instruction is not None:
        content.append(dict(type="text", text=instruction))
    return [dict(role="system", content=[dict(type="text", text=OMNI)]),
            dict(role="user", content=content)]


def contract():
    value = dict(version="fixed-listening-system-v2", omni=OMNI, gemma=GEMMA,
                 omni_messages=observation_messages(), add_generation_prompt=True)
    return dict(value, sha256=digest(value))


class FixedPrompt:
    def __init__(self, text=GEMMA):
        self.text = text

    def render_prompt(self):
        return self.text
