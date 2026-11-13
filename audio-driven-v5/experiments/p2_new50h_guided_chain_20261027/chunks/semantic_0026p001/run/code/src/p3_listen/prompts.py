from .common import digest

OMNI = """The reference image shows the listener. The input audio belongs to
the other participant, who holds the conversational floor.

Generate the listener's synchronized facial video and own audio
throughout this listening interval. Model the listener's expressions,
head movements, and vocal feedback in relation to the incoming speech.
Preserve periods without vocal feedback and maintain the listener's
identity. The listener remains in the listening role throughout."""

GEMMA = """A continuous close-up video of the person in the reference image
listening to another person speaking off-camera. The camera remains
fixed, and the person's identity and background remain consistent.
Facial expressions, head movements, and brief vocal feedback, when
appropriate, form a natural listening response. The person remains
in the listening role throughout. The soundtrack contains the
listener's own vocalizations and periods without vocalization."""


def contract():
    return dict(version="fixed-listening-v1", omni=OMNI, gemma=GEMMA,
                sha256=digest(dict(omni=OMNI, gemma=GEMMA)))


class FixedPrompt:
    def __init__(self, text=GEMMA):
        self.text = text

    def render_prompt(self):
        return self.text
