"""Training-only cache selection; frozen feature extractors/cache identities stay unchanged."""
from . import features
from .annotations import validate, validate_contract
from .config import feature_loss_enabled
from .contracts import path_at, read


def load_training_sample(c, obs, target, device="cpu"):
    if feature_loss_enabled(c):
        return features.load_training_sample(c, obs, target, device)
    # Never resolve, stat, hash or deserialize teacher data in AV-only mode.
    x, layout = features.load_input(c, obs, device)
    av = features.load_part(features.folder(c, "vae", obs["sample_id"]),
                            features.identity(c, "vae", obs, target), device)
    label = read(path_at(c["output"]) / "labels/canonical" / (obs["sample_id"] + ".json"))
    validate(label, obs)
    validate_contract(label, c)
    events = features.load_part(features.folder(c, "events", obs["sample_id"]),
                                features.identity(c, "events", obs, label=label), device)
    return x, layout, None, av, events
