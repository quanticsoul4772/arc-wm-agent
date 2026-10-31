"""One classifier and one gate-profile source (Plan C, M3).

The scope-alignment rule from the earlier harnesses: the metric, the prompt and the gate read the same
classifier. `classify` sorts a game by the labels it accepts; `GateProfile` holds every number the gate
enforces; the gate judges with the profile of the current class, the forced prompt addendum is rendered from
a profile, and every gate log record carries the class.

Enforced values are today's constants for every class (`CLASS_OVERRIDES` is empty per class): tuning them on
the six probe games and then validating on the same games would be circular. `SHADOW_OVERRIDES` are
alternatives that are only logged: at each refusal or batch cap the gate records whether each alternative
would also have refused, so per-class values can be chosen later from data.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

GAME_CLASSES = ("keys", "click", "mixed")
CLICK_LABELS = frozenset({"MOUSE", "ACTION6"})
IGNORED_LABELS = frozenset({"RESET"})


def classify(valid_actions: list[str]) -> str | None:
    """"click" (clicks only), "keys" (no clicks; ACTION7 and unmapped labels count as keys) or "mixed".
    None when no playable label is listed (a terminal state), so the caller keeps the class it had."""
    labels = {str(a).strip().upper() for a in valid_actions} - IGNORED_LABELS
    if not labels:
        return None
    clicks = bool(labels & CLICK_LABELS)
    keys = bool(labels - CLICK_LABELS)
    if clicks and keys:
        return "mixed"
    return "click" if clicks else "keys"


@dataclass(frozen=True)
class GateProfile:
    probe_actions: int
    change_floor: int
    min_informative: int
    pass_accuracy: float
    blocked_call_release: int
    forced: bool = True


CLASS_OVERRIDES: dict[str, dict] = {"keys": {}, "click": {}, "mixed": {}}
SHADOW_OVERRIDES: dict[str, dict] = {"optional": {"forced": False}, "probe16": {"probe_actions": 16}}


def profile_for(game_class: str | None, base: GateProfile) -> GateProfile:
    """The enforced profile for a class; an unknown class (before the first labels) gets the base profile."""
    return replace(base, **CLASS_OVERRIDES.get(game_class or "", {}))


def shadow_profiles(profile: GateProfile) -> dict[str, GateProfile]:
    return {name: replace(profile, **overrides) for name, overrides in SHADOW_OVERRIDES.items()}
