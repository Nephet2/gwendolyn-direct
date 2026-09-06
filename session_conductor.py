"""Session-local scheduling and speech receipts; no controller authority lives here."""
import copy
import math
import re
import threading


class SessionConductor:
    def __init__(self, milestones, started_at, journal):
        self.lock = threading.RLock()
        self.started_at = started_at
        self.journal = journal
        self.milestones = copy.deepcopy(milestones)
        self.announcements = []
        self.sequence = 0
        for index, item in enumerate(self.milestones):
            item.update(id=index, state="pending", evidence=None)
            item.setdefault("due_seconds", 0)

    def due(self, now):
        with self.lock:
            return [copy.deepcopy(m) for m in self.milestones
                    if m["state"] == "pending" and now - self.started_at >= m["due_seconds"]]

    def restart(self, now):
        with self.lock:
            self.started_at = now
            for item in self.milestones:
                item.update(state="pending", evidence=None)
            self.journal("session_brief_schedule_started", {"milestones": copy.deepcopy(self.milestones)})

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.milestones)

    def transition(self, ident, state, evidence=None):
        with self.lock:
            item = self.milestones[ident]
            allowed = {"pending": {"attempted"}, "attempted": {"confirmed", "failed"}}
            if state not in allowed.get(item["state"], set()):
                return False
            item.update(state=state, evidence=copy.deepcopy(evidence))
            self.journal("session_brief_milestone_" + ("completed" if state == "confirmed" else state),
                         copy.deepcopy(item))
            return True

    def announce(self, text):
        with self.lock:
            self.sequence += 1
            self.announcements.append((self.sequence, text))

    def pending_text(self):
        with self.lock:
            return " ".join(text for _, text in self.announcements)

    def receipt(self, spoken):
        with self.lock:
            return max((i for i, t in self.announcements if t in spoken), default=0)

    def acknowledge(self, spoken, through):
        # Only called after uninterrupted successful playback, not enqueue or display.
        with self.lock:
            self.announcements = [(i, t) for i, t in self.announcements if i > through or t not in spoken]


def due_seconds(text, default=0):
    """Accept timestamps, natural timing phrases, and Target window starts."""
    match = re.search(r"\[(\d+):(\d{2})\]", text)
    if match:
        return int(match[1]) * 60 + int(match[2])
    match = re.search(r"\b(?:at|after)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|seconds?|secs?)\b", text, re.I)
    if match:
        return float(match[1]) * (60 if match[2].lower().startswith("m") else 1)
    match = re.search(r"\bminute\s+(\d+(?:\.\d+)?)\b", text, re.I)
    if match:
        return float(match[1]) * 60
    match = re.search(
        r"\btarget\s+window\s*:\s*(\d+(?:\.\d+)?)\s*(?:[-–—]\s*\d+(?:\.\d+)?)?\s*(minutes?|mins?|seconds?|secs?)\b",
        text, re.I,
    )
    return float(match[1]) * (60 if match[2].lower().startswith("m") else 1) if match else default


def valid_time(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def matches(expected, actual):
    """Subset comparison for a controller-owned, possibly enriched response."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and matches(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, (float, int)) and not isinstance(expected, bool):
        return isinstance(actual, (float, int)) and not isinstance(actual, bool) and math.isclose(expected, actual, abs_tol=0.001)
    return str(expected).strip().casefold() == str(actual).strip().casefold()
