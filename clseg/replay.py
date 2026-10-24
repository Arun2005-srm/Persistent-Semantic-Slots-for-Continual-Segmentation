"""Replay stores only previously observed transformed tensors and masked labels."""
import io
import random
import torch


class ReplayBuffer:
    def __init__(self, budget_bytes, ignore, seed):
        self.budget, self.ignore = budget_bytes, ignore
        self.entries = {}
        self.rng = random.Random(seed)

    @property
    def bytes_used(self):
        if not self.entries:
            return 0
        stream = io.BytesIO()
        torch.save(self.entries, stream)
        return len(stream.getvalue())

    def add(self, sample):
        classes = sorted(set(sample["mask"].unique().tolist()) - {self.ignore})
        if not classes or sample["id"] in self.entries:
            return
        stream = io.BytesIO()
        torch.save({"image": sample["image"].cpu(), "mask": sample["mask"].cpu(), "id": sample["id"]}, stream)
        payload = stream.getvalue()
        if len(payload) > self.budget:
            return
        self.entries[sample["id"]] = {"payload": payload, "classes": classes}
        while self.bytes_used > self.budget:
            # Evict an example whose labels are best represented in the buffer.
            frequency = {}
            for entry in self.entries.values():
                for c in entry["classes"]:
                    frequency[c] = frequency.get(c, 0) + 1
            score = {k: min(frequency[c] for c in v["classes"]) for k, v in self.entries.items()}
            largest = max(score.values())
            key = self.rng.choice([k for k, v in score.items() if v == largest])
            del self.entries[key]

    def sample(self, count):
        by_class = {}
        for key, entry in self.entries.items():
            for c in entry["classes"]:
                by_class.setdefault(c, []).append(key)
        if not by_class:
            return []
        result = []
        for _ in range(count):
            c = self.rng.choice(sorted(by_class))
            key = self.rng.choice(by_class[c])
            result.append(torch.load(io.BytesIO(self.entries[key]["payload"]), weights_only=True))
        return result

    def state_dict(self):
        return {"entries": self.entries, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.entries = state["entries"]
        self.rng.setstate(state["rng"])
