import json
from typing import List, Dict, Optional
import hashlib

class ConversationMemory:
    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Add a message to memory. Role = 'user' | 'assistant' | 'system'."""
        self.history.append({"role": role, "content": content})
        # Trim old messages if exceeding max_turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_context(self, include_system: bool = True) -> List[Dict[str, str]]:
        """Get conversation history for building prompts."""
        if include_system:
            return self.history
        return [msg for msg in self.history if msg["role"] != "system"]

    def average_response_time(self):
        """Average response time of assistant messages"""
        times = [m.get("response_time", 0) for m in self.history if m["role"]=="assistant"]
        return sum(times)/len(times) if times else 0

    def compute_hash(self):
        """Compute a deterministic hash for the conversation"""
        h = hashlib.sha256()
        h.update(json.dumps(self.history, sort_keys=True).encode())
        return h.hexdigest()[:10]  # short hash

    def reset(self):
        """Clear conversation memory."""
        self.history = []

    def to_dict(self) -> Dict:
        """Serialize memory to dict (for saving)."""
        return {"history": self.history}

    @classmethod
    def from_dict(cls, data: Dict, max_turns: int = 20):
        """Rehydrate memory from saved dict."""
        mem = cls(max_turns=max_turns)
        mem.history = data.get("history", [])
        return mem

    def save_json(self, path: str):
        """Persist memory to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str, max_turns: int = 20):
        """Load memory from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data, max_turns=max_turns)
        except FileNotFoundError:
            return cls(max_turns=max_turns)
