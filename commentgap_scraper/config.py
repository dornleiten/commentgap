from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ScrapeConfig:
    year: int
    output_dir: Path
    user_agent: str
    contact: str
    request_interval: float = 1.0
    timeout: float = 30.0
    max_retries: int = 5
    reply_query_depth: int = 32
    progress_every_pages: int = 10

    def __post_init__(self) -> None:
        if not 1997 <= self.year <= 2100:
            raise ValueError("year must be between 1997 and 2100")
        if self.request_interval < 1.0:
            raise ValueError("request_interval must be at least one second")
        if not self.user_agent.strip():
            raise ValueError("an operator-identifying user agent is required")
        if not self.contact.strip() or "@" not in self.contact:
            raise ValueError("a contact email is required")
        if self.reply_query_depth < 1:
            raise ValueError("reply_query_depth must be positive")
        if self.progress_every_pages < 1:
            raise ValueError("progress_every_pages must be positive")

    @property
    def identifying_user_agent(self) -> str:
        return f"{self.user_agent.strip()} (contact: {self.contact.strip()})"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "crawl_manifest.sqlite3"
