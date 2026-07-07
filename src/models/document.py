from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class FAQDocument:
    id: str
    course: str
    section: str
    question: str
    answer: str


@dataclass(slots=True)
class SearchResult:
    id: str
    text: str
