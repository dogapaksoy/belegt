"""belegt — check that every fact an LLM extracts is backed by the source document."""

from belegt.matchers import ExactMatcher, FuzzyMatcher, NormalizedMatcher, QuoteMatcher
from belegt.types import Claim, Reason, Span, ValueType, Verification
from belegt.verifier import Verifier, verify

__all__ = [
    "Claim", "ExactMatcher", "FuzzyMatcher", "NormalizedMatcher", "QuoteMatcher", "Reason",
    "Span", "ValueType", "Verification", "Verifier", "verify",
]  # fmt: skip
__version__ = "0.1.0"
