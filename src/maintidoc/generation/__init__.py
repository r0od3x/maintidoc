"""Phase 5 — generation sourcee et verification des citations.

    from maintidoc.generation import Answerer, LexicalGroundingChecker
"""

from maintidoc.generation.answering import (
    Answerer,
    AnsweringResult,
    MalformedAnswer,
    parse_answer,
)
from maintidoc.generation.client import (
    GenerationError,
    GenerationRequest,
    GenerationResponse,
    OllamaVlmClient,
    StubVlmClient,
    VlmClient,
    build_vlm_client,
)
from maintidoc.generation.grounding import (
    GroundingCheck,
    GroundingChecker,
    GroundingReport,
    GroundingVerdict,
    LexicalGroundingChecker,
    LlmGroundingChecker,
    check_grounding,
    downgrade_if_unsupported,
)
from maintidoc.generation.prompting import (
    PreparedPrompt,
    format_evidence,
    prepare_answer_prompt,
)

__all__ = [
    "Answerer",
    "AnsweringResult",
    "GenerationError",
    "GenerationRequest",
    "GenerationResponse",
    "GroundingCheck",
    "GroundingChecker",
    "GroundingReport",
    "GroundingVerdict",
    "LexicalGroundingChecker",
    "LlmGroundingChecker",
    "MalformedAnswer",
    "OllamaVlmClient",
    "PreparedPrompt",
    "StubVlmClient",
    "VlmClient",
    "build_vlm_client",
    "check_grounding",
    "downgrade_if_unsupported",
    "format_evidence",
    "parse_answer",
    "prepare_answer_prompt",
]
