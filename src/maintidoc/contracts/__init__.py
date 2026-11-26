"""MaintiDoc data contracts.

Import from here rather than from the submodules, so that internal layout can
change without touching callers:

    from maintidoc.contracts import AnswerV1, EvidenceBundle
"""

from maintidoc.contracts.answer import (
    AnswerStatus,
    AnswerV1,
    Citation,
    ProposedAction,
    answer_json_schema,
)
from maintidoc.contracts.common import (
    Element,
    ElementRef,
    ExtractionMethod,
    Modality,
)
from maintidoc.contracts.evidence import Evidence, EvidenceBundle, RetrievalBranch
from maintidoc.contracts.principal import Principal
from maintidoc.contracts.run import PromptVersion, RunVersions
from maintidoc.contracts.tickets import ApprovalToken, TicketDraft

__all__ = [
    "AnswerStatus",
    "AnswerV1",
    "ApprovalToken",
    "Citation",
    "Element",
    "ElementRef",
    "Evidence",
    "ExtractionMethod",
    "EvidenceBundle",
    "Modality",
    "Principal",
    "ProposedAction",
    "PromptVersion",
    "RetrievalBranch",
    "RunVersions",
    "TicketDraft",
    "answer_json_schema",
]
