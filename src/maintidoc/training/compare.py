"""Three-way comparison: no reranker, base reranker, adapted reranker.

The brief asks for exactly this comparison (§5), and asks for the result to be
reported *even if the adaptation brings no improvement*. That shape is built
into the output: `delta` can be negative, and nothing here treats a negative
delta as a failure to hide.

Each arm runs the same scenarios through the same retriever, changing only the
reranker. Anything else changing between arms would make the comparison
meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from maintidoc.evaluation.dataset import Dataset
from maintidoc.evaluation.runner import EvaluationRunner, Report
from maintidoc.retrieval.rerank import NoReranker, Reranker


class ArmResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    reranker: str
    available: bool = True
    detail: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)

    def metric(self, key: str) -> float | None:
        value = self.metrics.get(key)
        return float(value) if isinstance(value, (int, float)) else None


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset: str = ""
    corpus_version: str = ""
    arms: tuple[ArmResult, ...] = ()
    primary_metric: str = "ndcg@5"

    def arm(self, name: str) -> ArmResult | None:
        return next((a for a in self.arms if a.name == name), None)

    @property
    def delta_vs_base(self) -> float | None:
        """Adapted minus base on the primary metric. Negative is a result."""
        base, adapted = self.arm("base"), self.arm("adapted")
        if base is None or adapted is None:
            return None
        left, right = base.metric(self.primary_metric), adapted.metric(self.primary_metric)
        if left is None or right is None:
            return None
        return right - left

    @property
    def verdict(self) -> str:
        delta = self.delta_vs_base
        if delta is None:
            return "comparaison incomplete : un bras n'a pas pu etre execute"
        if abs(delta) < 1e-4:
            return f"aucun effet mesurable ({self.primary_metric} inchange)"
        direction = "amelioration" if delta > 0 else "degradation"
        return f"{direction} de {delta:+.4f} sur {self.primary_metric}"

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")


@dataclass
class ComparisonHarness:
    """Runs the same dataset once per reranker arm."""

    retriever: Any
    dataset: Dataset
    service: Any | None = None
    top_k: int = 8

    def _run(self, reranker: Reranker) -> Report:
        previous = self.retriever.reranker
        self.retriever.reranker = reranker
        try:
            runner = EvaluationRunner(
                self.retriever, answerer=None, service=self.service, top_k=self.top_k
            )
            return runner.run(self.dataset)
        finally:
            # Always restore: a comparison must not leave the system altered.
            self.retriever.reranker = previous

    def arm(self, name: str, reranker: Reranker | None) -> ArmResult:
        if reranker is None:
            return ArmResult(
                name=name,
                reranker="indisponible",
                available=False,
                detail="bras non execute : reranker absent",
            )
        report = self._run(reranker)
        search = report.summary.get("recherche", {})
        categories = report.summary.get("par_categorie", {})
        return ArmResult(
            name=name,
            reranker=getattr(reranker, "name", name),
            metrics={
                **search,
                "latence_p95_ms": report.summary.get("latence", {}).get("p95_ms"),
                "fuites_acces": report.summary.get("fuites_acces"),
                "taux_documentaire": categories.get("documentaire_texte", {}).get("taux"),
            },
        )

    def compare(
        self, *, base: Reranker | None, adapted: Reranker | None
    ) -> Comparison:
        return Comparison(
            dataset=self.dataset.name,
            corpus_version=self.dataset.corpus_version,
            arms=(
                self.arm("sans_reranker", NoReranker()),
                self.arm("base", base),
                self.arm("adapted", adapted),
            ),
        )


def format_comparison(comparison: Comparison) -> str:
    lines = [
        f"Jeu de test : {comparison.dataset} | corpus {comparison.corpus_version}",
        "",
        f"{'bras':<16}{'reranker':<34}{'ndcg@5':>9}{'recall@5':>10}{'mrr@10':>9}",
    ]
    for arm in comparison.arms:
        if not arm.available:
            lines.append(f"{arm.name:<16}{arm.detail}")
            continue
        lines.append(
            f"{arm.name:<16}{arm.reranker[:33]:<34}"
            f"{arm.metrics.get('ndcg@5', float('nan')):>9.4f}"
            f"{arm.metrics.get('recall@5', float('nan')):>10.4f}"
            f"{arm.metrics.get('mrr@10', float('nan')):>9.4f}"
        )
    lines += ["", f"Verdict : {comparison.verdict}"]
    return "\n".join(lines)
