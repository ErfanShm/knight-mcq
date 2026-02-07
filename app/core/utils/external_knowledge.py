"""
External knowledge lookup interface for KNIGHT.

External knowledge is replaceable: the default source is Wikipedia.
Implement the ExternalKnowledgeLookup protocol and pass an instance
into term description generation to use a custom source (e.g. URL, file).
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ExternalKnowledgeLookup(Protocol):
    """Protocol for looking up external knowledge for a term.

    Implement this interface to plug in a custom source (e.g. URL, file).
    Returns (summary_or_none, is_ambiguous) to match Wikipedia lookup behavior.
    """

    def lookup(
        self,
        term: str,
        context_hint: str | None = None,
        *,
        llm: Any = None,
    ) -> tuple[str | None, bool]:
        """Look up external knowledge for a term.

        Args:
            term: The term to look up.
            context_hint: Optional context (e.g. parent term) for disambiguation.
            llm: Optional LLM instance; some implementations use it for relevance checks.

        Returns:
            Tuple of (summary text or None, is_ambiguous).
        """
        ...


class WikipediaLookup:
    """Default external knowledge implementation using Wikipedia."""

    def lookup(
        self,
        term: str,
        context_hint: str | None = None,
        *,
        llm: Any = None,
        doc_content_chars_max: int = 1000,
        num_search_results: int = 5,
    ) -> tuple[str | None, bool]:
        if llm is None:
            raise ValueError("WikipediaLookup requires an LLM instance for relevance checks.")
        from app.core.utils.wikipedia_lookup import get_wikipedia_summary

        return get_wikipedia_summary(
            llm=llm,
            term=term,
            context_hint=context_hint,
            doc_content_chars_max=doc_content_chars_max,
            num_search_results=num_search_results,
        )


default_external_knowledge: ExternalKnowledgeLookup = WikipediaLookup()
