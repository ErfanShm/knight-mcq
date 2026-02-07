"""Minimal sanity tests that run without Neo4j or API keys."""

import unittest

from app.core.common import config
from app.core.utils.external_knowledge import (
    WikipediaLookup,
    default_external_knowledge,
)


class TestConfig(unittest.TestCase):
    """Config module exposes required env-backed attributes."""

    def test_config_has_required_attrs(self):
        self.assertTrue(hasattr(config, "OPENAI_API_KEY"))
        self.assertTrue(hasattr(config, "OPENAI_MODEL"))
        self.assertTrue(hasattr(config, "GPT_NEO4J_URI"))
        self.assertTrue(hasattr(config, "MAX_DEPTH"))


class TestExternalKnowledge(unittest.TestCase):
    """External knowledge interface and default implementation."""

    def test_default_is_wikipedia_lookup(self):
        self.assertIsNotNone(default_external_knowledge)
        self.assertIsInstance(default_external_knowledge, WikipediaLookup)

    def test_default_has_lookup_method(self):
        self.assertTrue(hasattr(default_external_knowledge, "lookup"))
        self.assertTrue(callable(getattr(default_external_knowledge, "lookup")))

    def test_wikipedia_lookup_requires_llm(self):
        lookup = WikipediaLookup()
        with self.assertRaises(ValueError):
            lookup.lookup(term="test", llm=None)
