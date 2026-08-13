import os
import unittest
from unittest.mock import MagicMock, patch

from app.config import settings
from app.core.llm import openai_cache_kwargs
from app.core.prompts import (
    GENERATE_RETRY_HINT,
    SYSTEM_PROMPT,
    USER_CONTEXT_HEADER,
    USER_QUESTION_HEADER,
    build_messages,
    build_prompt,
    build_rewrite_messages,
    build_user_prompt,
)
from app.eval.tracing import (
    configure_tracing,
    graph_run_config,
    traced_invoke,
    tracing_enabled,
)


class PromptCacheLayoutTests(unittest.TestCase):
    def test_system_prefix_is_identical_across_calls(self):
        first = build_messages(
            "What is RAG?",
            [{"text": "Retrieval then generation.", "page": 1, "document": "a.pdf"}],
        )
        second = build_messages(
            "What granularity does DenseX use?",
            [{"text": "DenseX uses propositions.", "page": 6, "document": "b.pdf"}],
        )
        self.assertEqual(first[0]["role"], "system")
        self.assertEqual(first[0]["content"], SYSTEM_PROMPT)
        self.assertEqual(first[0]["content"], second[0]["content"])
        self.assertIs(first[0]["content"], SYSTEM_PROMPT)

    def test_frozen_prefix_meets_openai_cache_minimum(self):
        try:
            import tiktoken
        except ImportError:
            self.skipTest("tiktoken is not installed")
        encoder = tiktoken.get_encoding("o200k_base")
        self.assertGreaterEqual(len(encoder.encode(SYSTEM_PROMPT)), 1024)

    def test_context_and_question_are_in_the_user_message(self):
        messages = build_messages(
            "What is RAG?",
            [
                {
                    "text": "RAG retrieves then generates.",
                    "page": 3,
                    "document": "rag.pdf",
                    "section": "Introduction",
                    "content_type": "text",
                }
            ],
        )
        user = messages[1]["content"]
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn(USER_CONTEXT_HEADER, user)
        self.assertIn("RAG retrieves then generates.", user)
        self.assertIn(f"{USER_QUESTION_HEADER} What is RAG?", user)
        self.assertLess(user.find("Context:"), user.find("Question:"))
        self.assertGreater(user.rfind("Question:"), user.find("Context:"))
        self.assertTrue(user.strip().endswith("Answer:"))
        self.assertNotIn(SYSTEM_PROMPT, user)

    def test_retry_hint_stays_out_of_the_frozen_prefix(self):
        messages = build_messages("What is RAG?", [], retry=True)
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertNotIn(GENERATE_RETRY_HINT, messages[0]["content"])
        self.assertIn(GENERATE_RETRY_HINT, messages[1]["content"])
        self.assertGreater(
            messages[1]["content"].find(GENERATE_RETRY_HINT),
            messages[1]["content"].find("Question:"),
        )
        self.assertTrue(messages[1]["content"].strip().endswith("Answer:"))

    def test_string_prompt_keeps_system_before_question(self):
        prompt = build_prompt(
            "What is RAG?",
            [{"text": "passage", "page": 2, "document": "rag.pdf"}],
        )
        self.assertTrue(prompt.startswith(SYSTEM_PROMPT))
        self.assertGreater(prompt.find("Question: What is RAG?"), prompt.find(SYSTEM_PROMPT))
        self.assertGreater(prompt.find("Question:"), prompt.find("Context:"))

    def test_rewrite_messages_keep_static_instructions_first(self):
        messages = build_rewrite_messages(
            "What granularity?",
            "What granularity?",
            [{"text": "unrelated footnotes"}],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Original question:", messages[1]["content"])
        self.assertNotIn("Original question:", messages[0]["content"])

    def test_user_prompt_helper_matches_message_body(self):
        chunks = [{"text": "x", "page": 1, "document": "d.pdf"}]
        self.assertEqual(
            build_user_prompt("Q?", chunks),
            build_messages("Q?", chunks)[1]["content"],
        )


class OpenAICacheFlagTests(unittest.TestCase):
    def test_cache_kwargs_include_key_and_retention(self):
        kwargs = openai_cache_kwargs()
        self.assertEqual(kwargs["prompt_cache_key"], settings.prompt_cache_key)
        self.assertEqual(kwargs["prompt_cache_retention"], settings.prompt_cache_retention)
        self.assertTrue(kwargs["prompt_cache_key"])
        self.assertTrue(kwargs["prompt_cache_retention"])

    def test_cache_kwargs_empty_when_disabled(self):
        self.assertEqual(openai_cache_kwargs(use_prompt_cache=False), {})


class TracingConfigTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: os.environ.get(key)
            for key in (
                "LANGSMITH_TRACING",
                "LANGCHAIN_TRACING_V2",
                "LANGSMITH_API_KEY",
                "LANGCHAIN_API_KEY",
                "LANGSMITH_PROJECT",
                "LANGCHAIN_PROJECT",
                "LANGSMITH_ENDPOINT",
                "LANGCHAIN_ENDPOINT",
            )
        }

    def tearDown(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_configure_tracing_sets_langsmith_and_legacy_env(self):
        with (
            patch("app.eval.tracing.settings") as mock_settings,
        ):
            mock_settings.langsmith_api_key = "ls-test-key"
            mock_settings.langsmith_project = "rag-pipeline"
            mock_settings.langsmith_tracing = True
            mock_settings.langsmith_endpoint = ""
            configure_tracing()

        self.assertEqual(os.environ["LANGSMITH_TRACING"], "true")
        self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "true")
        self.assertEqual(os.environ["LANGSMITH_API_KEY"], "ls-test-key")
        self.assertEqual(os.environ["LANGCHAIN_API_KEY"], "ls-test-key")
        self.assertEqual(os.environ["LANGSMITH_PROJECT"], "rag-pipeline")
        self.assertEqual(os.environ["LANGCHAIN_PROJECT"], "rag-pipeline")

    def test_tracing_disabled_without_api_key(self):
        with patch("app.eval.tracing.settings") as mock_settings:
            mock_settings.langsmith_api_key = ""
            mock_settings.langsmith_tracing = True
            with patch.dict(os.environ, {"LANGSMITH_API_KEY": ""}, clear=False):
                os.environ.pop("LANGSMITH_API_KEY", None)
                self.assertFalse(tracing_enabled())

    def test_graph_run_config_names_the_ask_trace(self):
        config = graph_run_config({"question": "What is RAG?", "filters": {"page": 1}})
        self.assertEqual(config["run_name"], "rag_ask")
        self.assertIn("rag", config["tags"])
        self.assertTrue(config["metadata"]["has_filters"])
        self.assertEqual(config["metadata"]["pipeline"], "advanced")

    def test_graph_run_config_tags_baseline_eval(self):
        config = graph_run_config({"question": "What is RAG?"}, pipeline="baseline")
        self.assertEqual(config["run_name"], "rag_ask_baseline")
        self.assertIn("baseline", config["tags"])
        self.assertEqual(config["metadata"]["pipeline"], "baseline")

    def test_traced_invoke_passes_run_config(self):
        graph = MagicMock()
        graph.invoke.return_value = {"answer": "RAG retrieves then generates."}
        payload = {"question": "What is RAG?"}
        with patch("app.eval.tracing.tracing_enabled", return_value=False):
            result = traced_invoke(graph, payload)
        self.assertEqual(result["answer"], "RAG retrieves then generates.")
        args, kwargs = graph.invoke.call_args
        self.assertEqual(args[0], payload)
        self.assertEqual(kwargs["config"]["run_name"], "rag_ask")


class GenerateCacheCallTests(unittest.TestCase):
    @patch("app.core.llm._openai_client")
    @patch("app.core.llm.settings")
    def test_generate_sends_frozen_prefix_and_cache_flags(self, mock_settings, mock_client):
        mock_settings.openai_api_key = "sk-test"
        mock_settings.openai_model = "gpt-4.1-mini"
        mock_settings.prompt_cache_key = "rag-pipeline-v1"
        mock_settings.prompt_cache_retention = "in_memory"
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Grounded answer (page 1)."
        mock_client.return_value.chat.completions.create.return_value = mock_response

        from app.core.llm import generate

        messages = build_messages(
            "What is RAG?",
            [{"text": "RAG retrieves then generates.", "page": 1, "document": "rag.pdf"}],
        )
        text = generate(messages=messages)
        self.assertEqual(text, "Grounded answer (page 1).")
        kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-4.1-mini")
        self.assertEqual(kwargs["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(kwargs["messages"][1]["role"], "user")
        self.assertIn("Question: What is RAG?", kwargs["messages"][1]["content"])
        self.assertEqual(kwargs["prompt_cache_key"], "rag-pipeline-v1")
        self.assertEqual(kwargs["prompt_cache_retention"], "in_memory")


if __name__ == "__main__":
    unittest.main()
