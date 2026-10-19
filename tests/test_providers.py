"""Model providers against local fakes of each API (no keys, no network): what each SDK sends and how replies,
refusals and errors are handled. Skipped when the provider's SDK isn't installed."""
import importlib.util
import json
import os
import unittest
from unittest import mock

from agent import providers
from agent.config import ConfigError
from agent.decision import SCHEMA, validate
from agent.providers import ProviderError
from tests.fakes import FakeServer

DECISION = {"action": "buy", "ticker": "NVDA", "time_horizon": "1W", "confidence": 7, "stop_loss_pct": 6,
            "take_profit_pct": 12, "reasoning": "Three kinds of sources agree and it is up 4% since."}
CANDIDATES = {"NVDA": {"price": 100.0}}


def installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def claude_message(text=json.dumps(DECISION), stop_reason="end_turn"):
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 80}}


def chat_completion(content=json.dumps(DECISION), finish_reason="stop"):
    return {"id": "chatcmpl-test", "object": "chat.completion", "created": 1760000000, "model": "test-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content, "refusal": None},
                         "finish_reason": finish_reason, "logprobs": None}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280}}


def gemini_reply(text=json.dumps(DECISION)):
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP",
                            "index": 0}],
            "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 80, "totalTokenCount": 1280},
            "modelVersion": "gemini-3.8-flash"}


class ProviderTest(unittest.TestCase):
    def serve(self, routes):
        server = FakeServer(routes)
        self.addCleanup(server.close)
        return server

    def decide(self, provider):
        reply = provider.decide("system prompt", "user prompt", {}, CANDIDATES)
        return validate(reply, candidates=CANDIDATES, held=set())


@unittest.skipUnless(installed("anthropic"), "anthropic isn't installed")
class AnthropicProvider(ProviderTest):
    def test_request_shape(self):
        server = self.serve({("POST", "/v1/messages"): (200, claude_message())})
        decision = self.decide(providers.Anthropic("claude-opus-5-5", "sk-ant-test", base_url=server.url))
        self.assertEqual((decision.action, decision.ticker, decision.confidence), ("buy", "NVDA", 7))
        request = server.requests[0]
        self.assertEqual(request["headers"]["x-api-key"], "sk-ant-test")
        self.assertIn("server-side-fallback-2026-07-01", request["headers"]["anthropic-beta"])
        body = request["body"]
        self.assertEqual(body["model"], "claude-opus-5-5")
        self.assertEqual(body["fallbacks"], "default")
        self.assertEqual(body["output_config"], {"format": {"type": "json_schema", "schema": SCHEMA},
                                                 "effort": "medium"})
        self.assertEqual(body["system"], "system prompt")
        self.assertEqual(body["messages"], [{"role": "user", "content": "user prompt"}])
        self.assertNotIn("thinking", body)  # Opus 5.5 always thinks; effort is the control

    def test_other_models_skip_effort_and_fallbacks(self):
        server = self.serve({("POST", "/v1/messages"): (200, claude_message())})
        self.decide(providers.Anthropic("claude-haiku-5-5", "sk-ant-test", base_url=server.url))
        body = server.requests[0]["body"]
        self.assertNotIn("fallbacks", body)
        self.assertNotIn("effort", body["output_config"])

    def test_retries_plainly_when_the_model_rejects_the_extras(self):
        server = self.serve({("POST", "/v1/messages"): [
            (400, {"type": "error", "error": {"type": "invalid_request_error",
                                              "message": "output_config.format is not supported"}}),
            (200, claude_message("```json\n" + json.dumps(DECISION) + "\n```"))]})
        decision = self.decide(providers.Anthropic("claude-sonnet-4-6", "sk-ant-test", base_url=server.url))
        self.assertEqual(decision.ticker, "NVDA")
        self.assertEqual(len(server.requests), 2)
        self.assertNotIn("output_config", server.requests[1]["body"])

    def test_refusal_is_not_a_decision(self):
        server = self.serve({("POST", "/v1/messages"): (200, claude_message("", stop_reason="refusal"))})
        with self.assertRaises(ProviderError) as caught:
            providers.Anthropic("claude-opus-5-5", "sk-ant-test", base_url=server.url).decide("s", "u", {}, {})
        self.assertFalse(caught.exception.fatal)

    def test_bad_key_is_fatal(self):
        server = self.serve({("POST", "/v1/messages"): (401, {"type": "error", "error": {
            "type": "authentication_error", "message": "invalid x-api-key"}})})
        with self.assertRaises(ProviderError) as caught:
            providers.Anthropic("claude-opus-5-5", "sk-ant-bad", base_url=server.url).decide("s", "u", {}, {})
        self.assertTrue(caught.exception.fatal)
        self.assertNotIn("sk-ant-bad", str(caught.exception))


@unittest.skipUnless(installed("openai"), "openai isn't installed")
class OpenAIProvider(ProviderTest):
    def test_openai_uses_strict_json_schema(self):
        server = self.serve({("POST", "/v1/chat/completions"): (200, chat_completion())})
        provider = providers.OpenAI("gpt-6.1-sol", "sk-test", base_url=server.url + "/v1")
        provider.compatible = False  # behave like api.openai.com, at the fake's address
        self.assertEqual(self.decide(provider).ticker, "NVDA")
        request = server.requests[0]
        self.assertEqual(request["headers"]["authorization"], "Bearer sk-test")
        self.assertEqual(request["body"]["response_format"], {
            "type": "json_schema", "json_schema": {"name": "trade_decision", "strict": True, "schema": SCHEMA}})
        self.assertEqual([m["role"] for m in request["body"]["messages"]], ["system", "user"])

    def test_compatible_server_falls_back_to_plain_text(self):
        server = self.serve({("POST", "/v1/chat/completions"): [
            (400, {"error": {"message": "response_format is not supported", "type": "invalid_request_error"}}),
            (200, chat_completion("Sure! " + json.dumps(DECISION)))]})
        provider = providers.OpenAI("llama3.2", None, base_url=server.url + "/v1")
        self.assertEqual(provider.name, "openai-compatible")
        self.assertEqual(self.decide(provider).ticker, "NVDA")
        self.assertEqual(server.requests[0]["body"]["response_format"], {"type": "json_object"})
        self.assertNotIn("response_format", server.requests[1]["body"])

    def test_unknown_model_is_fatal(self):
        server = self.serve({("POST", "/v1/chat/completions"): (404, {"error": {
            "message": "The model `nope` does not exist", "type": "invalid_request_error",
            "code": "model_not_found"}})})
        with self.assertRaises(ProviderError) as caught:
            providers.OpenAI("nope", None, base_url=server.url + "/v1").decide("s", "u", {}, {})
        self.assertTrue(caught.exception.fatal)
        self.assertIn("LLM_MODEL", str(caught.exception))

    def test_cut_off_answer(self):
        server = self.serve({("POST", "/v1/chat/completions"): (200, chat_completion('{"action": "bu', "length"))})
        with self.assertRaisesRegex(ProviderError, "cut off"):
            providers.OpenAI("llama3.2", None, base_url=server.url + "/v1").decide("s", "u", {}, {})


@unittest.skipUnless(installed("google.genai"), "google-genai isn't installed")
class GeminiProvider(ProviderTest):
    PATH = "/v1beta/models/gemini-3.8-flash:generateContent"

    def setUp(self):
        patcher = mock.patch.object(providers.Gemini, "BUSY_WAITS", (0, 0))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_request_shape_and_busy_retry(self):
        server = self.serve({("POST", self.PATH): [
            (503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}),
            (200, gemini_reply())]})
        decision = self.decide(providers.Gemini("gemini-3.8-flash", "AIza-test-key", base_url=server.url))
        self.assertEqual(decision.ticker, "NVDA")
        self.assertEqual(len(server.requests), 2)
        request = server.requests[1]
        self.assertEqual(request["headers"]["x-goog-api-key"], "AIza-test-key")
        self.assertEqual(request["body"]["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(request["body"]["systemInstruction"]["parts"][0]["text"], "system prompt")

    def test_bad_key_is_fatal(self):
        server = self.serve({("POST", self.PATH): (400, {"error": {
            "code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}})})
        with self.assertRaises(ProviderError) as caught:
            providers.Gemini("gemini-3.8-flash", "AIza-bad", base_url=server.url).decide("s", "u", {}, {})
        self.assertTrue(caught.exception.fatal)

    def test_gives_up_after_the_busy_retries(self):
        busy = (503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
        server = self.serve({("POST", self.PATH): busy})
        with self.assertRaises(ProviderError) as caught:
            providers.Gemini("gemini-3.8-flash", "AIza-test", base_url=server.url).decide("s", "u", {}, {})
        self.assertFalse(caught.exception.fatal)
        self.assertEqual(len(server.requests), 3)


class Choose(unittest.TestCase):
    def choose(self, **environment):
        with mock.patch.dict(os.environ, environment, clear=True):
            return providers.choose()

    def test_no_key_means_rules(self):
        provider = self.choose()
        self.assertEqual((provider.name, provider.label), ("rules", "rules-based (no LLM)"))

    def test_first_key_wins_with_its_default_model(self):
        self.assertEqual(self.choose(ANTHROPIC_API_KEY="a", GEMINI_API_KEY="g").model, "claude-opus-5-5")
        self.assertEqual(self.choose(OPENAI_API_KEY="o").model, "gpt-6.1-sol")
        self.assertEqual(self.choose(GEMINI_API_KEY="g").model, "gemini-3.8-flash")
        self.assertEqual(self.choose(GOOGLE_API_KEY="g").name, "gemini")

    def test_blank_secrets_count_as_unset(self):
        self.assertEqual(self.choose(ANTHROPIC_API_KEY="", OPENAI_API_KEY="  ").name, "rules")

    def test_base_url_means_a_compatible_server(self):
        provider = self.choose(OPENAI_BASE_URL="http://localhost:11434/v1", LLM_MODEL="llama3.2")
        self.assertEqual((provider.name, provider.model, provider.label), ("openai-compatible", "llama3.2", "llama3.2"))
        with self.assertRaisesRegex(ConfigError, "LLM_MODEL"):
            self.choose(OPENAI_BASE_URL="http://localhost:11434/v1")

    def test_ollama_alias_defaults_to_the_local_server(self):
        provider = self.choose(LLM_PROVIDER="ollama", LLM_MODEL="qwen3:8b")
        self.assertEqual(provider.base_url, "http://localhost:11434/v1")

    def test_forced_provider_needs_its_key(self):
        with self.assertRaisesRegex(ConfigError, "ANTHROPIC_API_KEY"):
            self.choose(LLM_PROVIDER="anthropic", GEMINI_API_KEY="g")
        with self.assertRaisesRegex(ConfigError, "LLM_PROVIDER must be"):
            self.choose(LLM_PROVIDER="skynet")
        self.assertEqual(self.choose(LLM_PROVIDER="rules", ANTHROPIC_API_KEY="a").name, "rules")

    def test_model_override_and_profile_label(self):
        provider = self.choose(ANTHROPIC_API_KEY="a", LLM_MODEL="claude-sonnet-5-5")
        self.assertEqual(provider.model, "claude-sonnet-5-5")
        weird = providers.OpenAI("hf.co/org/model:Q4_K_M#1", None, base_url="http://x/v1")
        self.assertEqual(weird.label, "hf.co/org/model:Q4_K_M-1")


if __name__ == "__main__":
    unittest.main()
