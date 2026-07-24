"""Tests for the OpenAI-compatible provider -- request shape + response parsing, mocked HTTP.

No live endpoint: we stub `_post` to return OpenAI-shaped payloads and check that the two
Provider methods build the right request and parse the right answer out of it.
"""

from wobblelab import OpenAICompatibleProvider, Provider


class _Stub(OpenAICompatibleProvider):
    """Capture the last payload and return a canned response instead of hitting the network."""

    def __init__(self, response, **kw):
        super().__init__("m", **kw)
        self._response = response
        self.last_payload = None

    def _post(self, payload):
        self.last_payload = payload
        return self._response


def test_satisfies_the_provider_protocol():
    assert isinstance(OpenAICompatibleProvider("m"), Provider)


def test_ask_builds_chat_request_and_returns_content():
    p = _Stub({"choices": [{"message": {"content": "B"}}]})
    out = p.ask("pick one", seed=7)
    assert out == "B"
    assert p.last_payload["messages"] == [{"role": "user", "content": "pick one"}]
    assert p.last_payload["seed"] == 7
    assert p.last_payload["model"] == "m"


def test_rank_letters_parses_top_logprobs_and_argmaxes():
    resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "A",
                            "top_logprobs": [
                                {"token": "A", "logprob": -0.05},
                                {"token": "B", "logprob": -3.7},
                                {"token": "The", "logprob": -6.0},
                            ],
                        }
                    ]
                }
            }
        ]
    }
    p = _Stub(resp)
    pos, scores = p.rank_letters("q", n_options=4, seed=0)
    assert pos == 0  # "A" is argmax
    assert set(scores) == {"A", "B"}  # only in-range letters kept (not "The")
    # requests the log-likelihood readout, deterministically
    assert p.last_payload["logprobs"] is True and p.last_payload["max_tokens"] == 1
    assert p.last_payload["temperature"] == 0.0


def test_rank_letters_returns_none_when_no_candidate_letter_present():
    resp = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "The",
                            "top_logprobs": [{"token": "The", "logprob": -0.1}],
                        }
                    ]
                }
            }
        ]
    }
    pos, scores = _Stub(resp).rank_letters("q", n_options=4)
    assert pos is None and scores == {}


def test_api_key_becomes_a_bearer_header_and_extra_body_flows():
    # extra_body carries vLLM-only knobs into the request unchanged.
    p = _Stub(
        {"choices": [{"message": {"content": "A"}}]},
        api_key="sk-x",
        extra_body={"top_k": 20},
    )
    p.ask("hi", 0)
    assert p.last_payload["top_k"] == 20
    cfg = p.config()
    assert cfg["provider"] == "openai-compatible" and cfg["extra_body"] == {"top_k": 20}
