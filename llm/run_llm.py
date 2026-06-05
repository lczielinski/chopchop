import os
import time
from dataclasses import dataclass, field
from collections import Counter

import requests


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a `.env` file (searching the cwd and its
    parents) into os.environ, without overriding existing variables. Lets a
    project-local `.env` work without depending on `uv run --env-file`."""
    directory = os.path.abspath(os.getcwd())
    while True:
        path = os.path.join(directory, ".env")
        if os.path.isfile(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    if line.startswith("export "):
                        line = line[len("export ") :]
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
            return
        parent = os.path.dirname(directory)
        if parent == directory:  # reached filesystem root
            return
        directory = parent


def _default_api_key() -> str:
    if "OPENROUTER_API_KEY" not in os.environ:
        _load_dotenv()
    try:
        return os.environ["OPENROUTER_API_KEY"]
    except KeyError:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to a .env file in the project "
            "root, or export it:\n    export OPENROUTER_API_KEY=sk-or-..."
        )


@dataclass
class ModelConfig:
    model_id: str = "anthropic/claude-opus-4.8"
    api_key: str = field(default_factory=_default_api_key)
    base_url: str = OPENROUTER_BASE_URL


@dataclass
class Config:
    """
    Configuration for a single whole-program generation.
    """

    temperature: float = 0.5
    top_p: float = 1.0
    max_new_tokens: int = 512  # cap on tokens per generation (API max_tokens)
    timeout: int = 120  # per-request timeout in seconds
    verbose: bool = False


@dataclass
class RunInfo:
    llm_finished: bool
    output: str
    total_realizability_time: float = 0.0
    num_tokens_guessed: int = 0
    num_tokens_generated: int = 0
    tries_per_token: Counter = field(default_factory=Counter)
    timed_out: bool = False


class LanguageModelRunner:
    """
    Generates whole programs by calling an OpenRouter chat-completions endpoint.

    Unlike the previous local-model runner, this performs no token-by-token
    constrained decoding (a hosted API exposes no shared KV cache or per-token
    banning). The caller is expected to do rejection sampling: generate a whole
    program, then accept/reject it with a realizability check.
    """

    def __init__(self, model_config: ModelConfig | None = None):
        self.model_config = model_config or ModelConfig()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {model_config.api_key}",
                "Content-Type": "application/json",
            }
        )

    def run(
        self,
        config: Config,
        prompt: str,
        context: str,
        fixed_prefix: str = "",
    ) -> RunInfo:
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]
        if fixed_prefix:
            # Assistant prefill: continue from the given prefix (supported by
            # Anthropic models through OpenRouter).
            messages.append({"role": "assistant", "content": fixed_prefix})

        payload = {
            "model": self.model_config.model_id,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_new_tokens,
        }

        url = f"{self.model_config.base_url}/chat/completions"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.session.post(url, json=payload, timeout=config.timeout)
            except requests.Timeout:
                if config.verbose:
                    print("  [request timed out]", flush=True)
                return RunInfo(llm_finished=False, output="", timed_out=True)
            except requests.RequestException as e:
                last_error = e
                break

            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = requests.HTTPError(
                    f"{resp.status_code}: {resp.text[:200]}"
                )
                # back off briefly and retry transient errors
                if attempt < 2:
                    time.sleep(2**attempt)
                    continue
                break

            if resp.status_code != 200:
                # Non-retryable client error (bad request, auth, etc.) — surface it.
                raise RuntimeError(
                    f"OpenRouter returned {resp.status_code}: {resp.text[:500]}"
                )

            data = resp.json()
            choice = data["choices"][0]
            output = choice["message"]["content"] or ""
            if fixed_prefix:
                output = fixed_prefix + output
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage") or {}
            return RunInfo(
                llm_finished=(finish_reason == "stop"),
                output=output,
                num_tokens_generated=usage.get("completion_tokens", 0),
                num_tokens_guessed=1,
                timed_out=False,
            )

        if config.verbose:
            print(f"  [request failed: {last_error}]", flush=True)
        return RunInfo(llm_finished=False, output="", timed_out=False)
