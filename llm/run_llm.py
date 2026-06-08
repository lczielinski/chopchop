import gc
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


@dataclass
class ModelConfig:
    model_id: str = "codellama/CodeLlama-13b-Instruct-hf"
    device: str = "mps"  # Apple Silicon GPU
    # bf16-native models (Qwen, etc.); avoids fp16 overflow.
    dtype: torch.dtype = torch.bfloat16


@dataclass
class Config:
    """
    Configuration for language model generation.
    """

    temperature: float = 0.5
    repetition_penalty: float = 1.0
    top_p: float = 1.0
    top_k: float = 0
    timeout: float = 99999.0  # no timeout by default
    # Reject the attempt if it runs this long without finishing.
    max_new_tokens: int = 100


@dataclass
class RunInfo:
    llm_finished: bool
    output: str
    total_realizability_time: float
    num_tokens_guessed: int
    num_tokens_generated: int
    tries_per_token: Counter[int]
    timed_out: bool = False


class LanguageModelRunner:
    def __init__(self, model_config: ModelConfig | None = None):
        self.model_config = model_config or ModelConfig()
        self.device = torch.device(self.model_config.device)
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.eos_ids = self._resolve_eos_ids()

    def _resolve_eos_ids(self) -> set[int]:
        """Token ids that end generation.

        Chat models often stop on a turn-end token (e.g. Qwen's `<|im_end|>`) rather than
        the classic EOS, and may list several in their generation config; collect them all
        so generation is detected as finished regardless of which the model emits.
        """
        ids: set[int] = set()
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(gen_eos, int):
            ids.add(gen_eos)
        elif isinstance(gen_eos, (list, tuple)):
            ids.update(gen_eos)
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return ids

    def _load_model_and_tokenizer(self):
        """
        Load and configure the model and tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            dtype=self.model_config.dtype,
        ).to(self.device)
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        return model, tokenizer

    def _tokenize_prompt(self, prompt: str, context: str) -> list[int]:
        """
        Process and tokenize the input prompt.
        Returns a flat list of token IDs.
        """
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            return_dict=False,
        )
        return input_ids

    def _generate_next_token(
        self,
        input_ids: list[int],
        config: Config,
        generated_tokens: list[int],
        forbidden_tokens: set[int],
        cache: DynamicCache,
    ) -> Any:
        """
        Generate the next token using the model.
        """
        bad_words = (
            [[token_id] for token_id in forbidden_tokens] if forbidden_tokens else None
        )
        inp = torch.tensor([input_ids + generated_tokens])
        inp = inp.to(self.device)
        if self.tokenizer.eos_token_id in forbidden_tokens:
            eos_token_id = None
        else:
            eos_token_id = self.tokenizer.eos_token_id
        return self.model.generate(
            inp,
            attention_mask=torch.ones_like(inp),
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=eos_token_id,
            max_new_tokens=1,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            bad_words_ids=bad_words,
            repetition_penalty=config.repetition_penalty,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            past_key_values=cache,
        )

    def run(
        self,
        config: Config,
        prompt: str,
        context: str,
        fixed_prefix: str = "",
        realizability_checker=None,
        stream: bool = False,
    ) -> RunInfo:
        input_ids = self._tokenize_prompt(prompt, context)
        generated_tokens: list[int] = self.tokenizer(
            fixed_prefix, add_special_tokens=False
        )["input_ids"]
        forbidden_tokens: defaultdict[tuple[int, ...], set[int]] = defaultdict(set)
        cache = DynamicCache()
        decoded_output = fixed_prefix
        accepted_len = len(decoded_output)  # length of decoded text for accepted tokens
        if stream and decoded_output:
            sys.stdout.write(decoded_output)
            sys.stdout.flush()
        num_tokens_guessed = 0
        total_realizability_time = 0.0
        tries = 0
        try_counts: Counter[int] = Counter()
        start_time = time.time()

        while time.time() - start_time <= config.timeout:
            num_tokens_guessed += 1
            tries += 1
            output = self._generate_next_token(
                input_ids,
                config,
                generated_tokens,
                forbidden_tokens[tuple(generated_tokens)],
                cache,
            )
            new_token: int = output.sequences[0][-1].tolist()
            is_final = new_token in self.eos_ids
            decoded_output = self.tokenizer.decode(
                generated_tokens + [new_token], skip_special_tokens=True
            )

            if realizability_checker is None:
                is_realizable = True
            else:
                check_start = time.time()
                is_realizable = realizability_checker.realizable(
                    decoded_output, is_final
                )
                total_realizability_time += time.time() - check_start

            if is_realizable:
                try_counts[tries] += 1
                tries = 0
                generated_tokens.append(new_token)
                if stream:
                    sys.stdout.write(decoded_output[accepted_len:])
                    sys.stdout.flush()
                accepted_len = len(decoded_output)
                if is_final:
                    return RunInfo(
                        llm_finished=True,
                        output=decoded_output,
                        total_realizability_time=total_realizability_time,
                        num_tokens_guessed=num_tokens_guessed,
                        num_tokens_generated=len(generated_tokens),
                        tries_per_token=try_counts,
                        timed_out=False,
                    )
                # A stuck model can stay "realizable" via whitespace or recursive
                # arithmetic towers; reject attempts that run too long without finishing.
                if len(generated_tokens) >= config.max_new_tokens:
                    break
            else:
                forbidden_tokens[tuple(generated_tokens)].add(new_token)
                cache.crop(-1)
        return RunInfo(
            llm_finished=False,
            output=decoded_output,
            total_realizability_time=total_realizability_time,
            num_tokens_guessed=num_tokens_guessed,
            num_tokens_generated=len(generated_tokens),
            tries_per_token=try_counts,
            timed_out=time.time() - start_time > config.timeout,
        )

    def __del__(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
