import gc
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import time


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelConfig:
    model_id: str = "codellama/CodeLlama-13b-Instruct-hf"
    device: str = field(default_factory=_default_device)
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
    timeout: int = 99999  # no timeout by default
    verbose: bool = False  # print a live per-token progress heartbeat
    max_new_tokens: int = 0  # hard cap on accepted tokens (0 = unlimited)
    max_stall: int = 0  # abort after this many accepted tokens add no
    # non-whitespace content (0 = off); stops the model flooding whitespace
    max_tries: int = 0  # abort after this many rejected guesses at a single
    # position (0 = off); stops endless churn when the model won't emit EOS


@dataclass
class RunInfo:
    llm_finished: bool
    output: str
    total_realizability_time: float
    num_tokens_guessed: int
    num_tokens_generated: int
    tries_per_token: Counter
    timed_out: bool = False


class LanguageModelRunner:
    def __init__(self, model_config: ModelConfig = ModelConfig()):
        self.model_config = model_config
        self.device = torch.device(model_config.device)
        self.model, self.tokenizer = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        Load and configure the model and tokenizer.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_id,
            device_map="auto",
            dtype=self.model_config.dtype,
        )
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        return model, tokenizer

    def _tokenize_prompt(
        self, prompt: str, context: str, fixed_prefix: str = ""
    ) -> torch.Tensor:
        """
        Process and tokenize the input prompt with an optional fixed prefix.
        Returns a tensor of token IDs on the model's device.
        """
        messages = [
            {"role": "system", "content": context},
            {"role": "user", "content": prompt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        if fixed_prefix:
            prefix_tokens = self.tokenizer(
                fixed_prefix,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            input_ids = torch.cat([input_ids, prefix_tokens], dim=-1)

        return input_ids.to(self.model.device)

    def _generate_next_token(
        self,
        input_ids: torch.Tensor,
        config: Config,
        generated_tokens: list[int],
        forbidden_tokens: set[int],
        cache: DynamicCache,
    ) -> Any:
        """
        Generate the next token using the model.
        """
        bad_words = [[id] for id in forbidden_tokens] if forbidden_tokens else None
        inp = torch.tensor([list(input_ids[0]) + generated_tokens])
        inp = inp.to(self.model_config.device)
        attention_mask = torch.ones_like(inp)
        if self.tokenizer.eos_token_id in forbidden_tokens:
            eos_token_id = None
        else:
            eos_token_id = self.tokenizer.eos_token_id
        return self.model.generate(
            inp,
            attention_mask=attention_mask,
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
    ) -> RunInfo:
        input_ids = self._tokenize_prompt(prompt, context, fixed_prefix)
        generated_tokens: list[int] = self.tokenizer(
            fixed_prefix, add_special_tokens=False
        )["input_ids"]
        forbidden_tokens: dict = defaultdict(set)
        cache = DynamicCache()
        decoded_output = fixed_prefix
        num_tokens_guessed = 0
        total_realizability_time = 0.0
        tries = 0
        try_counts: Counter[int] = Counter()
        start_time = time.time()
        content_len = len("".join(decoded_output.split()))
        stall = 0

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
            is_final = new_token == self.tokenizer.eos_token_id
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

            if config.verbose:
                elapsed = time.time() - start_time
                preview = decoded_output.replace("\n", " ")[-40:]
                print(
                    f"\r  gen: {len(generated_tokens):>3} tok | "
                    f"retries@pos {tries - 1:>3} | guesses {num_tokens_guessed:>4} | "
                    f"{elapsed:>4.0f}s | …{preview}",
                    end="",
                    flush=True,
                )

            if is_realizable:
                try_counts[tries] += 1
                tries = 0
                generated_tokens.append(new_token)

                # Stall guard: count accepted tokens that add no real (non-whitespace)
                # content, so a whitespace flood can't run forever.
                new_content_len = len("".join(decoded_output.split()))
                if new_content_len > content_len:
                    content_len = new_content_len
                    stall = 0
                else:
                    stall += 1
                if config.max_stall and stall >= config.max_stall:
                    if config.verbose:
                        print(f"\n  [stalled: {stall} tokens with no content]", flush=True)
                    break
                if config.max_new_tokens and len(generated_tokens) >= config.max_new_tokens:
                    if config.verbose:
                        print(f"\n  [hit max_new_tokens={config.max_new_tokens}]", flush=True)
                    break

                if is_final:
                    if config.verbose:
                        print(flush=True)
                    return RunInfo(
                        llm_finished=True,
                        output=decoded_output,
                        total_realizability_time=total_realizability_time,
                        num_tokens_guessed=num_tokens_guessed,
                        num_tokens_generated=len(generated_tokens),
                        tries_per_token=try_counts,
                        timed_out=False,
                    )
            else:
                forbidden_tokens[tuple(generated_tokens)].add(new_token)
                cache.crop(-1)
                if config.max_tries and tries >= config.max_tries:
                    if config.verbose:
                        print(
                            f"\n  [stuck: {tries} rejected guesses at one position]",
                            flush=True,
                        )
                    break
        if config.verbose:
            print(flush=True)
        # We exited without the model emitting EOS (stall / token cap / timeout).
        # Salvage the run if what we have is already a complete, valid program:
        # the model often writes a finished program but won't emit the stop token.
        salvaged = (
            realizability_checker is not None
            and realizability_checker.realizable(decoded_output, True)
        )
        if config.verbose and salvaged:
            print("  [salvaged: output is already a complete valid program]", flush=True)
        return RunInfo(
            llm_finished=salvaged,
            output=decoded_output,
            total_realizability_time=total_realizability_time,
            num_tokens_guessed=num_tokens_guessed,
            num_tokens_generated=len(generated_tokens),
            tries_per_token=try_counts,
            timed_out=time.time() - start_time > config.timeout,
        )

    def __del__(self):
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()
