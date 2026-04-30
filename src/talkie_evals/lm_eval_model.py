"""lm-evaluation-harness wrapper for Talkie."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lm_eval.api.model import LM
from tqdm import tqdm

from talkie_evals.talkie_loader import load_talkie


class TalkieLM(LM):
    """EleutherAI lm-eval-harness adapter for Talkie."""

    def __init__(
        self,
        model_name: str,
        cache_dir: str = "/cache/huggingface",
        apply_talkie_chat_template: bool = False,
        max_length: int = 2048,
    ) -> None:
        super().__init__()
        import torch

        self.model_name = model_name
        self.apply_talkie_chat_template = apply_talkie_chat_template
        self.max_length = max_length
        self.talkie = load_talkie(model_name, cache_dir=cache_dir)
        self.tokenizer = self.talkie.tokenizer
        self._device = self.talkie.device
        self._torch = torch
        self._eot_token_id = self.tokenizer.encode_single_token("<|endoftext|>")
        self._stop_ids = {self._eot_token_id}
        if self.talkie.spec.style == "it":
            self._end_token_id = self.tokenizer.encode_single_token("<|end|>")
            self._stop_ids.add(self._end_token_id)
        else:
            self._end_token_id = None

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def tokenizer_name(self) -> str:
        suffix = "+talkie-chat-template" if self.apply_talkie_chat_template else ""
        return f"{self.model_name}{suffix}"

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        del add_special_tokens, kwargs
        return self.tokenizer.encode(string, allowed_special="all")

    def tok_decode(self, tokens: Sequence[int]) -> str:
        return self.tokenizer.decode([int(token) for token in tokens])

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        if not chat_template and not self.apply_talkie_chat_template:
            return None
        return "<|user|>{prompt}<|end|><|assistant|>"

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        from talkie.chat import Message, format_chat

        messages = [
            Message(role=message["role"], content=message["content"])
            for message in chat_history
        ]
        formatted = format_chat(messages)
        if not add_generation_prompt and formatted.endswith("<|assistant|>"):
            return formatted[: -len("<|assistant|>")]
        return formatted

    def _format_prompt(self, prompt: str) -> str:
        if self.apply_talkie_chat_template and self.talkie.spec.style == "it":
            from talkie.chat import format_prompt

            return format_prompt(prompt)
        return prompt

    def _trim_context(
        self, context_tokens: list[int], continuation_len: int
    ) -> list[int]:
        budget = self.max_length - max(continuation_len, 1)
        if len(context_tokens) <= budget:
            return context_tokens
        return context_tokens[-budget:]

    def _score_tokens(
        self, context_tokens: list[int], continuation_tokens: list[int]
    ) -> tuple[float, bool]:
        if not context_tokens:
            context_tokens = [self.eot_token_id]
        context_tokens = self._trim_context(context_tokens, len(continuation_tokens))
        input_tokens = list(context_tokens)
        total_logprob = 0.0
        is_greedy = True

        with self._torch.no_grad(), self.talkie._autocast:
            for target_token in continuation_tokens:
                token_tensor = self._torch.tensor(
                    input_tokens, dtype=self._torch.long, device=self.talkie.device
                ).unsqueeze(0)
                logits = self.talkie.model(token_tensor)[0]
                logprobs = self._torch.nn.functional.log_softmax(logits, dim=-1)
                top_token = int(self._torch.argmax(logits).item())
                target_token = int(target_token)
                total_logprob += float(logprobs[target_token].item())
                is_greedy = is_greedy and top_token == target_token
                input_tokens.append(target_token)
        return total_logprob, is_greedy

    def loglikelihood(
        self, requests: list[Any], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        results = []
        for request in tqdm(
            requests, desc="Talkie loglikelihood", disable=disable_tqdm
        ):
            context, continuation = request.args
            if self.apply_talkie_chat_template and self.talkie.spec.style == "it":
                context = self._format_prompt(context)
            if context:
                context_tokens, continuation_tokens = self._encode_pair(
                    context, continuation
                )
            else:
                context_tokens = [self.eot_token_id]
                continuation_tokens = self.tok_encode(continuation)
            results.append(self._score_tokens(context_tokens, continuation_tokens))
        return results

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole_tokens = self.tok_encode(context + continuation)
        context_tokens = self.tok_encode(context)
        return context_tokens, whole_tokens[len(context_tokens) :]

    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> list[float]:
        results = []
        for request in tqdm(
            requests, desc="Talkie rolling loglikelihood", disable=disable_tqdm
        ):
            (text,) = request.args
            tokens = self.tok_encode(text)
            if not tokens:
                results.append(0.0)
                continue
            total, _ = self._score_tokens([self.eot_token_id], tokens)
            results.append(total)
        return results

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        outputs = []
        for request in tqdm(
            requests, desc="Talkie greedy generation", disable=disable_tqdm
        ):
            context, gen_kwargs = request.args
            outputs.append(self._generate_one(context, gen_kwargs or {}))
        return outputs

    def _generate_one(self, context: str, gen_kwargs: dict[str, Any]) -> str:
        context = self._format_prompt(context)
        tokens = self.tok_encode(context)
        max_gen_toks = int(
            gen_kwargs.get("max_gen_toks")
            or gen_kwargs.get("max_tokens")
            or gen_kwargs.get("max_length")
            or 256
        )
        until = gen_kwargs.get("until") or []
        if isinstance(until, str):
            until = [until]

        generated: list[int] = []
        with self._torch.no_grad(), self.talkie._autocast:
            for _ in range(max_gen_toks):
                model_tokens = tokens[-self.max_length :]
                token_tensor = self._torch.tensor(
                    model_tokens, dtype=self._torch.long, device=self.talkie.device
                ).unsqueeze(0)
                logits = self.talkie.model(token_tensor)[0]
                next_token = int(self._torch.argmax(logits).item())
                tokens.append(next_token)
                if next_token in self._stop_ids:
                    break
                generated.append(next_token)
                text = self.tok_decode(generated)
                stop_at = _first_stop(text, until)
                if stop_at is not None:
                    return text[:stop_at]
        text = self.tok_decode(generated)
        stop_at = _first_stop(text, until)
        if stop_at is not None:
            return text[:stop_at]
        if self.apply_talkie_chat_template and self.talkie.spec.style == "it":
            from talkie.chat import truncate_at_stop

            text, _ = truncate_at_stop(text)
        return text


def _first_stop(text: str, stop_strings: Sequence[str]) -> int | None:
    positions = [text.find(stop) for stop in stop_strings if stop and stop in text]
    if not positions:
        return None
    return min(positions)
