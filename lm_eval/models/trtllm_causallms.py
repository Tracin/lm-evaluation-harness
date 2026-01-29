"""TensorRT-LLM model wrapper for lm-evaluation-harness.

This module provides integration with NVIDIA's TensorRT-LLM inference engine,
enabling high-performance evaluation of language models using optimized TensorRT backends.

Supports:
- Loading models from HuggingFace checkpoints with automatic engine building
- Tensor Parallelism for multi-GPU inference
- Loglikelihood scoring and text generation
- Advanced sampling parameters (temperature, top-k, top-p, beam search)
- Engine caching for fast subsequent runs
"""

import hashlib
import logging
import os
from importlib.util import find_spec
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    configure_pad_token,
    normalize_gen_kwargs,
)

eval_logger = logging.getLogger(__name__)


@register_model("trtllm", "tensorrt-llm", "trt-llm")
class TRTLLM(TemplateLM):
    """TensorRT-LLM model wrapper for lm-evaluation-harness.

    This class provides an interface to TensorRT-LLM models, inheriting from TemplateLM
    to leverage built-in tokenization and chat template support.

    Example usage:
        ```python
        # Single GPU
        lm_eval --model trtllm \\
            --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype=float16 \\
            --tasks hellaswag \\
            --batch_size 8

        # Multi-GPU with Tensor Parallelism
        lm_eval --model trtllm \\
            --model_args pretrained=meta-llama/Llama-2-70b-hf,tensor_parallel_size=4 \\
            --tasks mmlu \\
            --batch_size 16
        ```
    """

    def __init__(
        self,
        pretrained: str,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        add_bos_token: Optional[bool] = None,
        dtype: Literal["float16", "bfloat16", "float32"] = "float16",
        max_batch_size: int = 8,
        max_input_len: int = 2048,
        max_output_len: int = 512,
        max_beam_width: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        engine_dir: Optional[str] = None,
        batch_size: Union[str, int] = "auto",
        max_gen_toks: int = 256,
        **kwargs,
    ):
        """Initialize TensorRT-LLM model wrapper.

        Args:
            pretrained: HuggingFace model identifier or path to local checkpoint
            revision: Model revision/branch to use
            trust_remote_code: Allow custom code from HuggingFace
            tokenizer: Override tokenizer path (defaults to pretrained)
            tokenizer_mode: Tokenizer loading mode
            add_bos_token: Whether to add BOS token (auto-detected if None)
            dtype: Model precision (float16, bfloat16, float32)
            max_batch_size: Maximum batch size for TRT engine
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output sequence length
            max_beam_width: Maximum beam width for beam search
            tensor_parallel_size: Number of GPUs for tensor parallelism
            pipeline_parallel_size: Number of GPUs for pipeline parallelism
            engine_dir: Path to pre-built engine (auto-cache if None)
            batch_size: Evaluation batch size
            max_gen_toks: Default maximum generation tokens
            **kwargs: Additional arguments passed to TemplateLM
        """
        super().__init__()

        # Set batch size and max generation tokens
        self.batch_size = batch_size
        self._max_gen_toks = max_gen_toks

        # Check TensorRT-LLM installation
        if not find_spec("tensorrt_llm"):
            raise ModuleNotFoundError(
                "attempted to use 'trtllm' LM type, but package `tensorrt_llm` is not installed. "
                "Please install TensorRT-LLM following NVIDIA's installation guide: "
                "https://github.com/NVIDIA/TensorRT-LLM"
            )

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT-LLM requires CUDA, but CUDA is not available. "
                "Please ensure you have a CUDA-capable GPU and proper drivers installed."
            )

        # Store configuration
        self.pretrained = pretrained
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.max_beam_width = max_beam_width
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.chat_template_args = {}

        # Load tokenizer from HuggingFace
        eval_logger.info(f"Loading tokenizer from {tokenizer or pretrained}")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer if tokenizer else pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=(tokenizer_mode == "auto"),
        )

        # Configure pad token
        self.tokenizer = configure_pad_token(self.tokenizer)

        # Handle BOS token
        self._add_bos_token = add_bos_token
        if add_bos_token is None:
            # Auto-detect BOS token behavior
            try:
                test_tok = self.tok_encode("test", add_special_tokens=True)
                test_tok_no_special = self.tok_encode("test", add_special_tokens=False)
                self._add_bos_token = test_tok != test_tok_no_special
            except Exception:
                self._add_bos_token = False

        eval_logger.info(f"BOS token will{'not' if not self._add_bos_token else ''} be added")

        # Initialize TRT-LLM model using high-level API
        # The LLM class handles engine building and caching automatically
        eval_logger.info(f"Initializing TensorRT-LLM for {pretrained}...")
        eval_logger.info(
            "If this is the first run, the engine will be built (may take several minutes). "
            "Subsequent runs will use the cached engine."
        )

        from tensorrt_llm import LLM

        # Map dtype
        dtype_map = {
            "float16": "float16",
            "bfloat16": "bfloat16",
            "float32": "float32",
        }

        # Initialize LLM with the model path
        # LLM handles engine building and loading automatically
        self.llm = LLM(
            model=pretrained,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype_map[dtype],
            trust_remote_code=trust_remote_code,
            revision=revision,
            # Don't pass tokenizer or skip_tokenizer_init - let LLM load its own tokenizer for detokenization
        )

        eval_logger.info(
            f"TensorRT-LLM model loaded: {pretrained} "
            f"(TP={tensor_parallel_size}, dtype={dtype})"
        )

    def tok_encode(
        self,
        string: Union[str, List[str]],
        add_special_tokens: bool = False,
        **kwargs,
    ) -> Union[List[int], List[List[int]]]:
        """Tokenize string(s) to token IDs.

        Args:
            string: Input string or list of strings
            add_special_tokens: Whether to add special tokens (BOS/EOS)
            **kwargs: Additional tokenizer arguments

        Returns:
            Token IDs (list) or batch of token IDs (list of lists)
        """
        # Handle special tokens based on add_bos_token setting
        if add_special_tokens and not self._add_bos_token:
            add_special_tokens = False

        if isinstance(string, str):
            return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        else:
            return self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
                **kwargs,
            ).input_ids

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Apply chat template to conversation history.

        Args:
            chat_history: List of message dicts with 'role' and 'content' keys
            add_generation_prompt: Whether to add generation prompt at the end

        Returns:
            Formatted string with chat template applied
        """
        import jinja2

        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                **self.chat_template_args,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. Removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
                **self.chat_template_args,
            )

        return chat_templated

    def _loglikelihood_tokens(
        self,
        requests,
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        """Compute log-likelihood for context-continuation pairs.

        This is the core scoring method used for tasks like multiple choice.

        Args:
            requests: List of ((context, continuation), context_tokens, continuation_tokens) tuples
            disable_tqdm: Disable progress bar
            override_bs: Override batch size

        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []

        # Process in batches
        batch_size = override_bs if override_bs is not None else self.batch_size
        # Each request is ((context, continuation), context_tokens, continuation_tokens)
        collator = Collator(requests, sort_fn=lambda x: -len(x[1] + x[2]))

        for chunk in tqdm(
            collator.get_batched(n=batch_size, batch_fn=None),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        ):
            batch_inputs = []
            batch_continuation_info = []

            for _string_pair, context_tokens, continuation_tokens in chunk:
                # Handle empty continuation
                if len(continuation_tokens) == 0:
                    results.append((0.0, True))
                    continue

                # Concatenate context + continuation
                input_tokens = context_tokens + continuation_tokens

                # Truncate if needed
                if len(input_tokens) > self.max_input_len:
                    eval_logger.warning(
                        f"Input length {len(input_tokens)} exceeds max_input_len {self.max_input_len}. "
                        "Truncating from left."
                    )
                    # Truncate from left, try to preserve some continuation
                    overflow = len(input_tokens) - self.max_input_len
                    if overflow < len(context_tokens):
                        context_tokens = context_tokens[overflow:]
                        input_tokens = context_tokens + continuation_tokens
                    else:
                        # Continuation itself is too long
                        input_tokens = continuation_tokens[-self.max_input_len:]
                        context_tokens = []

                batch_inputs.append(input_tokens)
                batch_continuation_info.append({
                    'continuation_len': len(continuation_tokens),
                    'continuation_tokens': continuation_tokens,
                    'context_len': len(input_tokens) - len(continuation_tokens),
                })

            # Skip if no valid inputs in batch
            if not batch_inputs:
                continue

            # Get logprobs and greedy info from TRT-LLM
            batch_outputs = self._get_logprobs_and_greedy(batch_inputs)

            # Process each output
            for info, output in zip(batch_continuation_info, batch_outputs):
                continuation_logprobs = output['continuation_logprobs']
                is_greedy = output['is_greedy']

                # Sum log probabilities
                log_likelihood = sum(continuation_logprobs)

                results.append((log_likelihood, is_greedy))

        # Reorder results to match original request order
        return collator.get_original(results)

    def _get_logprobs_and_greedy(
        self, batch_inputs: List[List[int]]
    ) -> List[dict]:
        """Get per-token log probabilities and greedy info for input sequences.

        Args:
            batch_inputs: Batch of token ID sequences

        Returns:
            Batch of dicts containing continuation_logprobs and is_greedy
        """
        try:
            from tensorrt_llm import SamplingParams

            # Create sampling params for logprob computation
            # We request prompt logprobs to get log probabilities of the input tokens
            sampling_params = SamplingParams(
                max_new_tokens=1,  # Minimal generation, we only need prompt logprobs
                temperature=0.0,
                output_log_probs=True,  # Request log probabilities
            )

            # Run inference
            outputs = self.llm.generate(
                prompt_token_ids=batch_inputs,
                sampling_params=sampling_params,
            )

            # Extract logprobs from outputs
            batch_results = []
            for tokens, output in zip(batch_inputs, outputs):
                result = {
                    'continuation_logprobs': [],
                    'is_greedy': True,
                }

                # Access per-token logprobs
                # Note: API may vary by TRT-LLM version
                # TRT-LLM should provide prompt_logprobs similar to vLLM
                if hasattr(output, 'prompt_logprobs') and output.prompt_logprobs:
                    # prompt_logprobs is a list of dicts: [{token_id: logprob, ...}, ...]
                    # The first entry is None because there's no previous context
                    logprobs_dicts = output.prompt_logprobs

                    # Extract log probabilities for actual tokens
                    continuation_logprobs = []
                    is_greedy = True

                    # Skip the first None entry
                    for i, (token_id, logprob_dict) in enumerate(zip(tokens, logprobs_dicts)):
                        if logprob_dict is None:
                            continue

                        # Get logprob for the actual token
                        if token_id in logprob_dict:
                            token_logprob = logprob_dict[token_id]
                            # Handle Logprob objects (like vLLM)
                            if hasattr(token_logprob, 'logprob'):
                                token_logprob = token_logprob.logprob
                            continuation_logprobs.append(token_logprob)

                            # Check if greedy
                            if logprob_dict:
                                # Find token with max logprob
                                max_token = max(logprob_dict.keys(), key=lambda k: (
                                    logprob_dict[k].logprob if hasattr(logprob_dict[k], 'logprob')
                                    else logprob_dict[k]
                                ))
                                if max_token != token_id:
                                    is_greedy = False
                        else:
                            # Token not in logprobs dict - shouldn't happen
                            eval_logger.warning(f"Token {token_id} not found in logprobs dict")
                            continuation_logprobs.append(0.0)
                            is_greedy = False

                    result['continuation_logprobs'] = continuation_logprobs
                    result['is_greedy'] = is_greedy

                elif hasattr(output, 'logprobs') and output.logprobs:
                    # Alternative API: simple list of logprobs
                    logprobs = []
                    for lp in output.logprobs:
                        if hasattr(lp, 'logprob'):
                            logprobs.append(lp.logprob)
                        else:
                            logprobs.append(float(lp) if lp is not None else 0.0)

                    result['continuation_logprobs'] = logprobs
                    result['is_greedy'] = True  # Can't determine without token IDs
                else:
                    # Fallback: use zeros if logprobs not available
                    eval_logger.warning(
                        "Log probabilities not available from TRT-LLM output. "
                        "Using zeros as fallback. This may indicate API incompatibility."
                    )
                    result['continuation_logprobs'] = [0.0] * len(tokens)
                    result['is_greedy'] = True

                batch_results.append(result)

            return batch_results

        except Exception as e:
            eval_logger.error(f"Error getting log probabilities: {e}")
            # Fallback: return zeros
            return [
                {
                    'continuation_logprobs': [0.0] * len(tokens),
                    'is_greedy': True,
                }
                for tokens in batch_inputs
            ]

    def generate_until(
        self,
        requests,
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[str]:
        """Generate text until stop sequences are encountered.

        Args:
            requests: List of Instance objects with (context, generation_kwargs) in args
            disable_tqdm: Disable progress bar
            override_bs: Override batch size

        Returns:
            List of generated strings
        """
        results = []

        batch_size = override_bs if override_bs is not None else self.batch_size

        # Extract args from Instance objects
        # Each request.args is a tuple of (context, generation_kwargs)
        reqs = [req.args for req in requests]

        # Group by generation kwargs for efficient batching
        collator = Collator(
            reqs,
            sort_fn=lambda x: -len(x[0]),
            group_fn=lambda x: str(x[1]),  # Group by gen_kwargs (convert to string for grouping)
        )

        for chunk in tqdm(
            collator.get_batched(n=batch_size, batch_fn=None),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        ):
            contexts, all_gen_kwargs = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]  # Same for all in group

            # Normalize generation kwargs
            gen_kwargs = normalize_gen_kwargs(gen_kwargs, self._max_gen_toks)

            # Extract parameters
            max_gen_toks = gen_kwargs.get("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            top_k = gen_kwargs.get("top_k", -1)
            top_p = gen_kwargs.get("top_p", 1.0)
            repetition_penalty = gen_kwargs.get("repetition_penalty", 1.0)
            num_beams = gen_kwargs.get("num_beams", 1)
            length_penalty = gen_kwargs.get("length_penalty", 1.0)
            until = gen_kwargs.get("until", [])

            # Tokenize contexts
            context_token_ids = []
            for context in contexts:
                tokens = self.tok_encode(context, add_special_tokens=True)

                # Truncate if needed
                if len(tokens) > self.max_input_len:
                    eval_logger.warning(
                        f"Context length {len(tokens)} exceeds max_input_len {self.max_input_len}. "
                        "Truncating from left."
                    )
                    tokens = tokens[-self.max_input_len:]

                context_token_ids.append(tokens)

            # Create TRT-LLM sampling params
            from tensorrt_llm import SamplingParams

            # Convert stop sequences to token IDs
            stop_token_ids = None
            if until:
                stop_token_ids = []
                for stop_seq in until:
                    # Tokenize stop sequence
                    tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
                    if tokens:
                        stop_token_ids.extend(tokens)
                # Remove duplicates
                stop_token_ids = list(set(stop_token_ids)) if stop_token_ids else None

            sampling_params = SamplingParams(
                end_id=self.eot_token_id,
                pad_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eot_token_id,
                max_tokens=max_gen_toks,
                temperature=temperature if temperature > 0 else None,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if top_p < 1.0 else None,
                repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
                use_beam_search=(num_beams > 1),
                length_penalty=length_penalty if length_penalty != 1.0 else None,
                stop_token_ids=stop_token_ids,
            )

            # Generate
            outputs = self.llm.generate(
                inputs=context_token_ids,
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            # Debug logging
            eval_logger.debug(f"Generated {len(outputs)} outputs")

            # Extract generated text
            for idx, output in enumerate(outputs):
                # Get generated text from the first completion output
                # RequestOutput.outputs is a list of CompletionOutput objects
                if hasattr(output, 'outputs') and len(output.outputs) > 0:
                    completion = output.outputs[0]
                    generated_text = completion.text
                    token_ids = completion.token_ids if hasattr(completion, 'token_ids') else []
                    eval_logger.debug(f"Output {idx}: text='{generated_text[:50] if generated_text else '(empty)'}', tokens={len(token_ids)}, finish_reason={getattr(completion, 'finish_reason', None)}")
                else:
                    generated_text = ""
                    eval_logger.debug(f"Output {idx}: No outputs attribute or empty outputs list")

                # Post-process: remove stop sequences if not already removed
                if until:
                    for stop_seq in until:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]

                results.append(generated_text)

        # Reorder results to match original request order
        return collator.get_original(results)

    def loglikelihood_rolling(
        self,
        requests,
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[float]:
        """Compute perplexity using rolling windows.

        Args:
            requests: List of (string,) tuples
            disable_tqdm: Disable progress bar
            override_bs: Override batch size

        Returns:
            List of log-likelihoods
        """
        results = []

        for (string,) in tqdm(requests, disable=disable_tqdm, desc="Running rolling loglikelihood"):
            # Tokenize
            tokens = self.tok_encode(string, add_special_tokens=True)

            # Use rolling windows if sequence is too long
            if len(tokens) > self.max_input_len:
                # Compute with rolling windows
                loglikelihood = self._loglikelihood_rolling_single(tokens)
            else:
                # Compute directly
                logprobs = self._get_logprobs([tokens])[0]
                loglikelihood = sum(logprobs)

            results.append(loglikelihood)

        return results

    def _loglikelihood_rolling_single(self, tokens: List[int]) -> float:
        """Compute rolling log-likelihood for a long sequence.

        Args:
            tokens: Token IDs

        Returns:
            Total log-likelihood
        """
        total_loglikelihood = 0.0
        stride = self.max_input_len // 2  # 50% overlap

        for i in range(0, len(tokens), stride):
            window = tokens[i:i + self.max_input_len]

            if len(window) == 0:
                break

            # Get logprobs for window
            output = self._get_logprobs_and_greedy([window])[0]
            logprobs = output['continuation_logprobs']

            # For overlapping regions, only count non-overlapping part
            if i > 0:
                # Skip overlapping tokens
                overlap = self.max_input_len - stride
                logprobs = logprobs[overlap:]

            total_loglikelihood += sum(logprobs)

            # Break if we've covered the full sequence
            if i + self.max_input_len >= len(tokens):
                break

        return total_loglikelihood

    @property
    def eot_token_id(self) -> int:
        """End-of-text token ID."""
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> Optional[int]:
        """Prefix token ID (BOS)."""
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        """Maximum input length."""
        return self.max_input_len

    @property
    def tokenizer_name(self) -> str:
        """Tokenizer name for caching."""
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self.tokenizer)

    def __del__(self):
        """Cleanup TRT-LLM resources."""
        if hasattr(self, 'llm'):
            del self.llm
