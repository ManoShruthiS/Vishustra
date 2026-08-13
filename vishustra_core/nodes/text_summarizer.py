import logging
import re
from typing import Any, Dict, List

# Assuming vishustra_core.nodes.base_node exists in the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that summarizes text input.

    This node implements a heuristic summarization strategy, primarily
    extracting leading sentences based on configured minimum/maximum
    sentence counts and a target word ratio. It's designed to simulate
    a summarization service or model invocation within the orchestration
    framework, providing a structured interface for text summarization
    in a data pipeline.
    """

    def __init__(self, min_sentences: int = 2, max_sentences: int = 5, summary_ratio: float = 0.2):
        """
        Initializes the TextSummarizerNode with default summarization parameters.

        These parameters can be overridden for individual `process` calls via the
        `context` dictionary.

        Args:
            min_sentences (int): The minimum number of sentences to include in the summary.
                                 Must be a positive integer (>= 1).
            max_sentences (int): The maximum number of sentences to include in the summary.
                                 Must be greater than or equal to `min_sentences`.
            summary_ratio (float): A ratio (0.0 to 1.0, exclusive lower bound) indicating
                                   the desired length of the summary relative to the
                                   original text's word count. For example, 0.2 aims for
                                   a summary that is approximately 20% of the original text's
                                   word count.

        Raises:
            ValueError: If `summary_ratio` or sentence bounds are invalid during initialization.
        """
        if not (0.0 < summary_ratio <= 1.0):
            logger.error(f"Invalid summary_ratio during TextSummarizerNode initialization: {summary_ratio}. Must be between 0.0 (exclusive) and 1.0 (inclusive).")
            raise ValueError("summary_ratio must be between 0.0 (exclusive) and 1.0 (inclusive).")
        if not (1 <= min_sentences <= max_sentences):
            logger.error(f"Invalid sentence range during TextSummarizerNode initialization: min={min_sentences}, max={max_sentences}. min_sentences must be positive and less than or equal to max_sentences.")
            raise ValueError("min_sentences must be positive and less than or equal to max_sentences.")

        self._min_sentences = min_sentences
        self._max_sentences = max_sentences
        self._summary_ratio = summary_ratio
        logger.debug(f"[{self.node_name}] Initialized with min_sentences={min_sentences}, max_sentences={max_sentences}, summary_ratio={summary_ratio}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits a given text into a list of sentences.

        Note: This is a basic, regex-based sentence splitter for demonstration
        purposes and may not handle all linguistic nuances (e.g., abbreviations
        like "Dr.", numbered lists, or complex punctuation within sentences).
        For production systems requiring advanced accuracy, consider external
        NLP libraries (e.g., NLTK, spaCy).

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of strings, where each string is a sentence.
        """
        # Regex splits on .!? followed by whitespace, but looks ahead to avoid splitting
        # on common abbreviations (e.g., "Mr. Smith") by ensuring the next character
        # is uppercase or a digit, or it's the end of the string.
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9]|$)', text.strip())
        # Filter out any empty strings that might result from splitting or extra spaces
        return [s.strip() for s in sentences if s.strip()]

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, generating a summary of the text.

        The summarization strategy implemented here is an extractive approach,
        prioritizing initial sentences. It balances configurable minimum and
        maximum sentence counts with a target summary length proportional to
        the original text's word count.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        containing the text to summarize.
            context (Dict[str, Any]): A dictionary providing additional context or
                                      parameters for this specific processing run.
                                      Can include 'min_sentences' (int), 'max_sentences' (int),
                                      and 'summary_ratio' (float) to override instance defaults.
                                      An optional 'node_id' (str) can be provided for logging.

        Returns:
            Any: The summarized text as a string. Returns an empty string if the input
                 is empty or cannot be meaningfully summarized according to the rules.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If configuration parameters resolved from `context` or defaults are invalid.
        """
        node_id_for_logging = context.get('node_id', self.node_name) # Use a specific ID if provided, otherwise the node's name
        logger.debug(f"[{node_id_for_logging}] Starting text summarization process.")

        # --- Input Validation ---
        if not isinstance(data, str):
            logger.error(f"[{node_id_for_logging}] Invalid input data type: Expected 'str', got '{type(data).__name__}'.")
            raise TypeError(f"Input data for '{node_id_for_logging}' must be a string. Received type: {type(data).__name__}.")

        stripped_data = data.strip()
        if not stripped_data:
            logger.warning(f"[{node_id_for_logging}] Input data is an empty string after stripping. Returning empty summary.")
            return ""

        # --- Parameter Resolution ---
        # Prioritize parameters from context for runtime overrides
        min_sentences = context.get('min_sentences', self._min_sentences)
        max_sentences = context.get('max_sentences', self._max_sentences)
        summary_ratio = context.get('summary_ratio', self._summary_ratio)

        # Validate resolved parameters
        if not (0.0 < summary_ratio <= 1.0):
            logger.error(f"[{node_id_for_logging}] Invalid 'summary_ratio' in context: {summary_ratio}. Must be between 0.0 (exclusive) and 1.0 (inclusive).")
            raise ValueError(f"Invalid 'summary_ratio' for '{node_id_for_logging}': {summary_ratio}. Must be between 0.0 (exclusive) and 1.0 (inclusive).")
        if not (1 <= min_sentences <= max_sentences):
            logger.error(f"[{node_id_for_logging}] Invalid sentence range in context: min={min_sentences}, max={max_sentences}. min_sentences must be positive and less than or equal to max_sentences.")
            raise ValueError(f"Invalid sentence range for '{node_id_for_logging}': min={min_sentences}, max={max_sentences}. min_sentences must be positive and less than or equal to max_sentences.")

        # --- Summarization Logic ---
        original_sentences = self._split_into_sentences(stripped_data)
        if not original_sentences:
            logger.warning(f"[{node_id_for_logging}] No distinct sentences found in input text. Returning empty summary.")
            return ""

        total_original_words = len(stripped_data.split())
        target_summary_word_count = int(total_original_words * summary_ratio)

        summary_sentences: List[str] = []
        current_summary_word_count = 0

        for i, sentence in enumerate(original_sentences):
            sentence_word_count = len(sentence.split())

            # Always add if we haven't met the minimum sentence requirement
            if len(summary_sentences) < min_sentences:
                summary_sentences.append(sentence)
                current_summary_word_count += sentence_word_count
            # If we've met minimum, check against max sentences and target word count
            elif len(summary_sentences) < max_sentences and \
                 (current_summary_word_count + sentence_word_count <= target_summary_word_count):
                summary_sentences.append(sentence)
                current_summary_word_count += sentence_word_count
            else:
                # Stop if we've hit max sentences, or adding the next sentence exceeds target
                break

        # Post-processing: If input text was too short to meet min_sentences, include all
        # available sentences up to max_sentences.
        if len(summary_sentences) < min_sentences and len(original_sentences) > len(summary_sentences):
            # Fill up to min_sentences or all available sentences
            for i in range(len(summary_sentences), min(min_sentences, len(original_sentences))):
                summary_sentences.append(original_sentences[i])
        
        # Ensure we never exceed max_sentences, even if min_sentences logic accidentally added too many
        summary_sentences = summary_sentences[:max_sentences]


        final_summary = " ".join(summary_sentences)
        final_summary_word_count = len(final_summary.split())

        logger.info(f"[{node_id_for_logging}] Summarization complete. Original words: {total_original_words}, Summary words: {final_summary_word_count} (Target ratio: {summary_ratio:.2f}). Sentences: {len(summary_sentences)} (Min/Max: {min_sentences}/{max_sentences}).")
        return final_summary