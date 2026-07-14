import logging
import re
from typing import Any, Dict

# Assuming vishustra_core is installed or available in the Python path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that summarizes input text.

    This node takes a string as input and produces a condensed summary.
    The summarization logic is configurable via the `context` dictionary,
    allowing for control over the length and filtering of the summary.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Summarizes the input text based on parameters provided in the context.

        The `context` dictionary can contain:
        - `max_sentences` (int): The maximum number of sentences to include in the summary.
                                 Defaults to 3 if not provided or invalid.
        - `min_words_per_sentence` (int): Minimum words a sentence must have to be considered for summarization.
                                          Defaults to 5. Sentences shorter than this will be ignored.

        Args:
            data (Any): The input data, expected to be a string containing the text to summarize.
            context (Dict[str, Any]): A dictionary containing configuration parameters
                                      for summarization.

        Returns:
            str: The summarized text. Returns an empty string if input is empty
                 or no valid sentences can be extracted.

        Raises:
            TypeError: If `data` is not a string or `context` is not a dictionary.
        """
        if not isinstance(data, str):
            logger.error("TextSummarizerNode received non-string data: %s", type(data))
            raise TypeError(f"TextSummarizerNode expects 'data' to be a string, but got {type(data)}")

        if not isinstance(context, dict):
            logger.error("TextSummarizerNode received non-dict context: %s", type(context))
            raise TypeError(f"TextSummarizerNode expects 'context' to be a dictionary, but got {type(context)}")

        original_text = data.strip()
        if not original_text:
            logger.warning("TextSummarizerNode received empty text for summarization.")
            return ""

        # --- Context Parameter Extraction and Validation ---
        max_sentences = context.get("max_sentences", 3)
        min_words_per_sentence = context.get("min_words_per_sentence", 5)

        if not isinstance(max_sentences, int) or max_sentences <= 0:
            logger.warning(
                "Invalid 'max_sentences' in context: %s (must be a positive integer). Defaulting to 3.",
                max_sentences
            )
            max_sentences = 3
        
        if not isinstance(min_words_per_sentence, int) or min_words_per_sentence <= 0:
            logger.warning(
                "Invalid 'min_words_per_sentence' in context: %s (must be a positive integer). Defaulting to 5.",
                min_words_per_sentence
            )
            min_words_per_sentence = 5

        logger.debug(
            "Attempting to summarize text with max_sentences=%d, min_words_per_sentence=%d",
            max_sentences, min_words_per_sentence
        )

        # --- Simple Sentence Tokenization ---
        # This regex aims to split sentences by common terminal punctuation (period, exclamation, question mark)
        # followed by one or more whitespace characters. While pragmatic for a simulation,
        # more advanced NLP libraries offer robust sentence tokenization for production use.
        sentences = re.split(r'(?<=[.!?])\s+', original_text)
        
        # Filter out empty strings and sentences that do not meet the minimum word count threshold.
        valid_sentences = [
            s.strip() for s in sentences 
            if s.strip() and len(s.strip().split()) >= min_words_per_sentence
        ]

        if not valid_sentences:
            logger.info(
                "No valid sentences found after filtering for min_words_per_sentence=%d. Returning original text if short, otherwise empty.",
                min_words_per_sentence
            )
            # If no sentences meet the criteria, and the original text is short (heuristic),
            # return the original text. Otherwise, return an empty string as effective summarization
            # was not possible based on the parameters.
            if len(original_text.split()) < (min_words_per_sentence * 2):
                return original_text
            return ""

        # --- Summarization Logic ---
        # Take the first 'max_sentences' valid sentences identified.
        summary_sentences = valid_sentences[:max_sentences]
        
        summary = " ".join(summary_sentences)
        
        logger.info(
            "Text summarized. Original characters: %d, Summary characters: %d. Used %d of %d potential sentences.",
            len(original_text), len(summary), len(summary_sentences), len(valid_sentences)
        )

        return summary