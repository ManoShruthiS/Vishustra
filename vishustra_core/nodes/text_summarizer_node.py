import logging
import re
from typing import Any, Dict

# Assuming BaseNode is available at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node designed for abstractive text summarization.

    This node takes an input string and produces a concise summary. The summarization
    logic presented here is a simulation, demonstrating the configurable parameters
    and data flow that would typically integrate with an external Language Model
    (LLM) service in a production environment.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, generating a summarized version of the text.

        The summarization behavior can be influenced by parameters provided
        in the `context` dictionary.

        Args:
            data (Any): The input data, expected to be a string containing the text
                        to be summarized.
            context (Dict[str, Any]): A dictionary containing runtime configuration
                                      and parameters for the summarization process.
                                      Expected keys:
                                      - `target_length_ratio` (float): Desired summary length
                                        as a ratio of the original text's sentence count.
                                        (e.g., 0.3 for 30%). Defaults to 0.3.
                                      - `min_sentences` (int): The minimum number of sentences
                                        to include in the summary. Defaults to 2.
                                      - `max_sentences` (int): The maximum number of sentences
                                        to include in the summary. Defaults to 5.
                                      - `ellipses_suffix` (bool): If True, appends "..." to the
                                        summary if truncation occurs. Defaults to True.

        Returns:
            Any: A string containing the summarized text.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string, indicating no text
                        to summarize.
        """
        if not isinstance(data, str):
            logger.error(
                f"TextSummarizerNode received invalid input type. Expected 'str', "
                f"but got '{type(data).__name__}'."
            )
            raise TypeError(
                f"TextSummarizerNode requires input 'data' to be a string, "
                f"but received {type(data).__name__}."
            )

        if not data.strip():
            logger.warning("TextSummarizerNode received an empty or whitespace-only string. Returning empty string.")
            return ""

        # Retrieve configuration parameters from context with robust defaults
        target_length_ratio = context.get("target_length_ratio", 0.3)
        min_sentences = context.get("min_sentences", 2)
        max_sentences = context.get("max_sentences", 5)
        ellipses_suffix = context.get("ellipses_suffix", True)

        # Validate configuration parameters
        if not (0.0 < target_length_ratio <= 1.0):
            logger.warning(
                f"Invalid 'target_length_ratio' '{target_length_ratio}'. "
                f"Defaulting to 0.3 for summarization."
            )
            target_length_ratio = 0.3

        if not isinstance(min_sentences, int) or min_sentences < 0:
            logger.warning(f"Invalid 'min_sentences' '{min_sentences}'. Defaulting to 2.")
            min_sentences = 2

        if not isinstance(max_sentences, int) or max_sentences < min_sentences:
            logger.warning(f"Invalid 'max_sentences' '{max_sentences}'. Defaulting to 5.")
            max_sentences = 5
        
        # Simple sentence splitting heuristic for demonstration.
        # In a production system, a more robust NLP library (e.g., NLTK, spaCy)
        # would be used for accurate sentence tokenization.
        sentences = re.split(r'(?<=[.!?])\s+', data)
        num_original_sentences = len(sentences)

        # If the text is already very short or ratio dictates full text, return original
        if num_original_sentences <= min_sentences or target_length_ratio >= 1.0:
            logger.debug(
                f"Input text has {num_original_sentences} sentences (<= {min_sentences}) "
                f"or target ratio {target_length_ratio} >= 1.0. Returning original text."
            )
            return data

        # Calculate the desired number of sentences for the summary
        desired_sentences_count = max(
            min_sentences,
            min(max_sentences, int(num_original_sentences * target_length_ratio))
        )

        # Ensure we don't attempt to take more sentences than are available
        actual_sentences_to_take = min(desired_sentences_count, num_original_sentences)

        if actual_sentences_to_take == 0:
            logger.warning(
                "TextSummarizerNode determined 0 sentences to include. "
                "This might indicate very short or malformed input. Returning original."
            )
            return data # Or return an empty string, depending on desired strictness

        summarized_sentences = sentences[:actual_sentences_to_take]
        summary_text = " ".join(summarized_sentences).strip()

        # Append ellipses if the summary is shorter than the original text and enabled
        if ellipses_suffix and len(summary_text) < len(data.strip()):
            summary_text += "..."

        logger.info(
            f"Text summarization complete. Original sentences: {num_original_sentences}, "
            f"Summarized sentences: {len(summarized_sentences)}. "
            f"Original length: {len(data)} chars, Summary length: {len(summary_text)} chars."
        )

        return summary_text