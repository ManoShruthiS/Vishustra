import logging
import re
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizer(BaseNode):
    """
    A Vishustra processing node designed to simulate text summarization.
    This node extracts key sentences from the input text based on configurable
    length or ratio parameters provided in the execution context.

    It aims to provide a concise representation of longer input text, useful
    in various stages of an LLM orchestration pipeline for pre-processing
    or intermediate summary generation.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name of this processing node.
        """
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by performing a simulated text summarization.

        The summarization logic is based on extracting the first N sentences
        or a percentage of the original text's sentences, configurable via
        the `context` dictionary.

        Args:
            data: The input content to be summarized. Expected to be a string.
            context: A dictionary containing runtime parameters for summarization:
                     - 'summary_length' (int, optional): The target maximum number
                       of sentences for the summary. If both 'summary_length' and
                       'summary_ratio' are provided, 'summary_length' takes precedence.
                     - 'summary_ratio' (float, optional): A float between 0.0 and 1.0
                       representing the desired proportion of original sentences to keep.
                       E.g., 0.3 for a 30% summary.

        Returns:
            A string containing the summarized version of the input text. Returns an
            empty string if the input data is empty or invalid after processing.

        Raises:
            TypeError: If the input `data` is not of type `str`.
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizer received invalid data type: {type(data)}. Expected string.")
            raise TypeError("TextSummarizer node expects 'data' to be a string.")

        if not data.strip():
            logger.warning("TextSummarizer received empty or whitespace-only data. Returning an empty string.")
            return ""

        # Simple sentence tokenization for simulation.
        # This regex splits by common sentence-ending punctuation followed by a space.
        sentences: List[str] = [s.strip() for s in re.split(r'(?<=[.!?])\s+', data.strip()) if s.strip()]
        num_original_sentences = len(sentences)

        if num_original_sentences == 0:
            logger.warning("TextSummarizer could not identify any distinct sentences in the input. Returning original data.")
            return data.strip()
        
        target_num_sentences: int = 0
        summary_length_param = context.get('summary_length')
        summary_ratio_param = context.get('summary_ratio')

        # Prioritize 'summary_length' if provided and valid
        if isinstance(summary_length_param, int) and summary_length_param > 0:
            target_num_sentences = min(summary_length_param, num_original_sentences)
            logger.debug(f"Context parameter 'summary_length' found: {summary_length_param}. Target: {target_num_sentences} sentences.")
        # Otherwise, check 'summary_ratio'
        elif isinstance(summary_ratio_param, (float, int)) and 0.0 < summary_ratio_param <= 1.0:
            target_num_sentences = max(1, int(num_original_sentences * summary_ratio_param)) # Ensure at least 1 sentence
            logger.debug(f"Context parameter 'summary_ratio' found: {summary_ratio_param}. Target: {target_num_sentences} sentences.")
        else:
            # Default behavior: take the first 3 sentences or all if fewer than 3
            default_length = 3
            target_num_sentences = min(default_length, num_original_sentences)
            if summary_length_param is not None or summary_ratio_param is not None:
                logger.warning(
                    f"Invalid or out-of-range 'summary_length' ({summary_length_param}) or 'summary_ratio' "
                    f"({summary_ratio_param}) provided in context. Defaulting to first {target_num_sentences} sentences."
                )
            else:
                logger.debug(f"No specific summary length/ratio in context. Defaulting to first {target_num_sentences} sentences.")

        summarized_sentences = sentences[:target_num_sentences]
        
        # Fallback: if calculated target results in no sentences but original text had them, take the first.
        if not summarized_sentences and num_original_sentences > 0:
            logger.warning(
                "Summarization logic resulted in an empty summary despite input having sentences. "
                "Returning the first original sentence as a minimal fallback."
            )
            summarized_sentences = [sentences[0]]

        # Join the selected sentences to form the summary, ensuring proper spacing.
        summarized_text = ' '.join(summarized_sentences).strip()

        logger.info(
            f"Successfully summarized text from {num_original_sentences} sentences "
            f"to {len(summarized_sentences)} sentences."
        )
        return summarized_text