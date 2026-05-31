import logging
from typing import Any, Dict, Union

# Assuming vishustra_core is available in the Python path or as a package
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node designed to summarize input text.

    This node performs a simple extractive summarization by taking a
    portion of the original text. The target length of the summary
    can be configured via context parameters, offering flexibility
    in controlling the output conciseness.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Summarizes the input text data based on provided context parameters.

        Expected `data`: A string containing the text content to be summarized.
        Expected `context` parameters (optional):
            - 'summary_length': `int`, the desired maximum number of words for the summary.
                                If specified, it takes precedence over 'summary_ratio'.
            - 'summary_ratio': `float`, a value between 0.0 and 1.0, representing the
                               proportion of the original text's word count to retain.

        If neither 'summary_length' nor 'summary_ratio' is provided, the node
        defaults to producing a summary that is approximately 20% of the original
        text's word count, clamped between a minimum of 30 words and a maximum of 150 words.

        Raises:
            ValueError: If `data` is not a string or is empty after stripping whitespace.
            TypeError: If provided context parameters ('summary_length', 'summary_ratio')
                       are of an incorrect type, though these are typically handled via
                       warnings and defaults.

        Returns:
            str: The summarized text. If the input text is too short or summarization
                 parameters result in a zero-length summary, an empty string might be returned.
        """
        if not isinstance(data, str) or not data.strip():
            logger.error(
                f"[{self.node_name}] Invalid input data: expected a non-empty string, "
                f"but received type '{type(data).__name__}'. Data: '{data}'."
            )
            raise ValueError(
                f"{self.node_name} requires a non-empty string as input data for summarization."
            )

        original_text: str = data.strip()
        words = original_text.split()
        original_word_count = len(words)

        if original_word_count == 0:
            logger.info(
                f"[{self.node_name}] Input text is empty after splitting into words. "
                "Returning an empty string as no content to summarize."
            )
            return ""

        target_summary_length: int = 0
        
        # Priority: summary_length from context
        if 'summary_length' in context:
            if isinstance(context['summary_length'], int) and context['summary_length'] > 0:
                target_summary_length = context['summary_length']
                logger.debug(
                    f"[{self.node_name}] Using 'summary_length' from context: {target_summary_length} words."
                )
            else:
                logger.warning(
                    f"[{self.node_name}] Invalid 'summary_length' in context "
                    f"('{context['summary_length']}' of type '{type(context['summary_length']).__name__}'). "
                    "Expected a positive integer. Ignoring and checking 'summary_ratio'."
                )
        
        # Second priority: summary_ratio from context
        if target_summary_length == 0 and 'summary_ratio' in context:
            if isinstance(context['summary_ratio'], (int, float)) and 0.0 < float(context['summary_ratio']) <= 1.0:
                summary_ratio = float(context['summary_ratio'])
                target_summary_length = max(1, int(original_word_count * summary_ratio))
                logger.debug(
                    f"[{self.node_name}] Using 'summary_ratio' from context: {summary_ratio} "
                    f"resulting in ~{target_summary_length} words."
                )
            else:
                logger.warning(
                    f"[{self.node_name}] Invalid 'summary_ratio' in context "
                    f"('{context['summary_ratio']}' of type '{type(context['summary_ratio']).__name__}'). "
                    "Expected a float between 0.0 and 1.0. Ignoring and using default."
                )

        # Fallback: Default summarization heuristic if no context parameters are valid
        if target_summary_length == 0:
            default_ratio = 0.2
            min_default_words = 30
            max_default_words = 150
            
            # Calculate a base length and then apply min/max constraints
            calculated_length = int(original_word_count * default_ratio)
            target_summary_length = max(min_default_words, calculated_length)
            target_summary_length = min(target_summary_length, max_default_words)
            
            logger.debug(
                f"[{self.node_name}] No valid summarization parameters found in context. "
                f"Applying default ratio of {default_ratio} (min {min_default_words} words, "
                f"max {max_default_words} words). Target length: {target_summary_length} words."
            )

        # Ensure the target length does not exceed the original text's word count
        target_summary_length = min(target_summary_length, original_word_count)

        if target_summary_length <= 0:
            logger.info(
                f"[{self.node_name}] Calculated effective summary length is {target_summary_length}. "
                "Returning an empty string."
            )
            return ""

        # Perform simple extractive summarization (taking the first N words)
        # In a more advanced implementation, this would involve NLP models or more sophisticated algorithms.
        summarized_words = words[:target_summary_length]
        summarized_text = " ".join(summarized_words)

        logger.info(
            f"[{self.node_name}] Successfully summarized text from {original_word_count} words "
            f"to {len(summarized_words)} words."
        )
        return summarized_text
