import logging
from typing import Any, Dict, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to simulate text summarization.
    It extracts a specified number of initial sentences from the input text.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Simulates summarizing the input text by extracting a configurable
        number of initial sentences.

        Args:
            data (Any): The input text to be summarized. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing runtime context.
                                      Can include 'summary_length_sentences' (int)
                                      to specify the number of sentences to extract.
                                      Defaults to 3 sentences if not provided.

        Returns:
            str: The summarized text.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"TextSummarizerNode received non-string input: {type(data)}")
            raise TypeError(f"Input data for TextSummarizerNode must be a string, got {type(data)}.")

        if not data.strip():
            logger.warning("TextSummarizerNode received empty or whitespace-only input. Returning empty string.")
            return ""

        summary_length_sentences = context.get('summary_length_sentences', 3)
        if not isinstance(summary_length_sentences, int) or summary_length_sentences <= 0:
            logger.warning(
                f"Invalid 'summary_length_sentences' in context: {summary_length_sentences}. "
                "Defaulting to 3 sentences."
            )
            summary_length_sentences = 3

        logger.info(
            f"TextSummarizerNode processing text of length {len(data)} to extract "
            f"{summary_length_sentences} sentences."
        )

        # Simple sentence splitting heuristic for simulation purposes.
        # In a real-world scenario, a more robust NLP library (e.g., NLTK, spaCy)
        # would be used for accurate sentence tokenization.
        sentences: List[str] = []
        current_sentence: List[str] = []
        for char in data:
            current_sentence.append(char)
            if char in ['.', '?', '!']:
                sentence_text = "".join(current_sentence).strip()
                if sentence_text:
                    sentences.append(sentence_text)
                current_sentence = []
        
        # Add any remaining text as a sentence if it's not empty
        remaining_text = "".join(current_sentence).strip()
        if remaining_text:
            sentences.append(remaining_text)

        if not sentences:
            logger.info("No sentences detected in the input text. Returning original text.")
            return data.strip()

        summarized_sentences = sentences[:summary_length_sentences]
        summarized_text = " ".join(summarized_sentences)

        if not summarized_text.strip():
            logger.warning(
                f"Summarization resulted in an empty string after processing {len(sentences)} sentences. "
                "Returning original text."
            )
            return data.strip()

        logger.debug(f"TextSummarizerNode successfully summarized text. Original length: {len(data)}, "
                     f"Summarized length: {len(summarized_text)}")
        return summarized_text
