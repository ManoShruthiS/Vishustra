import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A processing node designed to simulate text summarization.

    This node takes a string as input and produces a shorter version
    by extracting the initial sentences. The number of sentences to extract
    can be configured via the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "TextSummarizer"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data (expected to be a string) and returns a simulated summary.

        The summarization logic is a simple extraction of the initial sentences.
        If the text is very short or cannot be reliably split into sentences,
        the original text is returned as a fallback. The number of sentences to extract
        can be specified in the `context` dictionary under the key 'summary_sentences'.

        Args:
            data (Any): The input text to be summarized. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing node. May include 'summary_sentences'
                                       (int) to specify how many sentences to include in the summary.

        Returns:
            str: The simulated summarized text.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error("TextSummarizerNode received non-string data. Type: %s", type(data))
            raise TypeError(f"TextSummarizerNode expects string input, but received {type(data)}")

        text = data.strip()

        if not text:
            logger.warning("TextSummarizerNode received empty string data. Returning empty summary.")
            return ""

        # A very basic simulation of summarization:
        # If the text is very short (e.g., less than 100 characters),
        # return it as is, as it likely doesn't need summarization.
        if len(text) < 100:
            logger.info("Text below minimum length for advanced summarization. Returning original text.")
            return text

        # Split text into sentences using a regex that matches sentence-ending punctuation
        # followed by whitespace. The positive lookbehind ensures punctuation is kept with the sentence.
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out any potential empty strings or excessive whitespace resulting from the split
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            logger.warning("Could not reliably split text into sentences. Returning original text as fallback.")
            return text

        # Determine the number of sentences to include in the summary.
        # Default to 2 if not specified or invalid in the context dictionary.
        num_sentences_to_summarize = context.get('summary_sentences')
        if not isinstance(num_sentences_to_summarize, int) or num_sentences_to_summarize <= 0:
            default_sentences = 2
            logger.debug(
                "Invalid or missing 'summary_sentences' in context (%s). Defaulting to %d sentences.",
                num_sentences_to_summarize, default_sentences
            )
            num_sentences_to_summarize = default_sentences

        # Select the initial sentences for the summary.
        summary_sentences = sentences[:num_sentences_to_summarize]
        
        # Rejoin the selected sentences to form the final summary string.
        summary = " ".join(summary_sentences)
        
        # Ensure the summary ends with a common punctuation mark for better readability,
        # especially if the last chosen sentence didn't originally end with one.
        if summary and not re.search(r'[.!?]$', summary):
            summary += "."

        logger.info(
            "Text summarized (simulated). Original length: %d chars, Summary length: %d chars, "
            "Selected sentences: %d/%d.",
            len(data), len(summary), len(summary_sentences), len(sentences)
        )
        return summary
