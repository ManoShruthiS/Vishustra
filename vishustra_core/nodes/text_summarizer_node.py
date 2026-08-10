import logging
from typing import Any, Dict

# Assuming vishustra_core is a package at the root or correctly configured in PYTHONPATH
# For the purpose of this isolated file, we'll use a relative import or assume the path is resolved.
# If this were a real project, the path 'vishustra_core.nodes.base_node' implies a package structure.
# For local testing, one might put the BaseNode definition directly in `base_node.py` and import as:
# from .base_node import BaseNode
# Given the project context implies a base class *for* all nodes, and the import instruction,
# I will use the path specified. Let's make an assumption that 'vishustra_core' refers to the provided base class itself
# if it were in a file named `base_node.py` within a `vishustra_core/nodes` directory.
# Since the prompt gave the BaseNode definition directly, I'll adapt to import from that definition's location.
# For demonstration, I will use a direct relative import to reflect it being part of the same conceptual library.
# In a real setup, `from vishustra_core.nodes.base_node import BaseNode` would be the correct external import.

# For this specific task, let's assume the BaseNode is available at a logical path.
# Since the prompt provided the BaseNode definition directly and asked for a file for ONE node,
# I'll embed the BaseNode definition locally for self-containment, as if it was imported.
# In a real project, this would be: `from vishustra_core.nodes.base_node import BaseNode`

# Start of assumed BaseNode definition from project context (would be an import in a real setup)
from abc import ABC, abstractmethod

class BaseNode(ABC):
    """
    Base class for all Vishustra processing nodes.
    Each node must implement the process method.
    """
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data and returns the result.
        """
        pass
        
    @property
    @abstractmethod
    def node_name(self) -> str:
        """Returns the name of the node."""
        pass
# End of assumed BaseNode definition

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra node that simulates abstractive text summarization.
    It takes a string as input and returns a truncated version,
    mimicking a summary based on a configurable maximum length.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input text to generate a simulated summary.

        Expects `data` to be a string. If a non-string is provided,
        a TypeError is raised.

        The summary length can be controlled via `summary_max_length`
        in the `context` dictionary. Defaults to 150 characters.

        Args:
            data: The input text (string) to be summarized.
            context: A dictionary containing additional information,
                     potentially including 'summary_max_length' (int).

        Returns:
            A string representing the simulated summary.

        Raises:
            TypeError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected string, got %s.",
                self.node_name,
                type(data).__name__
            )
            raise TypeError(f"Input data for {self.node_name} must be a string, but received {type(data).__name__}")

        if not data:
            logger.warning("[%s] Received empty string for summarization. Returning empty string.", self.node_name)
            return ""

        summary_max_length = context.get('summary_max_length', 150)
        if not isinstance(summary_max_length, int) or summary_max_length <= 0:
            logger.warning(
                "[%s] Invalid 'summary_max_length' in context (%s). Using default of 150.",
                self.node_name,
                summary_max_length
            )
            summary_max_length = 150

        logger.info(
            "[%s] Starting text summarization for input of length %d with max length %d.",
            self.node_name,
            len(data),
            summary_max_length
        )

        if len(data) <= summary_max_length:
            summary = data
            logger.debug("[%s] Input text is shorter than or equal to max length. Returning original text.", self.node_name)
        else:
            summary = data[:summary_max_length].strip()
            # Ensure we don't end abruptly mid-word and append an ellipsis
            if summary.endswith((' ', ',', '.', ';', '?', '!')):
                summary = summary.rstrip(' .,;?!') # remove trailing punctuation/space if present
            
            # Find the last space to avoid cutting a word in half
            last_space_index = summary.rfind(' ')
            if last_space_index != -1 and last_space_index > summary_max_length - 20: # Ensure we don't cut too much
                summary = summary[:last_space_index]
            
            summary += "..."
            logger.debug("[%s] Text truncated to length %d.", self.node_name, len(summary))

        logger.info(
            "[%s] Text summarization complete. Resulting summary length: %d.",
            self.node_name,
            len(summary)
        )
        return summary

# Example usage (for local testing, not part of the required output)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    summarizer_node = TextSummarizerNode()

    # Test case 1: Normal text
    text_long = ("Vishustra is a highly modular LLM orchestration framework designed "
                 "for complex AI workflows. It enables seamless integration of "
                 "various language models and provides robust tools for data "
                 "processing, context management, and conditional routing. "
                 "This allows developers to build sophisticated AI applications "
                 "with greater flexibility and scalability, abstracting away "
                 "the complexities of managing multiple LLM interactions.")
    summary1 = summarizer_node.process(text_long, {})
    print(f"Original Text (long):\n{text_long}\n")
    print(f"Summary 1 (default max length):\n{summary1}\n")

    # Test case 2: Shorter text than max_length
    text_short = "Vishustra simplifies LLM orchestration."
    summary2 = summarizer_node.process(text_short, {'summary_max_length': 100})
    print(f"Original Text (short):\n{text_short}\n")
    print(f"Summary 2 (max length 100):\n{summary2}\n")

    # Test case 3: Custom max_length in context
    text_medium = ("The new release of Vishustra introduces several performance "
                   "enhancements and new node types, further empowering developers "
                   "to create highly efficient and intelligent AI agents. "
                   "These updates are a direct result of community feedback and "
                   "extensive internal testing, ensuring stability and reliability.")
    summary3 = summarizer_node.process(text_medium, {'summary_max_length': 80})
    print(f"Original Text (medium):\n{text_medium}\n")
    print(f"Summary 3 (max length 80):\n{summary3}\n")

    # Test case 4: Empty string
    summary4 = summarizer_node.process("", {})
    print(f"Original Text (empty):\n''\n")
    print(f"Summary 4:\n'{summary4}'\n")

    # Test case 5: Invalid data type
    try:
        summarizer_node.process(123, {})
    except TypeError as e:
        print(f"Caught expected error: {e}\n")

    # Test case 6: Invalid summary_max_length in context
    summary6 = summarizer_node.process(text_long, {'summary_max_length': 'invalid'})
    print(f"Original Text (long):\n{text_long}\n")
    print(f"Summary 6 (invalid max length, falls back to default):\n{summary6}\n")

    summary7 = summarizer_node.process(text_long, {'summary_max_length': -10})
    print(f"Original Text (long):\n{text_long}\n")
    print(f"Summary 7 (negative max length, falls back to default):\n{summary7}\n")

    # Test case 8: Text ending with punctuation within the cut-off
    text_punctuation = "This is a sentence ending with a period. The quick brown fox jumps over the lazy dog."
    summary8 = summarizer_node.process(text_punctuation, {'summary_max_length': 30})
    print(f"Original Text (punctuation):\n{text_punctuation}\n")
    print(f"Summary 8 (max length 30, handles punctuation):\n{summary8}\n")

    text_comma = "Apple, banana, cherry, date, elderberry, fig, grape, honeydew, kiwi, lemon, mango."
    summary9 = summarizer_node.process(text_comma, {'summary_max_length': 25})
    print(f"Original Text (comma):\n{text_comma}\n")
    print(f"Summary 9 (max length 25, handles punctuation):\n{summary9}\n")