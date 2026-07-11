import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path within the Vishustra framework
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that simulates abstractive text summarization.

    This node takes a string as input and returns a truncated version,
    simulating a summary based on configurable length ratios and boundaries.
    It can be initialized with default parameters which can then be
    overridden on a per-process basis via the context dictionary.
    """

    def __init__(self, target_ratio: float = 0.25, min_length: int = 50, max_length: int = 500):
        """
        Initializes the TextSummarizerNode with default summarization parameters.

        Args:
            target_ratio: The default target ratio (0 to 1) of the original text length
                          for the summary. For example, 0.25 means the summary aims
                          to be 25% of the original text length.
            min_length: The default minimum length (in characters) the summary should be.
            max_length: The default maximum length (in characters) the summary should be.

        Raises:
            ValueError: If the provided parameters are outside valid ranges.
        """
        if not (0 < target_ratio <= 1):
            logger.error(f"Initialization failed: target_ratio must be between 0 and 1, got {target_ratio}")
            raise ValueError("target_ratio must be between 0 and 1.")
        if not (min_length >= 0):
            logger.error(f"Initialization failed: min_length must be non-negative, got {min_length}")
            raise ValueError("min_length must be non-negative.")
        if not (max_length >= min_length):
            logger.error(f"Initialization failed: max_length ({max_length}) must be >= min_length ({min_length})")
            raise ValueError("max_length must be greater than or equal to min_length.")

        self._target_ratio = target_ratio
        self._min_length = min_length
        self._max_length = max_length
        logger.info(
            f"TextSummarizerNode initialized with default parameters: "
            f"target_ratio={self._target_ratio}, min_length={self._min_length}, "
            f"max_length={self._max_length}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating text summarization.

        It expects a string input and truncates it based on the node's
        configured parameters or values provided in the `context` dictionary.
        The simulation attempts to end on a word boundary and adds an ellipsis
        if truncation occurs.

        Args:
            data: The input text to be summarized (expected to be a string).
            context: A dictionary containing contextual information.
                     It can optionally include override parameters for summarization:
                     - 'summarization_target_ratio' (float): Overrides `target_ratio`.
                     - 'summarization_min_length' (int): Overrides `min_length`.
                     - 'summarization_max_length' (int): Overrides `max_length`.

        Returns:
            A string representing the simulated summary of the input text.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"Received non-string data in TextSummarizerNode: {type(data).__name__}")
            raise TypeError(
                f"Input data for TextSummarizerNode must be a string, "
                f"but got {type(data).__name__}."
            )

        original_text = data.strip()
        original_length = len(original_text)

        # Retrieve effective parameters, allowing context to override defaults
        effective_target_ratio = context.get('summarization_target_ratio', self._target_ratio)
        effective_min_length = context.get('summarization_min_length', self._min_length)
        effective_max_length = context.get('summarization_max_length', self._max_length)

        logger.debug(
            f"Processing text of length {original_length}. "
            f"Effective parameters: target_ratio={effective_target_ratio}, "
            f"min_length={effective_min_length}, max_length={effective_max_length}."
        )

        if original_length <= effective_min_length:
            # If text is already shorter than or equal to min_length, return as-is
            summary = original_text
            logger.debug(f"Original text length ({original_length}) is <= min_length ({effective_min_length}). "
                         "Returning full text as summary.")
        else:
            # Calculate the ideal summary length based on ratio and bounds
            calculated_length = int(original_length * effective_target_ratio)
            target_summary_length = max(effective_min_length, min(calculated_length, effective_max_length))

            if original_length > target_summary_length:
                # Perform truncation and attempt to end on a word boundary
                truncated_text = original_text[:target_summary_length]
                last_space_index = truncated_text.rfind(' ')

                if last_space_index != -1 and last_space_index >= effective_min_length:
                    # Truncate at the last space if it doesn't make the summary too short
                    summary = truncated_text[:last_space_index] + "..."
                else:
                    # Fallback to direct truncation if no suitable space or it's too short
                    summary = truncated_text + "..."
            else:
                # Should ideally not happen if original_length > effective_min_length
                # and target_summary_length is correctly capped by original_length
                summary = original_text

        final_summary_length = len(summary)
        logger.info(
            f"Successfully summarized text from {original_length} characters to "
            f"{final_summary_length} characters."
        )
        return summary

# Basic logging configuration for standalone execution demonstration (optional, often configured globally)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_logger = logging.getLogger(__name__)
    test_logger.info("Demonstrating TextSummarizerNode functionality.")

    # Mock BaseNode for standalone testing if vishustra_core isn't installed
    # In a real project, this part wouldn't be necessary as BaseNode would be available.
    if 'BaseNode' not in globals():
        test_logger.warning("BaseNode not found, creating a mock BaseNode for demonstration.")
        class BaseNode(ABC):
            @abstractmethod
            def process(self, data: Any, context: Dict[str, Any]) -> Any: pass
            @property
            @abstractmethod
            def node_name(self) -> str: pass
        # Re-define TextSummarizerNode to inherit from the mock BaseNode for execution
        class TextSummarizerNode(BaseNode):
            def __init__(self, target_ratio: float = 0.25, min_length: int = 50, max_length: int = 500):
                if not (0 < target_ratio <= 1): raise ValueError("target_ratio must be between 0 and 1.")
                if not (min_length >= 0): raise ValueError("min_length must be non-negative.")
                if not (max_length >= min_length): raise ValueError("max_length must be greater than or equal to min_length.")
                self._target_ratio = target_ratio
                self._min_length = min_length
                self._max_length = max_length
                logger.info(
                    f"TextSummarizerNode (mocked) initialized with target_ratio={target_ratio}, "
                    f"min_length={min_length}, max_length={max_length}"
                )
            @property
            def node_name(self) -> str: return "TextSummarizerNode"
            def process(self, data: Any, context: Dict[str, Any]) -> Any:
                if not isinstance(data, str): raise TypeError(f"Input data must be a string, got {type(data).__name__}.")
                original_text = data.strip()
                original_length = len(original_text)
                effective_target_ratio = context.get('summarization_target_ratio', self._target_ratio)
                effective_min_length = context.get('summarization_min_length', self._min_length)
                effective_max_length = context.get('summarization_max_length', self._max_length)
                logger.debug(f"Mock process call: len={original_length}, params={effective_target_ratio},{effective_min_length},{effective_max_length}")

                if original_length <= effective_min_length:
                    summary = original_text
                else:
                    calculated_length = int(original_length * effective_target_ratio)
                    target_summary_length = max(effective_min_length, min(calculated_length, effective_max_length))

                    if original_length > target_summary_length:
                        truncated_text = original_text[:target_summary_length]
                        last_space_index = truncated_text.rfind(' ')
                        if last_space_index != -1 and last_space_index >= effective_min_length:
                            summary = truncated_text[:last_space_index] + "..."
                        else:
                            summary = truncated_text + "..."
                    else:
                        summary = original_text
                logger.info(f"Mock process result: Original length {original_length}, Summary length {len(summary)}")
                return summary

    summarizer = TextSummarizerNode(target_ratio=0.3, min_length=30, max_length=150)

    long_text = (
        "The quick brown fox jumps over the lazy dog. This is a classic "
        "pangram often used to test typewriters and computer keyboards. "
        "It contains all letters of the English alphabet. For the purpose "
        "of this demonstration, we are extending this text significantly "
        "to ensure that our TextSummarizerNode's truncation logic is properly "
        "exercised and behaves as expected. We want to see how it handles "
        "different lengths and whether it respects word boundaries. This "
        "additional content helps in testing the robustness of the "
        "summarization simulation, ensuring it adheres to the configured "
        "minimum and maximum lengths as well as the target ratio. "
        "Ultimately, the goal is to produce a concise yet coherent summary."
    )
    test_logger.info(f"\nOriginal Text (Length {len(long_text)}):\n{long_text}")
    summary_default = summarizer.process(long_text, {})
    test_logger.info(f"\nSummary (Default Params - Length {len(summary_default)}):\n{summary_default}")

    short_text = "Hello world! This is a short piece of text."
    test_logger.info(f"\nOriginal Text (Length {len(short_text)}):\n{short_text}")
    summary_short = summarizer.process(short_text, {})
    test_logger.info(f"\nSummary (Short Text - Length {len(summary_short)}):\n{summary_short}")

    context_override = {
        'summarization_target_ratio': 0.1,
        'summarization_min_length': 20,
        'summarization_max_length': 70
    }
    test_logger.info(f"\nOriginal Text (Length {len(long_text)}):\n{long_text}")
    summary_override = summarizer.process(long_text, context_override)
    test_logger.info(f"\nSummary (Context Override - Length {len(summary_override)}):\n{summary_override}")

    # Test error handling
    try:
        test_logger.info("\nTesting error handling with non-string data...")
        summarizer.process(12345, {})
    except TypeError as e:
        test_logger.error(f"Caught expected error: {e}")

    try:
        test_logger.info("\nTesting initialization error with invalid parameters...")
        invalid_summarizer = TextSummarizerNode(target_ratio=1.5)
    except ValueError as e:
        test_logger.error(f"Caught expected initialization error: {e}")