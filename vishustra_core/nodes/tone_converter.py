import logging
from typing import Any, Dict

# Assuming the project structure places base_node correctly within vishustra_core.nodes
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A processing node that simulates converting the tone of text data.
    The target tone is specified in the context dictionary under the 'target_tone' key.
    Supported tones for this simulation include: 'formal', 'casual', 'neutral', 'friendly'.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data to convert its tone based on the 'target_tone'
        specified in the context.

        Args:
            data: The input text data (expected to be a string).
            context: A dictionary containing operational context, expected to have
                     'target_tone': str (e.g., 'formal', 'casual', 'neutral', 'friendly').
                     Defaults to 'neutral' if not provided.

        Returns:
            The tone-converted text data (string).

        Raises:
            TypeError: If the input data is not a string.
            ValueError: If an unsupported or invalid 'target_tone' is provided in the context.
        """
        logger.debug(f"[{self.node_name}] Starting process for data (first 50 chars): '{str(data)[:50]}{'...' if len(str(data)) > 50 else ''}'")

        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected str, got {type(data).__name__}.")
            raise TypeError(f"Input data for {self.node_name} must be a string, but got {type(data).__name__}.")

        target_tone: str = context.get('target_tone', 'neutral').lower()
        logger.debug(f"[{self.node_name}] Target tone identified: '{target_tone}'")

        transformed_data: str = data # Initialize with original data

        # Simulate tone conversion logic.
        # In a production Vishustra environment, this would typically involve integration
        # with an LLM via other specialized nodes, or a sophisticated NLP library.
        # Here, we use simple string manipulations for demonstration purposes.
        if target_tone == 'formal':
            # Example: Replace common contractions, add formal elements
            transformed_data = transformed_data.replace("I'm", "I am") \
                                            .replace("don't", "do not") \
                                            .replace("it's", "it is") \
                                            .replace("you're", "you are")
            if not transformed_data.strip().startswith(("Dear", "To Whom It May Concern")):
                transformed_data = "To Whom It May Concern:\n" + transformed_data
            if not transformed_data.strip().endswith((".", "!", "?")):
                transformed_data += "."
            logger.info(f"[{self.node_name}] Applied formal tone transformation.")
        elif target_tone == 'casual':
            # Example: Add informal greetings/closings, use abbreviations, remove formal punctuation
            transformed_data = transformed_data.replace("thank you", "thx") \
                                            .replace("hello", "hey there") \
                                            .replace("you are", "u r") \
                                            .replace("I am", "I'm") # Revert some formalizations
            if not any(char in transformed_data for char in ['!', '?']):
                transformed_data += " :)" # Add a friendly touch
            # Simplified punctuation removal for casual tone simulation
            transformed_data = transformed_data.replace(".", "").replace(",", "")
            logger.info(f"[{self.node_name}] Applied casual tone transformation.")
        elif target_tone == 'friendly':
            # Example: Add friendly words, emojis, warm greetings
            transformed_data = transformed_data.replace("problem", "hiccup") \
                                            .replace("issue", "matter") \
                                            .replace("concern", "thought")
            if ":)" not in transformed_data and ":D" not in transformed_data:
                transformed_data += " :)"
            if not any(g in transformed_data.lower() for g in ["hello", "hi", "hey"]):
                transformed_data = "Hi there! " + transformed_data
            logger.info(f"[{self.node_name}] Applied friendly tone transformation.")
        elif target_tone == 'neutral':
            # For neutral, we primarily normalize and remove excessive stylistic elements.
            transformed_data = ' '.join(transformed_data.split()).strip() # Normalize whitespace
            logger.info(f"[{self.node_name}] Applied neutral tone normalization.")
        else:
            logger.error(f"[{self.node_name}] Unsupported 'target_tone' specified: '{target_tone}'.")
            raise ValueError(
                f"Unsupported 'target_tone' '{target_tone}'. "
                "Supported tones are: 'formal', 'casual', 'friendly', 'neutral'."
            )

        logger.debug(f"[{self.node_name}] Finished process. Transformed data (first 50 chars): '{transformed_data[:50]}{'...' if len(transformed_data) > 50 else ''}'")
        return transformed_data