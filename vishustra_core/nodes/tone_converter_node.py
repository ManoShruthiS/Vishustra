import logging
from typing import Any, Dict, Optional

# Assuming the BaseNode is located here as per project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra processing node that simulates the conversion of text data
    to a specified tone. In a production environment, this would interface
    with a large language model or a specialized API.
    """

    def __init__(self):
        """
        Initializes the ToneConverter node.
        No specific configuration is required at instantiation for this simulation.
        """
        super().__init__()
        logger.debug("ToneConverter node initialized.")

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this processing node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to convert its tone based on context parameters.

        Args:
            data (Any): The input data to be processed. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing runtime parameters.
                                      Expected to contain 'target_tone' (str).

        Returns:
            Any: The tone-converted string data, or original data if conversion fails
                 gracefully, or raises an error for critical issues.

        Raises:
            ValueError: If the input 'data' is not a string.
        """
        logger.debug("ToneConverter received data for processing. Context: %s", context)

        if not isinstance(data, str):
            logger.error(
                "ToneConverter expects string data for tone conversion, but received type: %s. "
                "Failing early to ensure data integrity.", type(data)
            )
            raise ValueError("Input 'data' must be a string for ToneConverter.")

        target_tone: Optional[str] = context.get("target_tone")

        if not target_tone or not isinstance(target_tone, str):
            logger.warning(
                "No valid 'target_tone' found in context or it's not a string. "
                "Defaulting to 'neutral' tone. Context: %s", context
            )
            target_tone = "neutral"
        else:
            target_tone = target_tone.lower().strip() # Normalize tone string

        try:
            converted_data: str = self._simulate_tone_conversion(data, target_tone)
            logger.info("Successfully converted text to '%s' tone.", target_tone)
            logger.debug(
                "Original text (first 50 chars): '%s', Converted text (first 50 chars): '%s'",
                data[:50] + "..." if len(data) > 50 else data,
                converted_data[:50] + "..." if len(converted_data) > 50 else converted_data
            )
            return converted_data
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during tone conversion for target_tone '%s'. "
                "Returning original data to prevent pipeline failure.", target_tone
            )
            return data # Gracefully return original data on unexpected conversion errors

    def _simulate_tone_conversion(self, text: str, tone: str) -> str:
        """
        Simulates the tone conversion of the given text based on the specified tone.
        This method is a placeholder for actual LLM interactions.

        Args:
            text (str): The input text to convert.
            tone (str): The target tone (e.g., "formal", "casual", "professional").

        Returns:
            str: The text with the simulated tone applied.
        """
        # In a real-world Vishustra deployment, this would invoke an LLM API
        # with a carefully crafted prompt for tone transformation.
        
        if tone == "formal":
            return f"In a formal manner: {text}"
        elif tone == "casual":
            return f"Just casually saying: {text}"
        elif tone == "professional":
            return f"From a professional standpoint: {text}"
        elif tone == "humorous":
            return f"Here's a giggle for ya: {text} (chuckle, chuckle!)"
        elif tone == "serious":
            return f"On a serious note: {text}"
        elif tone == "neutral":
            return text  # No change for neutral tone
        else:
            logger.warning(
                "Unsupported target tone '%s' for simulation. Returning original text.", tone
            )
            return text
