import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to simulate converting the tone of text data.

    This node takes text as input and conceptually transforms its tone
    based on a specified target tone. In a real-world scenario within Vishustra,
    this would typically involve an interaction with an underlying Large Language Model
    to perform the actual tone transformation.
    """

    def __init__(self, target_tone: str = "neutral"):
        """
        Initializes the ToneConverterNode with a specified target tone.

        Args:
            target_tone (str): The desired tone to convert the input text to.
                               Examples include "formal", "informal", "sarcastic",
                               "joyful", "neutral", etc. Case-insensitive.

        Raises:
            ValueError: If 'target_tone' is not a non-empty string.
        """
        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.error("Initialization failed: target_tone must be a non-empty string.")
            raise ValueError("target_tone must be a non-empty string for ToneConverterNode.")
        self._target_tone = target_tone.strip().lower()
        logger.debug(f"ToneConverterNode initialized with target_tone: '{self._target_tone}'.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node, including its configured target tone.
        """
        return f"ToneConverter:{self._target_tone.capitalize()}"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating a tone conversion.

        This method expects the 'data' to be a string (text). It prepends a
        marker indicating the simulated tone conversion. In a full Vishustra
        implementation, this would involve calling an LLM service with the
        input text and the 'target_tone' parameter.

        The 'context' dictionary can be utilized to pass LLM client instances,
        specific model configurations, or user preferences relevant to the
        tone conversion process.

        Args:
            data (Any): The input data, expected to be a string containing text.
            context (Dict[str, Any]): A dictionary providing contextual information
                                      for the processing, such as LLM client,
                                      model parameters, or user settings.

        Returns:
            Any: The processed data, which is a string prefixed with a tone conversion
                 marker, or the original data if it was an empty string.

        Raises:
            TypeError: If the input 'data' is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input type for {self.node_name}. Expected 'str', "
                f"but received '{type(data).__name__}'. Data: '{data}'."
            )
            raise TypeError(f"{self.node_name} expects string data for tone conversion.")

        if not data.strip():
            logger.warning(
                f"Received empty string data for tone conversion in {self.node_name}. "
                "Returning original empty string."
            )
            return data

        # Simulate tone conversion. In a real scenario, this would involve
        # an API call to an LLM, possibly using resources from the 'context'.
        # Example of potential LLM call (conceptual):
        # llm_client = context.get("llm_client")
        # if llm_client:
        #     processed_data = llm_client.convert_tone(text=data, target_tone=self._target_tone)
        # else:
        #     logger.warning("No LLM client found in context. Performing simulated tone conversion.")
        processed_data = f"[Tone: {self._target_tone.capitalize()}]: {data}"

        logger.info(
            f"Successfully simulated tone conversion for {self.node_name}. "
            f"Target tone: '{self._target_tone}'."
        )
        logger.debug(f"Original text (first 50 chars): '{data[:50]}...'")
        logger.debug(f"Processed text (first 50 chars): '{processed_data[:50]}...'")

        return processed_data