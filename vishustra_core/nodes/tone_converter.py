import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node is available in the Python environment.
# This import path points to the abstract base class for all Vishustra nodes.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra processing node designed to convert the tone of a given text.
    
    This node expects the input 'data' to be a string and the 'context' 
    dictionary to contain a 'target_tone' key. The 'target_tone' specifies 
    the desired output tone, with support for 'formal', 'informal', 
    'professional', and 'friendly' tones.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Converts the tone of the input text based on the 'target_tone' 
        specified in the context.

        Args:
            data: The input text (expected to be a string) whose tone needs 
                  to be converted.
            context: A dictionary containing operational parameters. 
                     It must include a 'target_tone' (str) key, defining the 
                     desired output tone. Supported values are 'formal', 
                     'informal', 'professional', and 'friendly'.

        Returns:
            The processed text with its tone adjusted to the target tone.

        Raises:
            TypeError: If the 'data' argument is not a string.
            ValueError: If 'target_tone' is missing from the context or 
                        if the provided 'target_tone' is not supported.
        """
        logger.debug(f"[{self.node_name}] Starting process for data (first 50 chars): {str(data)[:50]!r}...")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. "
                f"Expected str, but received {type(data).__name__}."
            )
            raise TypeError(
                f"Input 'data' for ToneConverter must be a string, "
                f"but got {type(data).__name__}."
            )

        target_tone = context.get("target_tone")
        if not target_tone:
            logger.error(f"[{self.node_name}] 'target_tone' key is missing from the context.")
            raise ValueError(
                "Context must contain a 'target_tone' key for the ToneConverter node."
            )
        
        supported_tones = {"formal", "informal", "professional", "friendly"}
        if target_tone not in supported_tones:
            logger.error(
                f"[{self.node_name}] Unsupported target tone: '{target_tone}'. "
                f"Supported tones are: {', '.join(supported_tones)}."
            )
            raise ValueError(
                f"Unsupported 'target_tone': '{target_tone}'. "
                f"Must be one of {', '.join(supported_tones)}."
            )

        original_text = data
        converted_text = original_text # Initialize with original text

        logger.info(f"[{self.node_name}] Attempting to convert tone to '{target_tone}'.")

        # --- Simulated Tone Conversion Logic ---
        # In a real-world scenario, this would involve more sophisticated NLP techniques,
        # such as large language models, rule-based systems, or machine learning models.
        # For this demonstration, we use simple string manipulations to simulate the effect.
        if target_tone == "formal":
            converted_text = (
                original_text.replace("don't", "do not")
                .replace("can't", "cannot")
                .replace("it's", "it is")
                .replace("you're", "you are")
                + "\n\nSincerely,"
            )
        elif target_tone == "informal":
            converted_text = (
                "Hey there! "
                + original_text.replace("is not", "isn't")
                .replace("you are", "you're")
                .replace("I am", "I'm")
                + "\n\nTalk soon!"
            )
        elif target_tone == "professional":
            # Professional tone often requires careful wording; here, we ensure politeness
            # and clarity, adding a standard professional closing.
            converted_text = original_text + "\n\nRegards,"
        elif target_tone == "friendly":
            converted_text = (
                "Hi! 👋 " 
                + original_text 
                + " Hope you're having a great day! 😊"
            )
        
        logger.debug(
            f"[{self.node_name}] Tone conversion complete. "
            f"Original (first 50 chars): '{original_text[:50]}...', "
            f"Converted ('{target_tone}', first 50 chars): '{converted_text[:50]}...'"
        )
        return converted_text
