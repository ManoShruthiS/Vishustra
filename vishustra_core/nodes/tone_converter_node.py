import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node designed to convert the tone of text data.

    This node accepts a string as input and, based on the 'target_tone'
    specified in the context, applies a set of predefined transformations
    to simulate a change in tone. Currently, it supports 'formal', 'informal',
    and 'neutral' tones.

    Note: This node provides a rule-based simulation of tone conversion.
    For more sophisticated and context-aware tone transformations,
    integration with a large language model (LLM) would typically be employed
    in a production Vishustra pipeline.
    """

    _FORMAL_CONVERSIONS: Dict[str, str] = {
        # Contraction expansions
        "don't": "do not", "can't": "cannot", "won't": "will not", "it's": "it is",
        "you're": "you are", "I'm": "I am", "we're": "we are", "they're": "they are",
        "he's": "he is", "she's": "she is", "isn't": "is not", "aren't": "are not",
        "wasn't": "was not", "weren't": "were not", "haven't": "have not",
        "hasn't": "has not", "hadn't": "had not", "wouldn't": "would not",
        "couldn't": "could not", "shouldn't": "should not", "mustn't": "must not",
        "needn't": "need not",

        # Informal to formal word substitutions
        "awesome": "excellent",
        "cool": "impressive",
        "get in touch": "contact us",
        "a lot": "numerous",
        "stuff": "materials",
        "pretty much": "virtually",
        "kinda": "somewhat",
        "guys": "colleagues",
        "wanna": "want to",
        "gonna": "going to",
        "lemme": "let me",
        "gotta": "have to",
    }

    _INFORMAL_CONVERSIONS: Dict[str, str] = {
        # Contraction contractions
        "do not": "don't", "cannot": "can't", "will not": "won't", "it is": "it's",
        "you are": "you're", "I am": "I'm", "we are": "we're", "they are": "they're",
        "he is": "he's", "she is": "she's", "is not": "isn't", "are not": "aren't",
        "was not": "wasn't", "were not": "weren't", "have not": "haven't",
        "has not": "hasn't", "had not": "hadn't", "would not": "wouldn't",
        "could not": "couldn't", "should not": "shouldn't", "must not": "mustn't",
        "need not": "needn't",

        # Formal to informal word substitutions
        "excellent": "awesome",
        "impressive": "cool",
        "contact us": "get in touch",
        "numerous": "a lot",
        "materials": "stuff",
        "virtually": "pretty much",
        "somewhat": "kinda",
        "colleagues": "guys",
        "want to": "wanna",
        "going to": "gonna",
        "let me": "lemme",
        "have to": "gotta",
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by converting its tone based on the specified context.

        Args:
            data (Any): The input data, expected to be a string that needs tone conversion.
            context (Dict[str, Any]): A dictionary containing processing context parameters.
                                     It *must* include 'target_tone' (str), which specifies
                                     the desired output tone (e.g., 'formal', 'informal', 'neutral').

        Returns:
            Any: The tone-converted string. The type matches the input if conversion is successful.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing from the context, is not a string,
                        or specifies an unsupported tone.
        """
        if not isinstance(data, str):
            logger.error(
                "ToneConverterNode received non-string data. Expected str, but got %s.",
                type(data),
            )
            raise TypeError(
                f"ToneConverterNode expects 'data' to be a string, but received {type(data)}."
            )

        target_tone = context.get("target_tone")
        if not target_tone or not isinstance(target_tone, str):
            logger.error(
                "ToneConverterNode 'target_tone' not found or invalid in context. "
                "Context received: %s",
                context,
            )
            raise ValueError(
                "ToneConverterNode requires a 'target_tone' (str) in the context to operate."
            )

        processed_text = data
        lower_tone = target_tone.lower()
        conversions: Dict[str, str] = {}

        if lower_tone == "formal":
            conversions = self._FORMAL_CONVERSIONS
            logger.info("Applying formal tone conversions.")
        elif lower_tone == "informal":
            conversions = self._INFORMAL_CONVERSIONS
            logger.info("Applying informal tone conversions.")
        elif lower_tone == "neutral":
            logger.info("ToneConverterNode received 'neutral' tone; no text conversion applied.")
            return processed_text
        else:
            logger.error(
                "ToneConverterNode received unsupported 'target_tone': '%s'. "
                "Supported tones are 'formal', 'informal', 'neutral'.",
                target_tone,
            )
            raise ValueError(
                f"Unsupported 'target_tone': '{target_tone}'. "
                "Supported options are 'formal', 'informal', 'neutral'."
            )
        
        # Apply word-based conversions.
        # This simulation iterates through replacements to handle multiple occurrences
        # and attempts to cover both lowercase and title-case for basic sentence structure.
        for original, replacement in conversions.items():
            # Apply conversion for exact match (case-sensitive)
            processed_text = processed_text.replace(original, replacement)
            # Apply conversion for capitalized version (e.g., at sentence start)
            processed_text = processed_text.replace(original.capitalize(), replacement.capitalize())
        
        logger.info(
            "Tone conversion successfully completed for target tone '%s'. "
            "Original data length: %d, Processed data length: %d",
            target_tone, len(data), len(processed_text)
        )
        return processed_text