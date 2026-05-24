import logging
import re
from typing import Any, Dict

# Assuming BaseNode is correctly located at vishustra_core.nodes.base_node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A processing node responsible for converting the tone of textual data.
    It expects 'data' to be a string and 'context' to contain 'target_tone'
    which dictates the desired output tone (e.g., 'formal', 'informal', 'sarcastic').
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to convert its tone based on
        the 'target_tone' parameter provided in the context dictionary.

        Args:
            data: The input text as a string that needs tone conversion.
            context: A dictionary containing operational parameters. It must
                     include 'target_tone' (str) specifying the desired tone.

        Returns:
            The text with the converted tone if successful. If the target tone
            is unsupported or an error occurs during conversion, the original
            data is returned, and an appropriate warning or error is logged.

        Raises:
            TypeError: If 'data' is not a string.
            ValueError: If 'target_tone' is missing from context or is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Raising TypeError."
            )
            raise TypeError(f"ToneConverter expects string data, but received {type(data).__name__}.")

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.error(
                f"[{self.node_name}] Missing or invalid 'target_tone' in context. "
                f"Expected a non-empty string, but received '{target_tone}'. Raising ValueError."
            )
            raise ValueError(
                f"Context must contain a non-empty 'target_tone' string for ToneConverter, "
                f"but received '{target_tone}'."
            )

        converted_text = data
        lower_target_tone = target_tone.strip().lower()

        try:
            if lower_target_tone == "formal":
                converted_text = self._to_formal(data)
                logger.info(f"[{self.node_name}] Successfully converted text to formal tone.")
            elif lower_target_tone == "informal":
                converted_text = self._to_informal(data)
                logger.info(f"[{self.node_name}] Successfully converted text to informal tone.")
            elif lower_target_tone == "sarcastic":
                converted_text = self._to_sarcastic(data)
                logger.info(f"[{self.node_name}] Successfully converted text to sarcastic tone.")
            else:
                logger.warning(
                    f"[{self.node_name}] Unsupported target tone '{target_tone}'. "
                    "Returning original data without modification."
                )
                # converted_text remains original data
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An unexpected error occurred during tone conversion "
                f"to '{target_tone}': {e}", exc_info=True
            )
            # In case of an unexpected error, return the original data to ensure robustness
            converted_text = data

        return converted_text

    def _to_formal(self, text: str) -> str:
        """
        Simulates conversion of text to a formal tone.
        This implementation capitalizes sentences, replaces contractions, and ensures punctuation.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        formal_sentences = []
        for sentence in sentences:
            if not sentence:
                continue
            formal_sentence = sentence.strip()
            # Capitalize first letter of the sentence
            if formal_sentence:
                formal_sentence = formal_sentence[0].upper() + formal_sentence[1:]

            # Replace common informal contractions with formal equivalents
            formal_sentence = re.sub(r"\b(don't)\b", "do not", formal_sentence, flags=re.IGNORECASE)
            formal_sentence = re.sub(r"\b(can't)\b", "cannot", formal_sentence, flags=re.IGNORECASE)
            formal_sentence = re.sub(r"\b(it's)\b", "it is", formal_sentence, flags=re.IGNORECASE)
            formal_sentence = re.sub(r"\b(i'm)\b", "I am", formal_sentence, flags=re.IGNORECASE)
            formal_sentence = re.sub(r"\b(you're)\b", "you are", formal_sentence, flags=re.IGNORECASE)
            formal_sentence = re.sub(r"\b(we're)\b", "we are", formal_sentence, flags=re.IGNORECASE)

            # Ensure sentence ends with appropriate punctuation
            if not formal_sentence.endswith((".", "!", "?")):
                formal_sentence += "."
            formal_sentences.append(formal_sentence)
        return " ".join(formal_sentences)

    def _to_informal(self, text: str) -> str:
        """
        Simulates conversion of text to an informal tone.
        This implementation lowercases text, removes some punctuation, and adds informal greetings.
        """
        informal_text = text.lower()
        informal_text = re.sub(r"[.,;!?:-]", "", informal_text) # Remove some punctuation
        informal_text = informal_text.replace("hello", "hey").replace("hi", "yo")
        informal_text = informal_text.replace("thank you", "thanks")
        informal_text = informal_text.replace("great", "awesome")
        
        # Add a casual opening if not already present
        if not informal_text.strip().startswith(("hey", "yo")):
            informal_text = "hey there! " + informal_text
        
        return informal_text.strip() + " lol"

    def _to_sarcastic(self, text: str) -> str:
        """
        Simulates conversion of text to a sarcastic tone.
        This implementation uses alternating case and appends a sarcastic remark.
        """
        sarcastic_chars = []
        # Alternate case, ignoring non-alphabetic characters for case change
        toggle_upper = True
        for char in text:
            if char.isalpha():
                sarcastic_chars.append(char.upper() if toggle_upper else char.lower())
                toggle_upper = not toggle_upper
            else:
                sarcastic_chars.append(char)
        sarcastic_text = "".join(sarcastic_chars)

        # Append a common sarcastic phrase
        return sarcastic_text + " (how truly fascinating!)"

