import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra processing node that converts the tone of a given text.
    It simulates tone conversion based on the 'target_tone' specified in the context.
    """

    _supported_tones = {"formal", "casual", "sarcastic", "neutral"}

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Converts the tone of the input text based on the 'target_tone'
        provided in the context. This method simulates the conversion.

        Args:
            data (Any): The input text to be converted. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing node-specific configuration.
                                       Must include 'target_tone' (str), e.g., {'target_tone': 'formal'}.

        Returns:
            Any: The tone-converted text (string).

        Raises:
            ValueError: If 'data' is not a string, or 'target_tone' is missing
                        or invalid in the context.
        """
        if not isinstance(data, str):
            logger.error(f"ToneConverter received invalid data type. Expected str, got {type(data)}")
            raise ValueError("Input data for ToneConverter must be a string.")

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.error("ToneConverter context missing or invalid 'target_tone'. "
                         "Expected a non-empty string for 'target_tone'.")
            raise ValueError("Context must provide a 'target_tone' (string).")

        normalized_target_tone = target_tone.strip().lower()

        if normalized_target_tone not in self._supported_tones:
            logger.warning(f"Unsupported target tone '{target_tone}' requested for ToneConverter. "
                           f"Supported tones are: {', '.join(self._supported_tones)}. "
                           "Returning original data without conversion.")
            return data

        logger.info(f"Attempting to convert text to '{normalized_target_tone}' tone.")
        converted_text = data

        # --- SIMULATED TONE CONVERSION LOGIC ---
        if normalized_target_tone == "formal":
            sentences = data.split('.')
            converted_sentences = []
            for sentence_part in sentences:
                sentence_part = sentence_part.strip()
                if not sentence_part:
                    continue

                # Ensure sentence starts with capital
                if sentence_part and sentence_part[0].islower():
                    sentence_part = sentence_part[0].upper() + sentence_part[1:]

                # Replace common contractions
                sentence_part = (sentence_part.replace("don't", "do not")
                                              .replace("can't", "cannot")
                                              .replace("it's", "it is")
                                              .replace("i'm", "I am")
                                              .replace("we're", "we are"))

                # Add a formal prefix/suffix if appropriate
                if "please" in sentence_part.lower() and not sentence_part.lower().startswith("kindly"):
                    sentence_part = sentence_part.replace("please", "kindly", 1)

                converted_sentences.append(sentence_part)

            converted_text = ". ".join(converted_sentences)
            if converted_text and not converted_text.endswith(('.', '!', '?')):
                converted_text += '.'

        elif normalized_target_tone == "casual":
            converted_text = data.lower()
            # Use common contractions
            converted_text = (converted_text.replace("do not", "don't")
                                          .replace("cannot", "can't")
                                          .replace("it is", "it's")
                                          .replace("i am", "i'm")
                                          .replace("we are", "we're"))
            # Replace formal greetings/closings
            converted_text = converted_text.replace("hello", "hey")
            converted_text = converted_text.replace("thank you", "thanks")
            # Add a casual closing
            if not converted_text.endswith(('.', '!', '?')) and len(converted_text.split()) > 3:
                converted_text += "!"
            converted_text = converted_text.strip().replace("  ", " ") # Clean up spaces

        elif normalized_target_tone == "sarcastic":
            # Simulate "Spongebob case" by alternating case
            converted_chars = []
            upper_case_next = True
            for char in data:
                if char.isalpha():
                    if upper_case_next:
                        converted_chars.append(char.upper())
                    else:
                        converted_chars.append(char.lower())
                    upper_case_next = not upper_case_next
                else:
                    converted_chars.append(char)
            converted_text = "".join(converted_chars)
            # Add a sarcastic phrase if suitable
            if len(converted_text.split()) > 3 and not any(p in converted_text for p in ["?", "!"]):
                converted_text += " (Oh, how original!)"

        elif normalized_target_tone == "neutral":
            # Strip emotive words, ensure consistent casing and punctuation
            converted_text = data.replace("really good", "good").replace("absolutely fantastic", "good").replace("terrible", "bad")
            sentences = [s.strip().capitalize() for s in converted_text.split('.') if s.strip()]
            converted_text = ". ".join(sentences)
            converted_text = converted_text.strip().replace("  ", " ") # Clean up spaces
            if converted_text and not converted_text.endswith(('.', '!', '?')):
                converted_text += '.'

        logger.debug(f"Original text: '{data}' -> Converted to '{normalized_target_tone}' tone: '{converted_text}'")
        return converted_text
