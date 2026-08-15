import logging
from typing import Any, Dict
import random

# BaseNode is expected to be available from this specific path in the project.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra processing node designed to convert the tone of input text.

    This node simulates converting text to a specified tone (e.g., formal, informal, sarcastic, neutral)
    based on the 'target_tone' parameter provided in the context dictionary. It offers a simple
    demonstration of text transformation within the Vishustra framework.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to convert its textual tone.

        The `context` dictionary is expected to contain a 'target_tone' key,
        which dictates the desired output tone (e.g., 'formal', 'informal',
        'sarcastic', 'neutral'). If 'target_tone' is not provided or is
        unrecognized, the node defaults to a 'neutral' conversion (stripping whitespace).

        Args:
            data: The input text to be processed. Expected to be a string.
            context: A dictionary containing runtime parameters for the node,
                     potentially including 'target_tone' (str).

        Returns:
            The input text with its tone converted as a string.

        Raises:
            TypeError: If the input `data` is not a string, as this node
                       is specifically designed for text processing.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', got '{type(data).__name__}'. "
                "Raising TypeError to prevent downstream issues."
            )
            raise TypeError(f"'{self.node_name}' node expects string data for processing, but received {type(data).__name__}.")

        original_text: str = data
        # Retrieve target_tone from context, default to 'neutral' if not specified
        target_tone: str = context.get("target_tone", "neutral").lower()

        logger.info(f"[{self.node_name}] Initiating tone conversion for text to '{target_tone}' tone.")
        logger.debug(f"[{self.node_name}] Original text snippet: '{original_text[:50]}...'")

        converted_text: str = original_text

        # Simulate tone conversion based on the specified target_tone
        if target_tone == "formal":
            converted_text = self._convert_to_formal(original_text)
            logger.debug(f"[{self.node_name}] Text successfully converted to formal tone.")
        elif target_tone == "informal":
            converted_text = self._convert_to_informal(original_text)
            logger.debug(f"[{self.node_name}] Text successfully converted to informal tone.")
        elif target_tone == "sarcastic":
            converted_text = self._convert_to_sarcastic(original_text)
            logger.debug(f"[{self.node_name}] Text successfully converted to sarcastic tone.")
        elif target_tone == "neutral":
            converted_text = original_text.strip() # Simple neutral conversion: strip leading/trailing whitespace
            logger.debug(f"[{self.node_name}] Text processed for neutral tone (stripped whitespace).")
        else:
            logger.warning(
                f"[{self.node_name}] Unrecognized target tone '{target_tone}' provided in context. "
                "Returning original text after a neutral strip."
            )
            converted_text = original_text.strip() # Fallback to a neutral cleanup

        logger.debug(f"[{self.node_name}] Converted text snippet: '{converted_text[:50]}...'")
        return converted_text

    def _convert_to_formal(self, text: str) -> str:
        """
        Internal helper to simulate converting text to a formal tone.
        This is a rudimentary simulation for demonstration purposes.
        """
        # Basic simulation: sentence capitalization, common contraction expansion.
        sentences = text.split(". ")
        formal_sentences = []
        for sentence in sentences:
            if sentence:
                formal_sentence = sentence.strip()
                if formal_sentence: # Ensure not an empty string after strip
                    formal_sentence = formal_sentence[0].upper() + formal_sentence[1:]
                    formal_sentence = formal_sentence.replace("don't", "do not")
                    formal_sentence = formal_sentence.replace("can't", "cannot")
                    formal_sentence = formal_sentence.replace("it's", "it is")
                    formal_sentence = formal_sentence.replace("i'm", "I am")
                    formal_sentence = formal_sentence.replace("we're", "we are")
                formal_sentences.append(formal_sentence)
        
        # Rejoin and ensure proper sentence termination
        result = ". ".join(filter(None, formal_sentences)) # Filter empty strings
        if result and not result.endswith('.'):
            result += '.'
        return result

    def _convert_to_informal(self, text: str) -> str:
        """
        Internal helper to simulate converting text to an informal tone.
        This is a rudimentary simulation for demonstration purposes.
        """
        informal_text = text.lower()
        informal_text = informal_text.replace("hello", "hey")
        informal_text = informal_text.replace("very", "super")
        informal_text = informal_text.replace("it is", "it's")
        informal_text = informal_text.replace("i am", "i'm")
        informal_text = informal_text.replace("goodbye", "later!")
        return informal_text.strip()

    def _convert_to_sarcastic(self, text: str) -> str:
        """
        Internal helper to simulate converting text to a sarcastic tone.
        This is a rudimentary simulation for demonstration purposes.
        """
        sarcastic_phrases = [
            " (how fascinating)", " (what a surprise)", " (obviously)",
            " (I'm truly thrilled)", " (groundbreaking work)"
        ]
        words = text.split()
        if len(words) > 5: # Only inject sarcasm into sufficiently long text
            # Inject a sarcastic phrase at a random point
            injection_index = random.randint(1, len(words) - 2)
            words.insert(injection_index, random.choice(sarcastic_phrases))
        
        # Capitalize alternating characters for a "mocking" effect for longer texts
        # This is a common visual representation of sarcasm.
        combined_text = " ".join(words)
        mocking_text = []
        for i, char in enumerate(combined_text):
            mocking_text.append(char.upper() if i % 2 == 0 else char.lower())
        
        return "".join(mocking_text).strip()