import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node that simulates converting the tone of text data.

    This node expects the input `data` to be a string and the `context`
    dictionary to contain a 'target_tone' key specifying the desired tone.
    Supported tones for simulation are 'professional', 'casual', 'formal', 'friendly'.

    The tone conversion is simulated using a predefined mapping of common phrases.
    It performs case-insensitive, whole-word replacements to illustrate the concept.
    In a production LLM orchestration framework, this would typically involve
    calling an external LLM or a specialized NLP model.
    """

    # Internal mapping of common phrases/words for tone conversion simulation.
    # Keys are sorted by length (desc) during processing to ensure longer phrases are matched first.
    _TONE_MAPPINGS: Dict[str, Dict[str, str]] = {
        "professional": {
            "hello": "Greetings.",
            "hi": "Good day.",
            "hey": "Hello.",
            "need to": "It is required to",
            "get it done": "complete the task",
            "asap": "at your earliest convenience",
            "thanks": "Thank you for your attention.",
            "bye": "Sincerely.",
            "it's": "it is",
            "i'm": "I am",
            "you're": "you are",
            "we'll": "we will",
            "can't": "cannot",
            "don't": "do not",
            "won't": "will not",
            "shouldn't": "should not",
            "isn't": "is not",
            "aren't": "are not",
            "let's": "let us",
            "we're": "we are",
            "wouldn't": "would not",
            "couldn't": "could not",
        },
        "casual": {
            "greetings": "Hey there,",
            "good day": "Hi,",
            "esteemed colleague": "Friend,",
            "it is required to": "Gotta",
            "complete the task": "get it done",
            "at your earliest convenience": "ASAP",
            "thank you for your attention": "Thanks!",
            "sincerely": "Later!",
            "respectfully yours": "Cheers!",
            "it is imperative that we": "We really need to",
            "bring this to completion": "finish this up",
            "we extend our gratitude": "Much appreciated!",
            "it is": "it's",
            "i am": "I'm",
            "you are": "you're",
            "we will": "we'll",
            "cannot": "can't",
            "do not": "don't",
            "will not": "won't",
            "should not": "shouldn't",
            "is not": "isn't",
            "are not": "aren't",
            "let us": "let's",
            "we are": "we're",
            "would not": "wouldn't",
            "could not": "couldn't",
        },
        "formal": {
            "hello": "Greetings.",
            "hi": "Good day.",
            "hey": "Esteemed colleague,",
            "need to": "It is imperative that we",
            "get it done": "bring this to completion",
            "asap": "expeditiously",
            "thanks": "We extend our gratitude.",
            "bye": "Respectfully yours.",
            "it's": "it is",
            "i'm": "I am",
            "you're": "you are",
            "we'll": "we will",
            "can't": "cannot",
            "don't": "do not",
            "won't": "will not",
            "shouldn't": "should not",
            "isn't": "is not",
            "aren't": "are not",
            "let's": "let us",
            "we're": "we are",
            "wouldn't": "would not",
            "couldn't": "could not",
        },
        "friendly": {
            "hello": "Hi there!",
            "hi": "Hey!",
            "hey": "What's up?",
            "need to": "We should probably",
            "get it done": "knock this out",
            "asap": "super soon",
            "thanks": "Cheers!",
            "bye": "Talk soon!",
            "it is required to": "It's a good idea to",
            "complete the task": "finish up",
            "it is imperative that we": "Let's definitely",
            "bring this to completion": "wrap this up",
            "we extend our gratitude": "Thanks a bunch!",
            "at your earliest convenience": "when you get a chance",
            "sincerely": "Best,",
            "respectfully yours": "Talk soon!",
            "greetings": "Hi!",
            "good day": "Hello!",
            "esteemed colleague": "Hey buddy,",
        }
    }

    def __init__(self):
        """Initializes the ToneConverterNode."""
        logger.info(f"[{self.node_name}] Node initialized.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data to convert its tone based on the context.

        This method simulates tone conversion by replacing phrases in the input
        `data` with tone-appropriate alternatives defined in `_TONE_MAPPINGS`.
        The replacement is case-insensitive and respects word boundaries.

        Args:
            data: The input text data (expected to be a string).
            context: A dictionary containing processing parameters,
                     expected to have 'target_tone' (str).

        Returns:
            The tone-converted string.

        Raises:
            TypeError: If `data` is not a string.
            ValueError: If 'target_tone' is missing from context or
                        is not a supported tone.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, "
                f"but received {type(data).__name__}. Data: {data!r}"
            )
            raise TypeError(
                f"ToneConverterNode expects string input data, but received {type(data).__name__}"
            )

        target_tone = context.get("target_tone")
        if not target_tone:
            logger.error(f"[{self.node_name}] 'target_tone' key is missing in context.")
            raise ValueError(
                "Context must contain a 'target_tone' key for ToneConverterNode."
            )

        target_tone_lower = target_tone.lower()
        if target_tone_lower not in self._TONE_MAPPINGS:
            supported_tones = ", ".join(self._TONE_MAPPINGS.keys())
            logger.error(
                f"[{self.node_name}] Unsupported target tone '{target_tone}'. "
                f"Supported tones are: {supported_tones}."
            )
            raise ValueError(
                f"Unsupported target tone: '{target_tone}'. "
                f"Supported tones are: {supported_tones}"
            )

        converted_text = data
        tone_map = self._TONE_MAPPINGS[target_tone_lower]

        # Sort phrases by length in descending order to ensure longer, more specific
        # phrases are replaced before their shorter constituents (e.g., "get it done" before "get it").
        sorted_phrases = sorted(tone_map.items(), key=lambda item: len(item[0]), reverse=True)

        for original_phrase, replacement_phrase in sorted_phrases:
            # Use re.sub for robust case-insensitive, whole-word replacement.
            # \b ensures that only whole words/phrases are matched.
            # re.escape() is used to treat original_phrase literally, in case it contains regex special characters.
            pattern = r'\b' + re.escape(original_phrase) + r'\b'
            converted_text = re.sub(
                pattern,
                replacement_phrase,
                converted_text,
                flags=re.IGNORECASE
            )
        
        logger.info(f"[{self.node_name}] Successfully converted data tone to '{target_tone}'.")
        return converted_text