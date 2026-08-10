import logging
from typing import Any, Dict

# Assuming vishustra_core is installed and discoverable in the Python environment.
# This import path is critical for integration into the Vishustra framework.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node designed to convert the tone of input text.

    This node accepts a string as input and, guided by the 'target_tone'
    parameter within the context, transforms the text's emotional or
    stylistic tone. Common target tones might include 'formal', 'casual',
    'enthusiastic', or 'neutral'.

    In a fully operational Vishustra deployment, this node would typically
    orchestrate interactions with a large language model (LLM) or a specialized
    natural language processing (NLP) service to achieve sophisticated tone
    conversion. For this implementation, a rule-based simulation is used
    to demonstrate the node's functionality and interface contract.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name for this processing node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the tone conversion process on the input data.

        The method expects 'data' to be a string representing the text
        to be converted. The 'context' dictionary must contain a 'target_tone'
        key, specifying the desired output tone.

        Args:
            data (Any): The input data, expected to be a string (the text content).
            context (Dict[str, Any]): A dictionary providing context for the
                                      processing. Must contain 'target_tone' (str).

        Returns:
            Any: The converted text as a string if successful. Returns the
                 original `data` if it's not a string, if `target_tone` is
                 missing/invalid, or if an error occurs during conversion.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Returning original data."
            )
            return data

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.warning(
                f"[{self.node_name}] 'target_tone' not specified or invalid in context. "
                "Unable to perform tone conversion. Returning original data."
            )
            return data

        original_text: str = data
        converted_text: str = original_text
        tone_lower = target_tone.strip().lower()

        logger.info(
            f"[{self.node_name}] Attempting to convert text to '{target_tone}' tone "
            f"for input of length {len(original_text)}."
        )

        try:
            if tone_lower == "formal":
                # Simulate formal tone: Expand contractions, use more reserved language.
                converted_text = (
                    original_text.replace("don't", "do not")
                    .replace("can't", "cannot")
                    .replace("it's", "it is")
                    .replace("isn't", "is not")
                    .replace("I'm", "I am")
                    .replace("you're", "you are")
                    .replace("we're", "we are")
                )
                converted_text = converted_text.replace("very", "exceedingly").replace("nice", "satisfactory")
                converted_text = converted_text.replace("!", ".") # Replace exclamations with periods
                converted_text = converted_text.replace("gonna", "going to").replace("wanna", "want to")

            elif tone_lower == "casual":
                # Simulate casual tone: Introduce contractions, informal vocabulary.
                converted_text = (
                    original_text.replace("do not", "don't")
                    .replace("cannot", "can't")
                    .replace("it is", "it's")
                    .replace("is not", "isn't")
                    .replace("I am", "I'm")
                    .replace("you are", "you're")
                    .replace("we are", "we're")
                )
                converted_text = converted_text.replace("exceedingly", "super").replace("satisfactory", "cool")
                if not any(converted_text.endswith(p) for p in [".", "?", "!"]):
                    converted_text += "." # Ensure some casual punctuation

            elif tone_lower == "enthusiastic":
                # Simulate enthusiastic tone: Add exclamations, positive adjectives.
                converted_text = original_text.replace(".", "! ").replace("!", "! ").replace("?", "?! ")
                if not converted_text.strip().endswith("!"):
                    converted_text += "!"
                converted_text = converted_text.replace("good", "amazing").replace("great", "fantastic").replace("okay", "awesome")
                converted_text = converted_text.replace("amazing", "ABSOLUTELY AMAZING").replace("fantastic", "TRULY FANTASTIC") # Capitalize for emphasis

            elif tone_lower == "neutral":
                # Simulate neutral tone: Remove strong emotions, normalize punctuation and vocabulary.
                converted_text = original_text.replace("!", ".").replace("?", "?") # Normalize to period/question mark
                converted_text = converted_text.replace("amazing", "good").replace("fantastic", "great").replace("awesome", "good")
                converted_text = converted_text.replace("super", "very").replace("cool", "satisfactory")
                # Ensure only a single, standard punctuation mark at the end if present.
                while converted_text.endswith(("!!", "??", "..")):
                    converted_text = converted_text[:-1]

            else:
                logger.warning(
                    f"[{self.node_name}] Unsupported target tone '{target_tone}'. "
                    "No conversion performed. Returning original data."
                )
                return original_text

            logger.info(
                f"[{self.node_name}] Successfully converted text tone to '{target_tone}'. "
                f"Output text length: {len(converted_text)}."
            )
            return converted_text

        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during tone conversion "
                f"for target tone '{target_tone}': {e}. Returning original data."
            )
            return original_text