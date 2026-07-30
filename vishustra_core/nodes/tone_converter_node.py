import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is defined as per project context and available at this path.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to convert the tone of input text data.

    This node takes a string as input data and, based on a specified
    'target_tone' in the context, simulates a transformation of the
    text's style. This is a foundational step for dynamic content generation
    and audience-specific communication within Vishustra.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "ToneConverterNode"

    def _apply_tone_transformation(self, text: str, target_tone: str) -> str:
        """
        Applies a simulated tone transformation to the given text.

        This is a placeholder for actual LLM-driven tone conversion.
        For demonstration, it uses simple string replacements based on common
        characteristics of different tones.
        """
        transformed_text = text

        if target_tone.lower() == "professional":
            transformed_text = transformed_text.replace("hey", "Dear Sir/Madam,")
            transformed_text = transformed_text.replace("hi", "Greetings,")
            transformed_text = transformed_text.replace("lol", "with regards,")
            transformed_text = transformed_text.replace("?", ".")
            if not transformed_text.strip().endswith(('.', '!', '?')):
                transformed_text += "."
            # Simulate more formal phrasing
            transformed_text = transformed_text.replace("gonna", "going to")
            transformed_text = transformed_text.replace("wanna", "want to")
            transformed_text = transformed_text.replace("can't", "cannot")
            transformed_text = transformed_text.replace("it's", "it is")
            transformed_text = transformed_text.replace("i'm", "I am")
            transformed_text = transformed_text.replace("you're", "you are")
            transformed_text = transformed_text.replace("we're", "we are")

        elif target_tone.lower() == "casual":
            transformed_text = transformed_text.replace("Dear Sir/Madam,", "Hey there!")
            transformed_text = transformed_text.replace("Greetings,", "Hi!")
            transformed_text = transformed_text.replace("with regards,", "lol")
            transformed_text = transformed_text.replace(".", "...")
            if not transformed_text.strip().endswith(('.', '!', '?')):
                transformed_text += " :)"
            # Simulate informal phrasing
            transformed_text = transformed_text.replace("going to", "gonna")
            transformed_text = transformed_text.replace("want to", "wanna")
            transformed_text = transformed_text.replace("cannot", "can't")
            transformed_text = transformed_text.replace("it is", "it's")
            transformed_text = transformed_text.replace("I am", "i'm")
            transformed_text = transformed_text.replace("you are", "you're")
            transformed_text = transformed_text.replace("we are", "we're")


        elif target_tone.lower() == "friendly":
            # A lighter, more approachable version of professional
            transformed_text = transformed_text.replace("Dear Sir/Madam,", "Hello!")
            transformed_text = transformed_text.replace("Greetings,", "Hi there!")
            transformed_text = transformed_text.replace("with regards,", "Best regards,")
            # Ensure a positive ending if not already present
            if not transformed_text.strip().endswith(('.', '!', '?')):
                transformed_text += "!"

        elif target_tone.lower() == "concise":
            # Simulate removing filler words or shortening phrases
            transformed_text = transformed_text.replace("in order to", "to")
            transformed_text = transformed_text.replace("due to the fact that", "because")
            transformed_text = transformed_text.replace("at this point in time", "now")
            transformed_text = transformed_text.replace("it is important to note that", "")
            # Remove redundant punctuation
            transformed_text = transformed_text.replace("...", ".")
            transformed_text = transformed_text.replace("  ", " ").strip() # Clean up extra spaces

        # Fallback for unsupported tones, return original or default
        else:
            logger.warning(
                f"Unsupported target tone '{target_tone}' provided. "
                "Returning original data without transformation."
            )

        return transformed_text

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data to convert its tone based on the 'target_tone'
        specified in the context.

        Args:
            data (Any): The input data, expected to be a string (e.g., a message, paragraph).
            context (Dict[str, Any]): A dictionary containing contextual information.
                                       Must include 'target_tone' (str).

        Returns:
            str: The processed text with the converted tone.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing or not a string in the context.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string, "
                f"but got {type(data).__name__}."
            )

        target_tone: Optional[str] = context.get("target_tone")

        if not isinstance(target_tone, str) or not target_tone:
            logger.error(
                f"[{self.node_name}] 'target_tone' not found or invalid in context. "
                "Expected a non-empty string."
            )
            raise ValueError(
                f"'target_tone' must be provided as a non-empty string in the context "
                f"for '{self.node_name}'."
            )

        logger.info(
            f"[{self.node_name}] Converting tone of input data to '{target_tone}'."
        )

        converted_data = self._apply_tone_transformation(data, target_tone)

        logger.info(
            f"[{self.node_name}] Tone conversion complete for target tone '{target_tone}'."
        )
        return converted_data

# Example Usage (for testing purposes, not part of the delivered file content)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    node = ToneConverterNode()

    # Test cases
    test_data_1 = "Hey, I'm gonna be late. Can't wait to see you lol."
    test_data_2 = "It is important to note that we need to proceed in order to achieve the goal."
    test_data_3 = "Greetings, we're currently reviewing the proposal. It's a great initiative."

    # Test professional tone
    try:
        context_pro = {"target_tone": "professional"}
        result_pro = node.process(test_data_1, context_pro)
        logger.info(f"\nOriginal (1): {test_data_1}\nProfessional: {result_pro}")
        # Expected: Dear Sir/Madam, I am going to be late. Cannot wait to see you with regards.
    except Exception as e:
        logger.error(f"Error in professional conversion: {e}")

    # Test casual tone
    try:
        context_casual = {"target_tone": "casual"}
        result_casual = node.process(test_data_1, context_casual)
        logger.info(f"\nOriginal (1): {test_data_1}\nCasual: {result_casual}")
        # Expected: Hey there! i'm gonna be late... can't wait to see you lol :)
    except Exception as e:
        logger.error(f"Error in casual conversion: {e}")

    # Test friendly tone
    try:
        context_friendly = {"target_tone": "friendly"}
        result_friendly = node.process(test_data_3, context_friendly)
        logger.info(f"\nOriginal (3): {test_data_3}\nFriendly: {result_friendly}")
        # Expected: Hi there! we're currently reviewing the proposal. It's a great initiative!
    except Exception as e:
        logger.error(f"Error in friendly conversion: {e}")

    # Test concise tone
    try:
        context_concise = {"target_tone": "concise"}
        result_concise = node.process(test_data_2, context_concise)
        logger.info(f"\nOriginal (2): {test_data_2}\nConcise: {result_concise}")
        # Expected: we need to proceed to achieve the goal.
    except Exception as e:
        logger.error(f"Error in concise conversion: {e}")

    # Test unsupported tone
    try:
        context_unsupported = {"target_tone": "sarcastic"}
        result_unsupported = node.process(test_data_1, context_unsupported)
        logger.info(f"\nOriginal (1): {test_data_1}\nUnsupported: {result_unsupported}")
        # Expected: Warning logged, original data returned
    except Exception as e:
        logger.error(f"Error in unsupported tone conversion: {e}")

    # Test error handling: invalid data type
    try:
        node.process(123, {"target_tone": "professional"})
    except TypeError as e:
        logger.info(f"\nCaught expected error: {e}")

    # Test error handling: missing target_tone
    try:
        node.process("Hello world.", {})
    except ValueError as e:
        logger.info(f"\nCaught expected error: {e}")

    # Test error handling: invalid target_tone type
    try:
        node.process("Hello world.", {"target_tone": 123})
    except ValueError as e:
        logger.info(f"\nCaught expected error: {e}")<ctrl63>