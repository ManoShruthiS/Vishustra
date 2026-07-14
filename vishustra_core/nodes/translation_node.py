import logging
from typing import Any, Dict, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslator(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text.

    This node takes a string as input data and translates it to a specified
    target language. The target language can be set during node initialization
    or overridden via the context dictionary during processing.
    """

    def __init__(self, target_language: str, model_id: str = "google-nmt-v2") -> None:
        """
        Initializes the LanguageTranslator node.

        Args:
            target_language (str): The default target language for translation
                                   (e.g., 'es', 'fr', 'de'). This can be
                                   overridden by `context['target_language']`.
                                   Must be a non-empty string.
            model_id (str): Identifier for the underlying translation model
                            to be used. Defaults to "google-nmt-v2".
                            Must be a non-empty string.

        Raises:
            ValueError: If `target_language` or `model_id` is not a valid non-empty string.
        """
        if not isinstance(target_language, str) or not target_language.strip():
            raise ValueError("Target language must be a non-empty string.")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Model ID must be a non-empty string.")

        self._default_target_language = target_language.strip()
        self._model_id = model_id.strip()
        logger.debug(
            f"LanguageTranslator node initialized with default target language "
            f"'{self._default_target_language}' and model '{self._model_id}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by translating it to the target language.

        The input `data` is expected to be a string containing the text to be
        translated. The `context` dictionary can optionally provide an
        override for the target language using the key 'target_language'.

        Args:
            data (Any): The input data to process, expected to be a string.
            context (Dict[str, Any]): A dictionary containing context-specific
                                      information. Can include 'target_language'
                                      to override the default.

        Returns:
            Any: The translated string.

        Raises:
            ValueError: If the input data is not a string, or if no target
                        language can be determined.
            RuntimeError: If a simulated translation error occurs (e.g., due to
                          an external service issue).
        """
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected str, got {type(data).__name__}.")
            raise ValueError(
                f"[{self.node_name}] Input data must be a string for translation. "
                f"Received type: {type(data).__name__}"
            )

        text_to_translate = data.strip()
        if not text_to_translate:
            logger.warning(f"[{self.node_name}] Received empty string for translation. Returning as is.")
            return ""

        # Determine the effective target language, prioritizing context over default
        effective_target_language = context.get('target_language', self._default_target_language)
        if not isinstance(effective_target_language, str) or not effective_target_language.strip():
            logger.error(
                f"[{self.node_name}] No valid target language specified. "
                f"Default: '{self._default_target_language}', Context override: '{context.get('target_language')}'."
            )
            raise ValueError(
                f"[{self.node_name}] A valid target language (non-empty string) must be "
                f"provided via initialization or context for translation."
            )
        effective_target_language = effective_target_language.strip() # Ensure stripped if from context

        logger.info(
            f"[{self.node_name}] Attempting to translate text (first 50 chars: '{text_to_translate[:50]}...') "
            f"to '{effective_target_language}' using model '{self._model_id}'."
        )

        try:
            # Simulate a translation operation.
            # In a production environment, this would involve an actual API call
            # to a translation service (e.g., Google Translate, DeepL, custom ML model).
            translated_text = (
                f"{text_to_translate} "
                f" (translated to {effective_target_language} by {self._model_id})"
            )

            # Introduce a mechanism to simulate an external service failure for robustness testing
            if context.get("simulate_translation_error", False):
                raise RuntimeError("Simulated external translation service outage or API error.")

            logger.info(
                f"[{self.node_name}] Successfully translated text to "
                f"'{effective_target_language}'. Result (first 50 chars: '{translated_text[:50]}...')"
            )
            return translated_text
        except Exception as e:
            logger.error(
                f"[{self.node_name}] An error occurred during translation "
                f"to '{effective_target_language}' using model '{self._model_id}': {e}",
                exc_info=True # Log the full traceback for debugging
            )
            # Re-raise the exception, possibly wrapped, to ensure upstream orchestration
            # can handle the failure appropriately.
            raise RuntimeError(f"[{self.node_name}] Failed to translate text due to an internal or external issue: {e}") from e