import logging
from typing import Any, Dict

# Importing BaseNode as per Vishustra's modular architecture
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node responsible for translating text data.

    This node expects the input `data` to be a string representing the text
    to be translated. The `context` dictionary must contain a 'target_language'
    key, specifying the desired language for translation using its ISO 639-1 code
    (e.g., 'es' for Spanish, 'fr' for French, 'de' for German).

    For demonstration purposes, translation is simulated via a basic word-by-word
    dictionary lookup. In a production environment, this node would typically
    integrate with robust external translation APIs (e.g., Google Translate,
    DeepL) or advanced internal NLP models for comprehensive linguistic processing.
    """

    # A static dictionary simulating a basic translation engine.
    # This serves as a placeholder for a real-world translation service integration.
    _DUMMY_TRANSLATIONS = {
        "hello": {"es": "hola", "fr": "bonjour", "de": "hallo"},
        "world": {"es": "mundo", "fr": "monde", "de": "welt"},
        "vishustra": {"es": "vishustra", "fr": "vishustra", "de": "vishustra"}, # Proper noun example
        "orchestration": {"es": "orquestación", "fr": "orchestration", "de": "orchestrierung"},
        "framework": {"es": "marco", "fr": "cadre", "de": "framework"},
        "python": {"es": "python", "fr": "python", "de": "python"},
        "engineering": {"es": "ingeniería", "fr": "ingénierie", "de": "ingenieurwesen"},
        "backend": {"es": "backend", "fr": "backend", "de": "backend"},
        "node": {"es": "nodo", "fr": "nœud", "de": "knoten"},
        "data": {"es": "datos", "fr": "données", "de": "daten"},
        "processing": {"es": "procesamiento", "fr": "traitement", "de": "verarbeitung"},
        "language": {"es": "idioma", "fr": "langue", "de": "sprache"},
        "translator": {"es": "traductor", "fr": "traducteur", "de": "übersetzer"},
    }
    
    # Explicitly supported languages by our dummy translator.
    # Requests for other languages will proceed with limited (word-not-found) translation.
    _SUPPORTED_LANGUAGES = ["es", "fr", "de"]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name for this processing node."""
        return "LanguageTranslator"

    def _translate_word(self, word: str, target_lang: str) -> str:
        """
        Helper method to simulate the translation of a single word.
        It attempts to preserve the original capitalization of the first letter.
        """
        word_lower = word.lower()
        if word_lower in self._DUMMY_TRANSLATIONS and target_lang in self._DUMMY_TRANSLATIONS[word_lower]:
            translated_word = self._DUMMY_TRANSLATIONS[word_lower][target_lang]
            # Attempt to preserve capitalization of the initial letter
            if word and word[0].isupper() and translated_word:
                return translated_word.capitalize()
            return translated_word
        return word # If no translation is found, return the original word

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input `data` (text string) to the `target_language`
        specified within the `context` dictionary.

        Args:
            data: The string containing the text to be translated.
            context: A dictionary expected to contain:
                     - 'target_language' (str): The ISO 639-1 code of the
                       desired target language (e.g., 'es', 'fr', 'de').

        Returns:
            A string containing the translated text.

        Raises:
            TypeError: If `data` is not a string, or if 'target_language' in
                       `context` is not a string.
            ValueError: If 'target_language' is missing from the `context`.
        """
        logger.debug(f"[{self.node_name}] Initiating translation process for data of type: {type(data).__name__}.")

        # --- Input Data Validation ---
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"received '{type(data).__name__}'. Aborting translation."
            )
            raise TypeError(
                f"Input data for {self.node_name} must be a string, "
                f"received {type(data).__name__}."
            )

        # --- Context Validation for Target Language ---
        target_language_raw = context.get("target_language")
        
        if target_language_raw is None:
            logger.error(
                f"[{self.node_name}] 'target_language' key is missing from the provided context. "
                "Unable to determine target language for translation."
            )
            raise ValueError(f"Missing 'target_language' in context for {self.node_name}.")
        
        if not isinstance(target_language_raw, str):
            logger.error(
                f"[{self.node_name}] Invalid type for 'target_language'. Expected 'str', "
                f"received '{type(target_language_raw).__name__}'. Aborting translation."
            )
            raise TypeError(
                f"'target_language' for {self.node_name} must be a string, "
                f"received {type(target_language_raw).__name__}."
            )

        target_language = target_language_raw.lower() # Standardize to lowercase for internal consistency

        if target_language not in self._SUPPORTED_LANGUAGES:
            logger.warning(
                f"[{self.node_name}] Target language '{target_language}' is not fully supported "
                "by the internal dummy translator. Proceeding with best-effort word-by-word "
                "translation, but results may be limited or incomplete."
            )
        else:
            logger.debug(f"[{self.node_name}] Translating content to language: '{target_language}'.")

        # --- Translation Logic (Simulated Word-by-Word) ---
        translated_words = []
        # A simple whitespace-based split. A real-world NLP scenario would involve
        # more sophisticated tokenization to handle punctuation, compound words, etc.
        words = data.split() 
        
        for word in words:
            translated_words.append(self._translate_word(word, target_language))
        
        translated_text = " ".join(translated_words)
        logger.info(
            f"[{self.node_name}] Successfully translated text (first 100 characters): "
            f"'{translated_text[:100]}{'...' if len(translated_text) > 100 else ''}'"
        )
        
        return translated_text