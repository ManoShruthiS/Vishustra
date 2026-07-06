import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ProfanityFilterNode(BaseNode):
    """
    A Vishustra node that filters profanity from text data.

    This node processes input data, identifying and replacing profane words
    with a configurable censor string. It supports filtering strings,
    lists of strings, and dictionary values that are strings.
    """

    # A simple, illustrative list of profane words. In a production system,
    # this list would typically be much more comprehensive, potentially
    # loaded from a configuration service or an external profanity dictionary.
    _DEFAULT_PROFANITY_LIST = [
        "badword", "anotherbad", "terrible", "offensivephrase", "swearword"
    ]
    _DEFAULT_CENSOR_STRING = "****"

    def __init__(self,
                 profanity_list: Union[List[str], None] = None,
                 censor_string: str = _DEFAULT_CENSOR_STRING):
        """
        Initializes the ProfanityFilterNode.

        Args:
            profanity_list: An optional list of words to consider profane.
                            If None, a default illustrative list is used.
                            Words are converted to lowercase for case-insensitive matching.
            censor_string: The string to replace profane words with.
        """
        self._profanity_list = [word.lower() for word in (profanity_list or self._DEFAULT_PROFANITY_LIST)]
        self._censor_string = censor_string
        logger.debug(
            f"{self.node_name} initialized with {len(self._profanity_list)} "
            f"profane words and censor string '{self._censor_string}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ProfanityFilterNode"

    def _censor_text(self, text: str) -> str:
        """
        Replaces profane words in a given string with the censor string.
        Performs case-insensitive matching using regular expressions to
        ensure robust replacement while preserving the original string's
        non-matching parts.
        """
        censored_text = text
        for word in self._profanity_list:
            # Use re.escape to treat the word as a literal string in the regex pattern
            # and re.IGNORECASE for case-insensitive matching.
            # This handles replacements like "BadWord" -> "****" if "badword" is in the list.
            censored_text = re.sub(re.escape(word), self._censor_string, censored_text, flags=re.IGNORECASE)
        return censored_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, filtering out profanity from string content.

        This method supports:
        -   `str`: Filters the string directly.
        -   `list[str]`: Iterates through the list, filtering each string element.
                         Non-string elements are passed through unchanged.
        -   `dict[str, str]`: Iterates through dictionary values, filtering
                               string values. Non-string values are passed
                               through unchanged.

        For any other data type, a warning is logged, and the data is
        returned without modification.

        Args:
            data: The input data to be processed. Expected to be a string,
                  list of strings, or a dictionary with string values.
            context: A dictionary containing contextual information for the
                     orchestration, which is passed through but not modified
                     by this node.

        Returns:
            The processed data with profanity filtered. If the data type is
            unsupported, the original data is returned.
        """
        logger.debug(f"{self.node_name} received data of type: {type(data)}.")

        if isinstance(data, str):
            return self._censor_text(data)
        elif isinstance(data, list):
            processed_list = []
            for item in data:
                if isinstance(item, str):
                    processed_list.append(self._censor_text(item))
                else:
                    # Pass through non-string items in a list
                    processed_list.append(item)
                    logger.debug(
                        f"{self.node_name} encountered non-string item of type "
                        f"{type(item)} in list. Passing through without modification."
                    )
            return processed_list
        elif isinstance(data, dict):
            processed_dict = {}
            for key, value in data.items():
                if isinstance(value, str):
                    processed_dict[key] = self._censor_text(value)
                else:
                    # Pass through non-string values in a dictionary
                    processed_dict[key] = value
                    logger.debug(
                        f"{self.node_name} encountered non-string value of type "
                        f"{type(value)} for key '{key}'. Passing through without modification."
                    )
            return processed_dict
        else:
            logger.warning(
                f"{self.node_name} received unsupported data type: {type(data)}. "
                "Expected str, list[str], or dict[str, str]. Data will be returned unchanged."
            )
            return data