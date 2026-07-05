
import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class IntentClassifierNode(BaseNode):
    """
    A Vishustra node that classifies the intent of a given text utterance.
    This node simulates intent classification using a simple keyword-matching
    approach. In a production environment, this would integrate with a
    sophisticated Natural Language Understanding (NLU) model.
    """

    def __init__(self, default_intent: str = "UNKNOWN_INTENT"):
        """
        Initializes the IntentClassifierNode.

        Args:
            default_intent (str): The intent to return if no specific intent
                                  can be classified from the utterance.
        """
        self._default_intent = default_intent
        # A simple keyword-to-intent mapping for demonstration purposes.
        # This would typically be replaced by a loaded NLU model or service client.
        self._intent_map: Dict[str, str] = {
            "schedule meeting": "ScheduleMeeting",
            "book appointment": "ScheduleMeeting",
            "cancel meeting": "CancelMeeting",
            "reschedule meeting": "RescheduleMeeting",
            "weather forecast": "GetWeather",
            "play music": "PlayMusic",
            "stop music": "StopMusic",
            "order food": "OrderFood",
            "restaurant reservation": "OrderFood",
            "tell me a joke": "TellJoke",
            "create reminder": "CreateReminder",
            "set alarm": "SetAlarm",
            "send email": "SendEmail",
            "help me": "GetHelp",
        }
        logger.info(f"IntentClassifierNode initialized with default intent: '{self._default_intent}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to classify its intent.

        This method expects the `data` to be a string representing a user utterance.
        It performs a simple keyword-based matching to identify the intent.

        Args:
            data (Any): The input data, expected to be a string utterance.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. This node
                                       does not modify the context but receives it.

        Returns:
            Dict[str, Any]: A dictionary containing the original utterance, its
                            classified intent, and optionally the matched keyword.
                            Example: {"utterance": "...", "intent": "...", "matched_keyword": "..."}.

        Raises:
            TypeError: If the input `data` is not a string, as this node
                       is designed to process text utterances.
        """
        if not isinstance(data, str):
            error_message = (
                f"{self.node_name} expects input 'data' to be a string utterance, "
                f"but received type {type(data).__name__}. This data cannot be processed."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        utterance = data.strip().lower()
        classified_intent = self._default_intent
        matched_keyword = None

        logger.debug(f"Attempting to classify intent for utterance: '{utterance}'")

        for keyword, intent in self._intent_map.items():
            if keyword in utterance:
                classified_intent = intent
                matched_keyword = keyword
                break  # Found the first matching keyword, assign intent and exit

        if matched_keyword:
            logger.info(
                f"Classified intent for utterance '{utterance}' as '{classified_intent}' "
                f"based on keyword '{matched_keyword}'."
            )
        else:
            logger.info(
                f"No specific intent found for utterance '{utterance}', "
                f"defaulting to '{classified_intent}'."
            )

        # The context can be used for logging or for more complex nodes,
        # but for this specific node, it's primarily passed through.
        _ = context # Suppress unused variable warning if context isn't directly used here.

        result = {
            "utterance": data,  # Retain original casing for the output utterance
            "intent": classified_intent,
            "matched_keyword": matched_keyword,
        }

        logger.debug(f"{self.node_name} processed successfully, result: {result}")
        return result

