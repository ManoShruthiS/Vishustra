import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A node responsible for modifying the stylistic tone of textual data.
    
    This node expects the input data to be a string and looks for a 
    'target_tone' key within the context to determine the transformation logic.
    """

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "ToneConverterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input text to match a specified target tone.
        
        Args:
            data (Any): The input text string to be transformed.
            context (Dict[str, Any]): Metadata containing 'target_tone'.
            
        Returns:
            str: The processed text with the applied tone shift.
            
        Raises:
            ValueError: If the input data is not a string.
            KeyError: If processing fails due to missing context parameters.
        """
        if not isinstance(data, str):
            logger.error("ToneConverterNode received non-string data type: %s", type(data).__name__)
            raise ValueError(f"Invalid data type: expected str, got {type(data).__name__}")

        target_tone = context.get("target_tone", "neutral").lower()
        
        logger.info("Initiating tone conversion to '%s'.", target_tone)

        try:
            # Simulation of linguistic transformation logic.
            # In a full Vishustra implementation, this would likely involve 
            # a prompt template or a call to a specific fine-tuned model.
            
            transformation_map = {
                "professional": "Please be advised: ",
                "casual": "Hey, just so you know: ",
                "urgent": "IMMEDIATE ACTION REQUIRED: ",
                "neutral": ""
            }

            prefix = transformation_map.get(target_tone, f"[{target_tone.upper()}] ")
            processed_text = f"{prefix}{data}"

            logger.debug("Successfully transformed text to %s tone.", target_tone)
            return processed_text

        except Exception as e:
            logger.error("An error occurred during tone conversion: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to process node '{self.node_name}': {str(e)}") from e

def _get_node_instance() -> ToneConverterNode:
    """Helper for node registry initialization."""
    return ToneConverterNode()

```python
# End of file
