import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A transformation node responsible for adjusting the linguistic tone of the input text.
    
    This node expects a string input and a target tone specification within the context.
    If no tone is specified, it defaults to a 'professional' tone.
    """

    def __init__(self, default_tone: str = "professional"):
        self._default_tone = default_tone

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for the ToneConverterNode."""
        return "ToneConverterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data to simulate a tone transformation.
        
        Args:
            data (Any): The input text to be transformed. Expected to be a string.
            context (Dict[str, Any]): The execution context, potentially containing 'target_tone'.
            
        Returns:
            str: The processed text with the requested tone applied.
            
        Raises:
            ValueError: If the input data is not a string or is empty.
        """
        if not isinstance(data, str) or not data.strip():
            logger.error(f"[{self.node_name}] Invalid input data type. Expected non-empty string, got {type(data).__name__}.")
            raise ValueError("ToneConverterNode requires a non-empty string as input data.")

        target_tone = context.get("target_tone", self._default_tone)
        logger.info(f"[{self.node_name}] Transforming text to '{target_tone}' tone.")

        try:
            # Simulation of the transformation logic.
            transformed_text = self._apply_tone_logic(data, target_tone)
            
            logger.debug(f"[{self.node_name}] Successfully processed text of length {len(data)}.")
            return transformed_text

        except Exception as e:
            logger.exception(f"[{self.node_name}] Unexpected error during tone conversion: {str(e)}")
            raise

    def _apply_tone_logic(self, text: str, tone: str) -> str:
        """
        Internal logic to wrap the text with tone-specific metadata or 
        simulated transformation markers.
        """
        # Placeholder for processing logic
        return f"[Tone: {tone}] {text}"

    def __repr__(self) -> str:
        return f"<{self.node_name}(default_tone='{self._default_tone}')>"