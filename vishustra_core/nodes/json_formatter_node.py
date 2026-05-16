import json
import logging
from typing import Any, Dict
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class JSONFormatterNode(BaseNode):
    """
    A utility node within the Vishustra framework designed to serialize 
    Python objects into standardized, human-readable JSON strings.
    
    This node handles indentation and key sorting based on the provided 
    execution context, ensuring downstream nodes or external consumers 
    receive consistent data formats.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique identifier for this node type.
        """
        return "JSONFormatterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Transforms the input data into a prettified JSON string.

        Args:
            data (Any): The data structure (dict, list, etc.) to be serialized.
            context (Dict[str, Any]): Execution context. 
                Supported keys:
                - 'indent': Integer for JSON indentation (default: 4).
                - 'sort_keys': Boolean to sort dictionary keys (default: True).

        Returns:
            str: A formatted JSON string representation of the input data.

        Raises:
            ValueError: If the input data is not JSON serializable.
        """
        indent = context.get("json_indent", 4)
        sort_keys = context.get("json_sort_keys", True)

        try:
            logger.info(f"Executing {self.node_name}: Serializing input data.")
            
            # Perform serialization
            formatted_json = json.dumps(
                data, 
                indent=indent, 
                sort_keys=sort_keys,
                ensure_ascii=False
            )
            
            logger.debug(f"{self.node_name} successfully processed {len(formatted_json)} characters.")
            return formatted_json

        except (TypeError, ValueError) as e:
            logger.error(
                f"Serialization failed in {self.node_name}. "
                f"Data type '{type(data).__name__}' may not be serializable. Error: {str(e)}"
            )
            raise ValueError(
                f"Node '{self.node_name}' failed to process input: {str(e)}"
            ) from e

        except Exception as e:
            logger.critical(f"Unexpected error in {self.node_name}: {str(e)}")
            raise
