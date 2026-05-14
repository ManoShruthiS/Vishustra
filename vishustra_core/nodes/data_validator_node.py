import logging
from typing import Any, Dict, List, Type, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A robust validation node designed to ensure data integrity within the LLM orchestration pipeline.
    It verifies input data against required schemas, types, and custom constraints defined in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the canonical name for this node type."""
        return "Data Validator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the incoming data based on the configuration provided in the context.

        Args:
            data: The payload to be validated.
            context: A dictionary containing validation rules:
                - 'required_fields' (List[str]): Keys that must exist if data is a dict.
                - 'expected_type' (Type): The python type the data should conform to.
                - 'allow_empty' (bool): Whether to allow empty strings/collections. Defaults to True.

        Returns:
            Any: The original data if validation passes.

        Raises:
            ValueError: If data is null or fails content constraints.
            TypeError: If data does not match the expected type.
            KeyError: If required fields are missing from a dictionary payload.
        """
        logger.debug(f"Executing {self.node_name} logic.")

        if data is None:
            logger.error("Validation failed: Received NoneType data.")
            raise ValueError("Input data to DataValidatorNode cannot be None.")

        # 1. Type Verification
        expected_type: Optional[Type] = context.get("expected_type")
        if expected_type and not isinstance(data, expected_type):
            msg = f"Type mismatch. Expected {expected_type.__name__}, got {type(data).__name__}."
            logger.error(msg)
            raise TypeError(msg)

        # 2. Emptiness Check
        allow_empty: bool = context.get("allow_empty", True)
        if not allow_empty:
            if hasattr(data, "__len__") and len(data) == 0:
                logger.error("Validation failed: Data object is empty while allow_empty=False.")
                raise ValueError("Data cannot be empty.")

        # 3. Structural Validation for Mappings
        if isinstance(data, dict):
            required_fields: List[str] = context.get("required_fields", [])
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                msg = f"Validation failed: Missing required fields: {', '.join(missing_fields)}"
                logger.error(msg)
                raise KeyError(msg)
            
            logger.debug("Dictionary structure validated successfully.")

        logger.info(f"Node '{self.node_name}' successfully validated data payload.")
        return data

    def _handle_error(self, error: Exception) -> None:
        """
        Internal helper for consistent error logging across the node's lifecycle.
        """
        logger.exception(f"Critical error in {self.node_name}: {str(error)}")