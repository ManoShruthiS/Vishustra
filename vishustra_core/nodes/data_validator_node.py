import logging
from typing import Any, Dict, List, Optional, Type

# Assuming vishustra_core is a package and nodes.base_node is a module within it
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Custom exception raised when data fails validation rules within Vishustra."""
    pass

class DataValidatorNode(BaseNode):
    """
    A processing node responsible for validating input data against a set of predefined
    or dynamically provided rules.

    This node ensures data integrity and adherence to expected schemas before data
    proceeds to further processing stages in the orchestration flow. Validation rules
    can specify required fields, expected data types, and potentially more complex
    custom validation logic.

    Rules can be set during node initialization or provided/overridden via the
    'validation_rules' key in the context dictionary during processing.
    """

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        field_types: Optional[Dict[str, Type]] = None,
        node_name: str = "DataValidator"
    ):
        """
        Initializes the DataValidatorNode with static validation rules.

        Args:
            required_fields: A list of field names that must be present in the data.
            field_types: A dictionary mapping field names to their expected Python types.
                         E.g., `{"id": int, "name": str}`.
            node_name: The descriptive name of this node instance. Defaults to "DataValidator".
        """
        self._node_name = node_name
        self._required_fields = required_fields if required_fields is not None else []
        self._field_types = field_types if field_types is not None else {}
        logger.debug(
            f"Initialized DataValidatorNode '{self._node_name}' with "
            f"static required_fields={self._required_fields}, "
            f"static field_types={self._field_types}."
        )

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return self._node_name

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against configured rules and any rules
        dynamically supplied in the `context`.

        The `context` dictionary can optionally include a 'validation_rules' key,
        which itself is a dictionary that can contain:
        - 'required_fields': A `List[str]` to augment or override static required fields.
        - 'field_types': A `Dict[str, Type]` to augment or override static field types.

        Args:
            data: The input data to be validated. This node primarily expects a dictionary
                  for field-level validation, but will perform a basic type check otherwise.
            context: A dictionary containing contextual information, potentially including
                     dynamic validation rules under the 'validation_rules' key.

        Returns:
            The original, unmodified data if all validation checks pass successfully.

        Raises:
            DataValidationError: If the data fails any validation rule, indicating
                                 an issue with the input data structure or content.
        """
        current_required_fields = list(self._required_fields)
        current_field_types = dict(self._field_types)

        # Merge or override validation rules from context
        if 'validation_rules' in context and isinstance(context['validation_rules'], Dict):
            context_rules = context['validation_rules']
            if 'required_fields' in context_rules and isinstance(context_rules['required_fields'], List):
                # Extend required fields, avoiding duplicates
                current_required_fields.extend(
                    [f for f in context_rules['required_fields'] if f not in current_required_fields]
                )
                logger.debug(f"Context added/augmented required_fields: {context_rules['required_fields']}.")

            if 'field_types' in context_rules and isinstance(context_rules['field_types'], Dict):
                # Update/override field types from context
                current_field_types.update(context_rules['field_types'])
                logger.debug(f"Context updated/augmented field_types: {context_rules['field_types']}.")

        logger.debug(
            f"Node '{self.node_name}' commencing data validation. "
            f"Effective required_fields={current_required_fields}, "
            f"effective field_types={current_field_types}."
        )

        # 1. Basic data type check for dictionary expected structure
        if not isinstance(data, Dict):
            error_msg = (
                f"Validation failed for node '{self.node_name}': "
                f"Expected input data to be a dictionary for field-level validation, "
                f"but received type {type(data).__name__}."
            )
            logger.error(error_msg)
            raise DataValidationError(error_msg)

        # 2. Check for presence of all required fields
        for field in current_required_fields:
            if field not in data:
                error_msg = (
                    f"Validation failed for node '{self.node_name}': "
                    f"Required field '{field}' is missing from the input data."
                )
                logger.error(error_msg)
                raise DataValidationError(error_msg)
            logger.debug(f"Required field '{field}' is present in data.")

        # 3. Check types of specified fields
        for field, expected_type in current_field_types.items():
            if field in data:  # Only check type if the field exists
                if not isinstance(data[field], expected_type):
                    error_msg = (
                        f"Validation failed for node '{self.node_name}' for field '{field}': "
                        f"Expected type {expected_type.__name__}, but found "
                        f"{type(data[field]).__name__}."
                    )
                    logger.error(error_msg)
                    raise DataValidationError(error_msg)
                logger.debug(f"Field '{field}' has correct type {expected_type.__name__}.")
            # If a field type is specified for a field that is also required
            # but missing, the missing field error would have been raised first.

        logger.info(f"Node '{self.node_name}' successfully validated the input data.")
        return data