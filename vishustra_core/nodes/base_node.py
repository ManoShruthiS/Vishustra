from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseNode(ABC):
    """
    Base class for all Vishustra processing nodes.
    Each node must implement the process method.
    """
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data and returns the result.
        """
        pass
        
    @property
    @abstractmethod
    def node_name(self) -> str:
        """Returns the name of the node."""
        pass
