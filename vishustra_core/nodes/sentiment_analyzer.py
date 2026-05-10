import logging
from typing import Any, Dict, Union, List
from vishustra_core.nodes.base_node import BaseNode

# Configure logger for the module
logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A processing node that evaluates the sentiment of textual input.
    
    This node extracts emotional polarity from the provided data, categorizing it
    as positive, negative, or neutral, along with a confidence score. It is
    designed to be used within the Vishustra orchestration pipeline to facilitate
    conditional branching based on user intent or feedback.
    """

    @property
    def node_name(self) -> str:
        """Returns the unique identifier for this node type."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to determine sentiment polarity.

        Args:
            data: The input to analyze. Expected to be a string or a dict containing a 'text' key.
            context: The global orchestration context, providing shared state across nodes.

        Returns:
            A dictionary containing the analysis results:
            - 'label': (str) The detected sentiment category.
            - 'score': (float) Normalized polarity score between -1.0 and 1.0.
            - 'processed_at': (str) Metadata timestamp or state identifier.

        Raises:
            ValueError: If the input data format is unsupported or missing text content.
            RuntimeError: If an unexpected error occurs during data transformation.
        """
        try:
            # Input normalization
            text = self._extract_text(data)
            
            if not text or not text.strip():
                logger.warning(f"[{self.node_name}] Received empty text input. Defaulting to neutral.")
                return self._build_result("neutral", 0.0)

            logger.debug(f"[{self.node_name}] Analyzing sentiment for text payload.")

            # Simulated sentiment heuristic (Interface for future LLM or VADER integration)
            score = self._calculate_sentiment_score(text)
            
            # Categorization logic
            if score >= 0.05:
                label = "positive"
            elif score <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            logger.info(f"[{self.node_name}] Analysis complete. Result: {label} (Score: {score})")
            
            return self._build_result(label, score)

        except Exception as e:
            logger.error(f"[{self.node_name}] Failed to process data: {str(e)}", exc_info=True)
            raise RuntimeError(f"SentimentAnalyzer execution failed: {e}") from e

    def _extract_text(self, data: Any) -> str:
        """Helper to safely extract string data from various input formats."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            return data.get("text", data.get("content", ""))
        return str(data)

    def _calculate_sentiment_score(self, text: str) -> float:
        """
        Simulates sentiment calculation logic using keyword weighting.
        In a production environment, this would wrap a transformer model or a specialized library.
        """
        content = text.lower()
        
        positive_lexicon = {"excellent", "great", "positive", "success", "helpful", "perfect", "good", "love"}
        negative_lexicon = {"bad", "error", "failure", "fail", "broken", "terrible", "issue", "negative", "hate"}
        
        words = content.split()
        if not words:
            return 0.0
            
        pos_hits = sum(1 for word in words if word in positive_lexicon)
        neg_hits = sum(1 for word in words if word in negative_lexicon)
        
        # Calculate a simple normalized polarity score
        diff = pos_hits - neg_hits
        total_hits = pos_hits + neg_hits
        
        if total_hits == 0:
            return 0.0
            
        return round(diff / total_hits, 4)

    def _build_result(self, label: str, score: float) -> Dict[str, Any]:
        """Constructs a standardized output schema for the node."""
        return {
            "node": self.node_name,
            "sentiment_data": {
                "label": label,
                "score": score,
                "confidence_threshold_met": abs(score) > 0.2
            }
        }