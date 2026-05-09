import logging
from typing import Any, Dict, Optional
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SentimentAnalyzerNode(BaseNode):
    """
    A modular node within the Vishustra framework responsible for performing 
    sentiment analysis on incoming textual data. It provides polarity 
    classification and confidence scoring.
    """

    @property
    def node_name(self) -> str:
        """Returns the identifier for this node type."""
        return "SentimentAnalyzer"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the sentiment of the input data.
        
        Args:
            data: The input to analyze. Can be a raw string or a dictionary 
                  containing a 'text' key.
            context: Shared state and configuration for the current pipeline execution.

        Returns:
            A dictionary containing the sentiment label, score, and source text.

        Raises:
            TypeError: If the input data format is unsupported.
            Exception: For unexpected processing failures.
        """
        logger.info(f"Execution started for node: {self.node_name}")
        
        try:
            text = self._extract_text(data)
            
            # Simulate sentiment heuristic. In a production environment, this would
            # likely interface with a local transformer model or a specialized NLP service.
            analysis_result = self._analyze_polarity(text)
            
            output = {
                "sentiment": analysis_result["label"],
                "confidence_score": analysis_result["score"],
                "source_text_snippet": text[:50] + "..." if len(text) > 50 else text,
                "node_metadata": {
                    "engine": "Vishustra_Heuristic_v1",
                    "char_count": len(text)
                }
            }
            
            logger.debug(f"Successfully processed sentiment: {output['sentiment']}")
            return output

        except (TypeError, ValueError) as ve:
            logger.error(f"Data validation error in {self.node_name}: {str(ve)}")
            raise
        except Exception as e:
            logger.exception(f"Critical failure in {self.node_name} processing: {str(e)}")
            raise

    def _extract_text(self, data: Any) -> str:
        """Helper to normalize input into a string."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict) and "text" in data:
            return str(data["text"])
        
        raise TypeError(
            f"Invalid input type '{type(data).__name__}'. "
            f"{self.node_name} requires a string or a dict with a 'text' key."
        )

    def _analyze_polarity(self, text: str) -> Dict[str, Any]:
        """
        Internal logic for determining text polarity.
        This provides a baseline implementation for the orchestration framework.
        """
        content = text.lower()
        
        # Reference keywords for basic polarity detection
        lexicon = {
            "positive": {"excellent", "great", "optimal", "success", "efficient", "robust", "happy"},
            "negative": {"error", "fail", "slow", "poor", "unstable", "bad", "terrible"}
        }
        
        pos_hits = sum(1 for word in lexicon["positive"] if word in content)
        neg_hits = sum(1 for word in lexicon["negative"] if word in content)
        
        if pos_hits > neg_hits:
            return {"label": "POSITIVE", "score": min(0.5 + (pos_hits * 0.1), 0.99)}
        elif neg_hits > pos_hits:
            return {"label": "NEGATIVE", "score": min(0.5 + (neg_hits * 0.1), 0.99)}
        
        return {"label": "NEUTRAL", "score": 0.50}

    def __repr__(self) -> str:
        return f"<SentimentAnalyzerNode(name='{self.node_name}')>"

# End of sentiment_analyzer_node.py