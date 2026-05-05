import time
import logging
from vishustra_core.engine import VishustraEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_dashboard() -> None:
    logger.info("Initializing Vishustra AI Orchestrator Dashboard")
    
    engine = VishustraEngine()
    active_nodes = engine.get_active_nodes()
    
    logger.info("System Health: ONLINE")
    logger.info(f"Total Active Modules/Nodes: {len(active_nodes)}")
    
    if active_nodes:
        logger.info("Loaded Modules:")
        for idx, name in enumerate(active_nodes[-10:], 1):
            logger.info(f"  {idx}. {name}")
        if len(active_nodes) > 10:
            logger.info(f"  ... and {len(active_nodes) - 10} more.")
            
        engine.run_simulation("Test input packet from main dashboard.")
    else:
        logger.warning("No active nodes found. Please configure the node pipeline to continue.")

if __name__ == "__main__":
    run_dashboard()
