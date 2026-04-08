import asyncio
import numpy as np
import logging
from core.agent_registry import AgentRegistry
from core.session_manager import SessionManager

# Configure logging to see the pipeline stages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verification")

async def test_full_pipeline():
    logger.info("--- Starting Full Pipeline Verification Test ---")
    
    # 1. Initialize Registry and Session
    registry = AgentRegistry()
    session_id = SessionManager.create_session(patient_id="patient_123")
    logger.info(f"Created session: {session_id}")
    
    # 2. Mock a frame (zeros)
    # We will run the pipeline 20 times to fill the perception buffer
    # Since perception is in DEMO_MODE, it will return a sign after 20 frames.
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    logger.info("Feeding 19 frames to buffer...")
    for i in range(19):
        res = await registry.run_pipeline(mock_frame, session_id)
        if res:
            logger.error(f"Unexpected result at frame {i+1}")
            
    logger.info("Feeding the 20th frame to trigger prediction...")
    final_res = await registry.run_pipeline(mock_frame, session_id)
    
    if final_res:
        logger.info("--- PIPELINE SUCCESS ---")
        logger.info(f"Final Response: {final_res['text_response']}")
        logger.info(f"Patient Card: {final_res['patient_card']}")
    else:
        logger.error("Pipeline failed to return a response after 20 frames.")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
