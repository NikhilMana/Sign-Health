import sys
import asyncio
import numpy as np
import logging
from unittest.mock import MagicMock

# Mock heavy deps
sys.modules['cv2'] = MagicMock()
sys.modules['mediapipe'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()

from core.agent_registry import AgentRegistry
from core.session_manager import SessionManager

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_pipeline")

async def test_pipeline():
    logger.info("Initializing AgentRegistry...")
    registry = AgentRegistry()
    
    # Force demo mode and known sign for stage 2
    registry.perception.demo_mode = True
    
    session_id = SessionManager.create_session("pipeline_test_user")
    logger.info(f"Created test session: {session_id}")
    
    # Mock frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    logger.info("Running pipeline with 20 frames to trigger perception...")
    # Frames 1-19: should return 'buffering'
    for i in range(1, 20):
        res = await registry.run_pipeline(mock_frame, session_id)
        assert res["pipeline_status"] == "buffering", f"Expected buffering at frame {i}, got {res['pipeline_status']}"
        assert "perception_process" in res["timings"]
    
    logger.info("Triggering 20th frame (Perception Inference)...")
    # Frame 20: perception ready, should proceed to context
    # Note: Default demo sign might be 'pain'. If context finds only 'pain', score might be < 0.4.
    res = await registry.run_pipeline(mock_frame, session_id)
    
    status = res["pipeline_status"]
    logger.info(f"Pipeline status at frame 20: {status}")
    
    if status == "collecting_signs":
        logger.info(f"Context completeness score: {res['complaint']['completeness_score']}")
        assert "context_process" in res["timings"]
        assert "context_extract" in res["timings"]
    elif status == "complete":
        logger.info(f"Pipeline completed successfully. Text: {res.get('text_response')}")
        assert "intent_classify" in res["timings"]
        assert "tool_execute" in res["timings"]
    
    logger.info("Test passed successfully!")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
