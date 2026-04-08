import time
import logging
from typing import Callable, List, Dict, Any
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("message_bus")

class MessageBus:
    """
    Synchronous Pub/Sub system for communication between agents and components.
    """
    
    def __init__(self):
        # topic -> list of callback functions
        self.subscriptions: Dict[str, List[Callable]] = {
            "sign_detected": [],
            "complaint_ready": [],
            "intent_classified": [],
            "action_decided": [],
            "tool_result_ready": [],
            "response_ready": [],
            "emergency_triggered": [],
            "agent_status": []
        }
        self.message_history = deque(maxlen=50)

    def subscribe(self, topic: str, callback: Callable):
        """Subscribe a callback to a specific topic."""
        if topic in self.subscriptions:
            if callback not in self.subscriptions[topic]:
                self.subscriptions[topic].append(callback)
                logger.debug(f"Subscribed callback to topic: {topic}")
        else:
            logger.warning(f"Attempted to subscribe to unknown topic: {topic}")

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe a callback from a specific topic."""
        if topic in self.subscriptions:
            if callback in self.subscriptions[topic]:
                self.subscriptions[topic].remove(callback)
                logger.debug(f"Unsubscribed callback from topic: {topic}")

    def publish(self, topic: str, message: Dict[str, Any]):
        """Publish a message to all subscribers of a topic."""
        if topic not in self.subscriptions:
            logger.warning(f"Attempted to publish to unknown topic: {topic}")
            return

        timestamp = time.time()
        time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))
        
        # Store in history
        msg_entry = {
            "topic": topic,
            "timestamp": timestamp,
            "message": message
        }
        self.message_history.append(msg_entry)
        
        # Log the message with first 100 chars
        summary = str(message)[:100]
        logger.info(f"[{time_str}] Topic: {topic} | Msg: {summary}")

        # Execute callbacks
        for callback in self.subscriptions[topic]:
            try:
                callback(message)
            except Exception as e:
                logger.error(f"Error in callback for topic {topic}: {e}")
