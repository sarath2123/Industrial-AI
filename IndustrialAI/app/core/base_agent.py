from abc import ABC, abstractmethod
import datetime

class BaseAgent(ABC):
    def __init__(self, agent_name, config):
        self.agent_name = agent_name
        self.config = config
        self.is_active = config.get("enabled", False)
        self.last_run = None

    @abstractmethod
    def process(self, frame):
        """
        Main logic for the agent.
        :param frame: The current video frame (numpy array).
        :return: A dictionary of telemetry data (counts, risk levels, etc.)
        """
        pass

    @abstractmethod
    def draw(self, frame, telemetry):
        """
        Draws visual overlays specific to this agent.
        :param frame: The current video frame to draw on.
        :param telemetry: The telemetry dict returned by process().
        """
        pass

    def reset(self):
        """
        Optional reset hook. Called by the Orchestrator on 'r' keypress.
        Subclasses should override this to clear their internal state.
        """
        pass

    def log_event(self, event_type, details):
        """Standardized logging for Autonomous Governance."""
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "ts": timestamp,
            "agent": self.agent_name,
            "event": event_type,
            **details
        }
        # This will be picked up by our database handler later
        print(f"[{self.agent_name}] {event_type}: {details}")
        return log_entry