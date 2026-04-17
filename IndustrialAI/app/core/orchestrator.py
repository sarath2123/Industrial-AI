import cv2
import yaml
import time


class Orchestrator:
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.agents             = []
        self.telemetry_snapshot = {}

        # Watchdog: trigger safety stop if no frame arrives within this time (seconds)
        self.WATCHDOG_TIMEOUT = 2.0

    def add_agent(self, agent_instance):
        if agent_instance.is_active:
            self.agents.append(agent_instance)
            print(f"Agent '{agent_instance.agent_name}' activated.")

    def start(self):
        cap = cv2.VideoCapture(self.config['system']['camera_source'])
        print(f"Backend: {cap.getBackendName()}")
        print(f"Is opened: {cap.isOpened()}")

        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        print("SentinelAI Platform Started. Press 'q' to quit, 'r' to reset.")

        prev_time       = time.time()
        last_frame_time = time.time()
        watchdog_active = False

        while cap.isOpened():
            now = time.time()

            # --- Watchdog ---
            if now - last_frame_time > self.WATCHDOG_TIMEOUT:
                if not watchdog_active:
                    print("[WATCHDOG] Frame timeout — triggering safety latch on all agents.")
                    for agent in self.agents:
                        if hasattr(agent, 'is_latched'):
                            agent.is_latched = True
                    watchdog_active = True

            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame.")
                break

            last_frame_time = time.time()
            watchdog_active = False

            # Flip for webcam mirror (so zone coordinates match what the user sees)
            frame = cv2.flip(frame, 1)

            # --- Process all agents ---
            for agent in self.agents:
                data = agent.process(frame)
                self.telemetry_snapshot[agent.agent_name] = data
                agent.draw(frame, data)

            # --- FPS overlay ---
            fps       = 1.0 / (time.time() - prev_time)
            prev_time = time.time()
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 90, 25),
                        0, 0.6, (200, 200, 200), 1)

            # --- Display ---
            cv2.imshow("SentinelAI Platform", frame)

            # --- Keyboard ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting agents...")
                for agent in self.agents:
                    agent.reset()
            elif key == 27:   # ESC
                break

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Platform shut down.")