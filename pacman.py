import sys
import cv2
import mediapipe as mp
import pygame
from model.board_definition import BoardDefinition
from model.level_config import LevelConfig
from settings import *
from levels.level_content_initializer import LevelContentInitializer
from model.direction import Direction

# -------------------------
# Configuration: Hand Tracking Settings
# -------------------------
CAMERA_INDEX = 0

# Sensitivity Bounds (From your Code 2)
LEFT_BOUND = 0.40
RIGHT_BOUND = 0.60
UP_BOUND = 0.40
DOWN_BOUND = 0.60

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(RESOLUTION)
        self.timer = pygame.time.Clock()
        self.game_engine = self.init()
        self.game_start_sfx = pygame.mixer.Sound('media/game_start.wav')

        # -------------------------
        # Setup Hand Tracking
        # -------------------------
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        print("Pac-Man Mode Started with Hand Tracking")

    def init(self):
        board = BOARD.copy()
        board_definition = BoardDefinition(board)
        level_1 = LevelConfig(wall_color='blue', gate_color='white',
                              board_definition=board_definition, power_up_limit=POWER_UP_LIMIT)
        level_init = LevelContentInitializer(level_1, self.screen)
        return level_init.init_game_engine()

    def process_hand_input(self):
        """
        Reads the camera frame, detects the hand, and sets the 
        game_engine.direction_command directly.
        """
        if not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip and process frame
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        # Draw the "Tight" Grid for visual feedback
        cv2.line(frame, (int(w * LEFT_BOUND), 0), (int(w * LEFT_BOUND), h), (255, 255, 255), 1)
        cv2.line(frame, (int(w * RIGHT_BOUND), 0), (int(w * RIGHT_BOUND), h), (255, 255, 255), 1)
        cv2.line(frame, (0, int(h * UP_BOUND)), (w, int(h * UP_BOUND)), (255, 255, 255), 1)
        cv2.line(frame, (0, int(h * DOWN_BOUND)), (w, int(h * DOWN_BOUND)), (255, 255, 255), 1)

        status_text = "NEUTRAL"
        status_color = (200, 200, 200)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Center of the palm (Middle Finger MCP)
            cx = hand_landmarks.landmark[9].x
            cy = hand_landmarks.landmark[9].y

            # Calculate distance from center (0.5, 0.5)
            x_dist = abs(cx - 0.5)
            y_dist = abs(cy - 0.5)

            # -------------------------
            # LOGIC: Axis Dominance & Game Control
            # -------------------------
            # Instead of pressing keys, we directly set the direction_command
            
            if x_dist > y_dist: 
                # Horizontal movement is dominant
                if cx < LEFT_BOUND:
                    self.game_engine.direction_command = Direction.LEFT
                    status_text = "LEFT"
                    status_color = (0, 255, 255)
                elif cx > RIGHT_BOUND:
                    self.game_engine.direction_command = Direction.RIGHT
                    status_text = "RIGHT"
                    status_color = (0, 255, 255)
            else:
                # Vertical movement is dominant
                if cy < UP_BOUND:
                    self.game_engine.direction_command = Direction.UP
                    status_text = "UP"
                    status_color = (0, 255, 0)
                elif cy > DOWN_BOUND:
                    self.game_engine.direction_command = Direction.DOWN
                    status_text = "DOWN"
                    status_color = (0, 0, 255)

            # Visual Feedback Dot
            cv2.circle(frame, (int(cx * w), int(cy * h)), 10, status_color, -1)

        # Show status text on the camera feed
        cv2.putText(frame, f"CMD: {status_text}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        
        # Display the Camera Feed in a separate window
        cv2.imshow("Hand Controller (Press ESC to Quit)", frame)

    def update(self):
        self.timer.tick(FPS)
        self.game_engine.tick()
        pygame.display.flip()

    def draw(self):
        self.screen.fill([12, 2, 25])

    def check_events(self):
        # We handle camera window 'ESC' key here too
        if cv2.waitKey(1) & 0xFF == 27:
            self.cleanup()
            sys.exit()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
                sys.exit()
            
            # Keep Keyboard ONLY for restarting the game (Spacebar)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.game_engine.game_over:
                    pygame.init()
                    self.game_start_sfx.play()
                    self.game_engine = self.init()

            if event.type == GHOST_EATEN_EVENT:
                self.game_engine.play_ghost_runsaway_sound()
            if event.type == PLAYER_EATEN_EVENT:
                self.game_engine.play_player_eaten_sound()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

    def run(self):
        self.game_start_sfx.play()
        while True:
            self.check_events()       # Check for Quit/Restart
            self.process_hand_input() # Update direction based on hand
            self.update()             # Update Game State
            self.draw()               # Render Game

if __name__ == '__main__':
    game = Game()
    game.run()