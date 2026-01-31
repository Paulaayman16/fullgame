import sys
import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
from collections import deque, Counter
from model.board_definition import BoardDefinition
from model.level_config import LevelConfig
from settings import *
from levels.level_content_initializer import LevelContentInitializer
from model.direction import Direction

# -------------------------
# Configuration
# -------------------------
CAMERA_INDEX = 0
SIDEBAR_WIDTH = 320
CAM_HEIGHT = 240

# --- FINGER JOYSTICK SETTINGS ---
# Wrist-to-Tip Logic
SMOOTHING_WINDOW = 3       
DEADZONE_THRESHOLD = 0.08  # Finger must extend 8% from wrist
LOCK_ANGLE_BUFFER = 0.25   

class Game:
    def __init__(self):
        pygame.init()
        self.game_width = RESOLUTION[0]
        self.game_height = RESOLUTION[1]
        self.canvas_width = self.game_width + SIDEBAR_WIDTH
        self.canvas_height = max(self.game_height, CAM_HEIGHT)
        self.screen = pygame.display.set_mode((1100, 600), pygame.RESIZABLE)
        pygame.display.set_caption("Pac-Man AI Controller")
        self.canvas = pygame.Surface((self.canvas_width, self.canvas_height))
        self.game_surface = pygame.Surface((self.game_width, self.game_height))
        self.timer = pygame.time.Clock()
        
        self.game_engine = self.init_game(self.game_surface)
        self.game_start_sfx = pygame.mixer.Sound('media/game_start.wav')

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cam_surface = None
        
        # State
        self.gesture_history = deque(maxlen=SMOOTHING_WINDOW)
        self.last_committed_direction = None 
        self.waiting_for_start = True
        self.is_paused = False
        
        self.game_surface.fill([12, 2, 25])
        self.game_engine.tick() 
        print("JUDGE MODE: Finger Joystick. Press SPACE to Start.")

    def init_game(self, surface_to_draw_on):
        board = BOARD.copy()
        board_definition = BoardDefinition(board)
        level_1 = LevelConfig(wall_color='blue', gate_color='white',
                              board_definition=board_definition, power_up_limit=POWER_UP_LIMIT)
        level_init = LevelContentInitializer(level_1, surface_to_draw_on)
        return level_init.init_game_engine()

    def check_palm_open(self, lm):
        """Pause Gesture: Are all fingers extended?"""
        fingers_open = 0
        if lm[8].y < lm[6].y: fingers_open += 1
        if lm[12].y < lm[10].y: fingers_open += 1
        if lm[16].y < lm[14].y: fingers_open += 1
        if lm[20].y < lm[18].y: fingers_open += 1
        return fingers_open >= 4

    def process_hand_input(self):
        if not self.cap.isOpened(): return
        ret, frame = self.cap.read()
        if not ret: return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        status_text = "NEUTRAL"
        status_color = (200, 200, 200)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # --- WAITING SCREEN (Hybrid Mode) ---
            if self.waiting_for_start or self.game_engine.game_over:
                # We don't need complex calibration logic for Finger Joystick
                # because the Wrist is the dynamic center.
                # We just need a trigger to start.
                pass 

            else:
                # --- GAMEPLAY LOGIC ---
                
                # 1. Pause Check (Open Hand)
                if self.check_palm_open(lm):
                    self.is_paused = True
                else:
                    self.is_paused = False
                    
                    # 2. FINGER JOYSTICK LOGIC (Wrist -> Tip)
                    wrist_x, wrist_y = lm[0].x, lm[0].y
                    tip_x, tip_y = lm[8].x, lm[8].y
                    
                    # Visual Joystick Line
                    cv2.line(frame, (int(wrist_x*w), int(wrist_y*h)), 
                                    (int(tip_x*w), int(tip_y*h)), (255, 255, 0), 2)

                    # Vector Calculation
                    dx = tip_x - wrist_x
                    dy = tip_y - wrist_y
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    detected_direction = None

                    # 3. Deadzone Check
                    if distance > DEADZONE_THRESHOLD:
                        angle = math.atan2(dy, dx)
                        new_direction = None
                        
                        # 4. Sticky Angle Logic
                        if self.last_committed_direction == Direction.UP:
                             if (-2.35 - LOCK_ANGLE_BUFFER) < angle <= (-0.78 + LOCK_ANGLE_BUFFER): new_direction = Direction.UP
                        elif self.last_committed_direction == Direction.DOWN:
                             if (0.78 - LOCK_ANGLE_BUFFER) <= angle < (2.35 + LOCK_ANGLE_BUFFER): new_direction = Direction.DOWN
                        elif self.last_committed_direction == Direction.RIGHT:
                             if (-0.78 - LOCK_ANGLE_BUFFER) < angle < (0.78 + LOCK_ANGLE_BUFFER): new_direction = Direction.RIGHT
                        elif self.last_committed_direction == Direction.LEFT:
                             if (2.35 - LOCK_ANGLE_BUFFER) < angle or angle <= (-2.35 + LOCK_ANGLE_BUFFER): new_direction = Direction.LEFT

                        if new_direction is None:
                            if -0.78 < angle < 0.78: new_direction = Direction.RIGHT
                            elif 0.78 <= angle < 2.35: new_direction = Direction.DOWN
                            elif -2.35 < angle <= -0.78: new_direction = Direction.UP
                            else: new_direction = Direction.LEFT
                        
                        self.gesture_history.append(new_direction)
                        most_common_dir, count = Counter(self.gesture_history).most_common(1)[0]
                        
                        # 5. Continuous Input (Spamming)
                        if count >= 2:
                            self.game_engine.direction_command = most_common_dir
                            self.last_committed_direction = most_common_dir
                            
                            # Visuals
                            if most_common_dir == Direction.LEFT: 
                                status_text = "LEFT"
                                self.highlight_slice(frame, "LEFT", (255, 255, 0), wrist_x, wrist_y)
                            elif most_common_dir == Direction.RIGHT: 
                                status_text = "RIGHT"
                                self.highlight_slice(frame, "RIGHT", (0, 255, 255), wrist_x, wrist_y)
                            elif most_common_dir == Direction.UP: 
                                status_text = "UP"
                                self.highlight_slice(frame, "UP", (0, 255, 0), wrist_x, wrist_y)
                            elif most_common_dir == Direction.DOWN: 
                                status_text = "DOWN"
                                self.highlight_slice(frame, "DOWN", (0, 0, 255), wrist_x, wrist_y)
                            status_color = (0, 255, 0)
                    else:
                        self.last_committed_direction = None
                    
                    # Draw Anchor Points
                    cv2.circle(frame, (int(wrist_x*w), int(wrist_y*h)), 6, (0, 0, 255), -1) 
                    cv2.circle(frame, (int(tip_x*w), int(tip_y*h)), 6, status_color, -1)    

        cv2.putText(frame, f"{status_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        
        frame_resized = cv2.resize(frame, (SIDEBAR_WIDTH, CAM_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_rgb = np.transpose(frame_rgb, (1, 0, 2))
        self.cam_surface = pygame.surfarray.make_surface(frame_rgb)

    def highlight_slice(self, frame, direction, color, cx_rel, cy_rel):
        """Draws wedges relative to the WRIST (Dynamic Center)."""
        h, w, _ = frame.shape
        cx = int(cx_rel * w)
        cy = int(cy_rel * h)
        overlay = frame.copy()
        pts = []
        if direction == "RIGHT": pts = np.array([[cx, cy], [w, -h], [w, 2*h]])
        elif direction == "DOWN": pts = np.array([[cx, cy], [2*w, h], [-w, h]])
        elif direction == "LEFT": pts = np.array([[cx, cy], [0, 2*h], [0, -h]])
        elif direction == "UP": pts = np.array([[cx, cy], [-w, 0], [2*w, 0]])
        if len(pts) > 0:
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    def update(self):
        self.timer.tick(FPS)
        self.canvas.fill([12, 2, 25])
        
        if not self.waiting_for_start:
            self.game_surface.fill([12, 2, 25])
            if not self.is_paused:
                self.game_engine.tick() 
        
        sidebar_x = self.game_width
        pygame.draw.rect(self.canvas, (30, 30, 40), (sidebar_x, 0, SIDEBAR_WIDTH, self.canvas_height))
        self.canvas.blit(self.game_surface, (0, 0))
        
        if self.cam_surface is not None:
            cam_x, cam_y = sidebar_x, 50
            self.canvas.blit(self.cam_surface, (cam_x, cam_y))
            pygame.draw.rect(self.canvas, (0, 255, 0), (cam_x, cam_y, SIDEBAR_WIDTH, CAM_HEIGHT), 2)
            
            # --- "SCIFI" UI LOGIC ---
            current_time = pygame.time.get_ticks()
            
            if self.waiting_for_start or self.game_engine.game_over:
                label_text = "PRESS SPACE TO START"
                # Blink Yellow/White
                if current_time % 1000 < 500:
                    label_color = (255, 255, 0)
                else:
                    label_color = (255, 255, 255)
            elif self.is_paused:
                label_text = "!!! SYSTEM PAUSED !!!"
                label_color = (255, 0, 0)
            else:
                # PULSING "NEURAL LINK" TEXT
                label_text = ">>> NEURAL LINK ESTABLISHED <<<"
                if current_time % 500 < 250:
                    label_color = (0, 255, 255) # Cyan
                else:
                    label_color = (200, 255, 255) # White-ish
            
            font = pygame.font.SysFont('Arial', 16, bold=True)
            label = font.render(label_text, True, label_color)
            text_rect = label.get_rect(center=(cam_x + SIDEBAR_WIDTH//2, cam_y + CAM_HEIGHT + 20))
            self.canvas.blit(label, text_rect)

        scaled_surface = pygame.transform.smoothscale(self.canvas, self.screen.get_size())
        self.screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()

    def reset_game(self):
        self.game_start_sfx.play()
        self.game_engine = self.init_game(self.game_surface)
        self.gesture_history.clear()
        self.waiting_for_start = False
        self.last_committed_direction = None
        self.is_paused = False

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
                sys.exit()
            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            if event.type == pygame.KEYDOWN:
                # --- HYBRID SPACEBAR CONTROL ---
                if event.key == pygame.K_SPACE:
                    if self.waiting_for_start: 
                        self.waiting_for_start = False
                        self.game_start_sfx.play()
                    elif self.game_engine.game_over: 
                        self.reset_game()
                    else:
                        # Optional: Space to Toggle Pause during game
                        self.is_paused = not self.is_paused
            
            if event.type == GHOST_EATEN_EVENT:
                self.game_engine.play_ghost_runsaway_sound()
            if event.type == PLAYER_EATEN_EVENT:
                self.game_engine.play_player_eaten_sound()

    def cleanup(self):
        self.cap.release()
        pygame.quit()

    def run(self):
        while True:
            self.check_events()
            self.process_hand_input()
            self.update()

if __name__ == '__main__':
    game = Game()
    game.run()