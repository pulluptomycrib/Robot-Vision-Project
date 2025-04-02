# p2.py - SimpleNavPlayer with Goal ID and Visual Feedback
# Provides navigation guidance using VLAD distance to a specific Goal ID
# Displays the goal image and closest matching exploration image
# Uses existing VLAD/SIFT database without recomputing

import pygame
import cv2
import os
import numpy as np
import pickle
from sklearn.cluster import KMeans
from natsort import natsorted
from tqdm import tqdm

from vis_nav_game import Player, Action, Phase

class SimpleNavPlayer(Player):
    def __init__(self):
        super(SimpleNavPlayer, self).__init__()
        # Basic states
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None

        # SIFT / VLAD for visual similarity
        self.sift = cv2.SIFT_create()
        self.codebook = None
        self.vlad_database = None  # VLAD descriptors for exploration images
        self.goal_id = None  # Index of the closest exploration image to target.jpg
        self.goal_image = None  # The actual goal image (from exploration data)
        self.goal_vlad = None  # Precomputed VLAD descriptor for the goal image

        # Exploration data folder
        self.save_dir = "data/images_subsample/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")
            raise FileNotFoundError(f"Exploration data directory {self.save_dir} not found.")

        # Verify that some images exist
        self.exploration_images = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        if not self.exploration_images:
            print(f"No .jpg files found in {self.save_dir}. Please ensure exploration images are present.")
            raise FileNotFoundError(f"No exploration images found in {self.save_dir}.")
        print(f"Found {len(self.exploration_images)} exploration images in {self.save_dir}.")

        # Load existing SIFT descriptors
        if not os.path.exists("sift_descriptors.npy"):
            raise FileNotFoundError("sift_descriptors.npy not found. Please ensure it exists from a previous run.")
        self.sift_descriptors = np.load("sift_descriptors.npy")
        print("Loaded existing SIFT descriptors.")

        # Load existing codebook
        if not os.path.exists("codebook.pkl"):
            raise FileNotFoundError("codebook.pkl not found. Please ensure it exists from a previous run.")
        self.codebook = pickle.load(open("codebook.pkl", "rb"))
        print("Loaded existing codebook.")

        # Load existing VLAD database
        if not os.path.exists("vlad_database.pkl"):
            raise FileNotFoundError("vlad_database.pkl not found. Please ensure it exists from a previous run.")
        self.vlad_database = pickle.load(open("vlad_database.pkl", "rb"))
        print("Loaded existing VLAD database.")

        # Load target.jpg and set the Goal ID
        target_image = cv2.imread("target.jpg")
        if target_image is None:
            print("Failed to load target.jpg. Please ensure it exists in the project directory.")
            raise FileNotFoundError("target.jpg not found.")

        # Set the Goal ID by finding the closest exploration image to target.jpg
        self.set_goal_id(target_image)

        # Track navigation state
        self.action_sequence = [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.BACKWARD]
        self.action_index = 0  # Index to cycle through actions

    def set_goal_id(self, target_image):
        """Set the Goal ID by finding the closest exploration image to target.jpg."""
        print("Setting Goal ID...")
        target_vlad = self.get_VLAD(target_image)
        min_dist = float('inf')
        goal_id = None

        for idx, vlad in enumerate(self.vlad_database):
            dist = np.linalg.norm(target_vlad - vlad)
            if dist < min_dist:
                min_dist = dist
                goal_id = idx

        if goal_id is None:
            raise ValueError("Could not set Goal ID. VLAD database may be empty.")

        self.goal_id = goal_id
        self.goal_image = cv2.imread(os.path.join(self.save_dir, self.exploration_images[goal_id]))
        if self.goal_image is None:
            raise ValueError(f"Failed to load goal image: {self.exploration_images[goal_id]}")
        # Use the precomputed VLAD descriptor from the database
        self.goal_vlad = self.vlad_database[goal_id]
        print(f"Goal ID set to: {self.goal_id} (image: {self.exploration_images[goal_id]})")

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                elif event.key == pygame.K_q:
                    return self.navigate_to_goal()  # Return the action directly
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def get_VLAD(self, img):
        """Compute VLAD descriptor for a given image."""
        if self.codebook is None:
            return np.zeros(128, dtype=np.float32)

        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.codebook.cluster_centers_.shape[1], dtype=np.float32)

        des = np.sqrt(des / np.linalg.norm(des, axis=1, keepdims=True))
        pred_labels = self.codebook.predict(des)
        centroids = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])

        for i in range(k):
            if np.sum(pred_labels == i) > 0:
                VLAD_feature[i] = np.sum(des[pred_labels == i, :] - centroids[i], axis=0)

        VLAD_feature = VLAD_feature.flatten()
        VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature))
        VLAD_feature = VLAD_feature / np.linalg.norm(VLAD_feature)
        return VLAD_feature

    def find_closest_exploration_image(self, vlad_descriptor):
        """Find the exploration image closest to the given VLAD descriptor."""
        min_dist = float('inf')
        closest_idx = None

        for idx, vlad in enumerate(self.vlad_database):
            dist = np.linalg.norm(vlad_descriptor - vlad)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        if closest_idx is None:
            return None, None

        closest_image = cv2.imread(os.path.join(self.save_dir, self.exploration_images[closest_idx]))
        return closest_idx, closest_image

    def navigate_to_goal(self):
        """Estimate proximity to goal, suggest an action, and display images."""
        if self.fpv is None:
            print("[Navigate] No FPV available.")
            return Action.IDLE

        if self.goal_id is None or self.goal_image is None or self.goal_vlad is None:
            print("[Navigate] Goal ID, goal image, or goal VLAD not set.")
            return Action.IDLE

        # Compute VLAD distance to the goal image using precomputed goal_vlad
        current_vlad = self.get_VLAD(self.fpv)
        vlad_dist = np.linalg.norm(current_vlad - self.goal_vlad)
        print(f"[Navigate] VLAD distance to Goal ID {self.goal_id}: {vlad_dist:.4f} (lower is closer)")

        # Find the closest exploration image to the current FPV
        closest_idx, closest_image = self.find_closest_exploration_image(current_vlad)
        if closest_idx is not None:
            print(f"[Navigate] Closest exploration image to current view: {self.exploration_images[closest_idx]} (index: {closest_idx})")

        # Display the goal image and the closest matching exploration image
        if self.screen is not None:
            # Resize images to fit the screen
            h, w, _ = self.fpv.shape
            goal_display = cv2.resize(self.goal_image, (w // 3, h // 3))
            closest_display = closest_image if closest_image is not None else np.zeros_like(goal_display)

            # Convert to Pygame surfaces
            def convert_opencv_img_to_pygame(opencv_image):
                opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
                shape = opencv_image.shape[1::-1]
                pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
                return pygame_image

            # Create a new surface for the combined display
            combined_surface = pygame.Surface((w, h))
            combined_surface.blit(convert_opencv_img_to_pygame(self.fpv), (0, 0))
            combined_surface.blit(convert_opencv_img_to_pygame(goal_display), (0, 0))
            combined_surface.blit(convert_opencv_img_to_pygame(closest_display), (w - w // 3, 0))

            # Add labels
            font = pygame.font.Font(None, 36)
            goal_label = font.render("Goal", True, (255, 255, 255))
            closest_label = font.render("Closest Match", True, (255, 255, 255))
            combined_surface.blit(goal_label, (10, 10))
            combined_surface.blit(closest_label, (w - w // 3 + 10, 10))

            self.screen.blit(combined_surface, (0, 0))
            pygame.display.update()

        # Suggest an action by cycling through the sequence
        action = self.action_sequence[self.action_index]
        action_desc = {
            Action.FORWARD: "Move FORWARD (W or Up arrow)",
            Action.LEFT: "Turn LEFT (A or Left arrow)",
            Action.RIGHT: "Turn RIGHT (D or Right arrow)",
            Action.BACKWARD: "Move BACKWARD (S or Down arrow)"
        }[action]
        print(f"[Navigate] Suggested action: {action_desc}")
        print("[Navigate] After moving, press Q again to check if VLAD distance decreases.")

        # Move to the next action in the sequence for the next call
        self.action_index = (self.action_index + 1) % len(self.action_sequence)

        return action  # Return the action for the game to execute

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        # Display will be handled in navigate_to_goal when Q is pressed
        # For now, just update the FPV
        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image

        pygame.display.set_caption("SimpleNavPlayer:fpv")
        self.screen.blit(convert_opencv_img_to_pygame(fpv), (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    try:
        vis_nav_game.play(the_player=SimpleNavPlayer())
    except Exception as e:
        print(f"Error during game execution: {e}")
        print("This may be due to an NTP timeout. Ensure you have internet access or check if vis_nav_game can be configured to skip NTP synchronization.")
