# p2.py - SimpleNavPlayer with Goal ID
# Provides basic navigation guidance using VLAD distance to a specific Goal ID

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

        # Load or compute SIFT codebook
        if os.path.exists("sift_descriptors.npy"):
            self.sift_descriptors = np.load("sift_descriptors.npy")
        else:
            self.sift_descriptors = self.compute_sift_descriptors()
            np.save("sift_descriptors.npy", self.sift_descriptors)

        if os.path.exists("codebook.pkl"):
            self.codebook = pickle.load(open("codebook.pkl", "rb"))
        else:
            print("Computing codebook...")
            self.codebook = KMeans(n_clusters=128, init='k-means++', n_init=5, verbose=1).fit(self.sift_descriptors)
            pickle.dump(self.codebook, open("codebook.pkl", "wb"))

        # Load or compute VLAD database
        if os.path.exists("vlad_database.pkl"):
            self.vlad_database = pickle.load(open("vlad_database.pkl", "rb"))
        else:
            self.vlad_database = self.compute_vlad_database()
            with open("vlad_database.pkl", "wb") as f:
                pickle.dump(self.vlad_database, f)

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

    def compute_sift_descriptors(self):
        """Compute SIFT descriptors for all exploration images."""
        print("Computing SIFT features...")
        sift_descriptors = []
        for img in tqdm(self.exploration_images, desc="Processing images for SIFT"):
            img_path = os.path.join(self.save_dir, img)
            img_data = cv2.imread(img_path)
            if img_data is None:
                print(f"Failed to load image for SIFT: {img_path}")
                continue
            _, des = self.sift.detectAndCompute(img_data, None)
            if des is not None:
                des = np.sqrt(des / np.linalg.norm(des, axis=1, keepdims=True))
                sift_descriptors.extend(des)
        if not sift_descriptors:
            raise ValueError("No SIFT descriptors computed. Check if images are accessible.")
        return np.asarray(sift_descriptors)

    def compute_vlad_database(self):
        """Compute VLAD descriptors for all exploration images."""
        print("Computing VLAD database...")
        vlad_database = []
        for img in tqdm(self.exploration_images, desc="Processing images for VLAD"):
            img_path = os.path.join(self.save_dir, img)
            img_data = cv2.imread(img_path)
            if img_data is None:
                print(f"Failed to load image for VLAD: {img_path}")
                continue
            vlad = self.get_VLAD(img_data)
            vlad_database.append(vlad)
        if not vlad_database:
            raise ValueError("No VLAD descriptors computed for database. Check if images are accessible.")
        return vlad_database

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

    def navigate_to_goal(self):
        """Estimate proximity to goal and suggest an action."""
        if self.fpv is None:
            print("[Navigate] No FPV available.")
            return Action.IDLE

        if self.goal_id is None or self.goal_image is None:
            print("[Navigate] Goal ID or goal image not set.")
            return Action.IDLE

        # Compute VLAD distance to the goal image (the exploration image at Goal ID)
        goal_vlad = self.get_VLAD(self.goal_image)
        current_vlad = self.get_VLAD(self.fpv)
        vlad_dist = np.linalg.norm(goal_vlad - current_vlad)
        print(f"[Navigate] VLAD distance to Goal ID {self.goal_id}: {vlad_dist:.4f} (lower is closer)")

        # Since we can't simulate the new FPV after an action, cycle through actions
        # and let the user monitor the VLAD distance
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

        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')
            return pygame_image

        pygame.display.set_caption("SimpleNavPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=SimpleNavPlayer())
