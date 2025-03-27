##################################
# player.py - SemiAutoTargetFinder
# with w/h fix
##################################

import pygame
import cv2
import os
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from natsort import natsorted
from tqdm import tqdm

from vis_nav_game import Player, Action, Phase

class SemiAutoTargetFinder(Player):
    """
    Semi-automated player:
    - You control movement (WASD/Arrows).
    - Press 'Q' to compare current FPV with the 4 target images.
    - Press 'Space' to check in if you think you're at the goal.
    """

    def __init__(self):
        super(SemiAutoTargetFinder, self).__init__()
        # Basic states
        self.fpv = None
        self.screen = None
        self.last_act = Action.IDLE

        # SIFT / VLAD references
        self.sift = cv2.SIFT_create()
        self.codebook = None
        self.database = None
        self.num_clusters = 64

        # Exploration data folder
        self.save_dir = "data/images_subsample/"

        # Attempt to load precomputed data
        if os.path.exists("codebook.pkl"):
            self.codebook = pickle.load(open("codebook.pkl", "rb"))
        if os.path.exists("VLAD_database.pkl"):
            self.database = pickle.load(open("VLAD_database.pkl", "rb"))

    def reset(self):
        """Reset each run."""
        self.fpv = None
        self.screen = None
        self.last_act = Action.IDLE
        pygame.init()

    def act(self):
        """Listen for keyboard input and return movement actions."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                # WASD or Arrow keys for movement
                if event.key in [pygame.K_w, pygame.K_UP]:
                    self.last_act |= Action.FORWARD
                elif event.key in [pygame.K_a, pygame.K_LEFT]:
                    self.last_act |= Action.LEFT
                elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                    self.last_act |= Action.RIGHT
                elif event.key in [pygame.K_s, pygame.K_DOWN]:
                    self.last_act |= Action.BACKWARD
                elif event.key == pygame.K_SPACE:
                    # Attempt goal check-in
                    self.last_act |= Action.CHECKIN
                elif event.key == pygame.K_q:
                    # Compare current FPV to target images
                    self.query_target_match()

            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_UP]:
                    self.last_act ^= Action.FORWARD
                elif event.key in [pygame.K_a, pygame.K_LEFT]:
                    self.last_act ^= Action.LEFT
                elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                    self.last_act ^= Action.RIGHT
                elif event.key in [pygame.K_s, pygame.K_DOWN]:
                    self.last_act ^= Action.BACKWARD
                elif event.key == pygame.K_SPACE:
                    self.last_act ^= Action.CHECKIN

        return self.last_act

    def query_target_match(self):
        """Compare the current FPV with each of the 4 target images."""
        targets = self.get_target_images()
        if not targets:
            print("[Query] No target images available.")
            return

        if self.codebook is None:
            print("[Query] No codebook loaded—cannot compare.")
            return

        # Distances for each of the 4 target views
        view_names = ["Front", "Left", "Back", "Right"]
        for i, timg in enumerate(targets):
            dist_val = self.compare_images(self.fpv, timg)
            print(f"[Query] Distance to {view_names[i]} View: {dist_val:.4f}")
        print("[Query] Press Space if distance is small (close to goal).")

    def compare_images(self, imgA, imgB):
        """Compute VLAD distance between two images."""
        vA = self.get_VLAD(imgA)
        vB = self.get_VLAD(imgB)
        return np.linalg.norm(vA - vB)

    def get_VLAD(self, img):
        """Compute VLAD using the loaded codebook."""
        if self.codebook is None:
            return np.zeros(128, dtype=np.float32)

        # SIFT
        _, descriptors = self.sift.detectAndCompute(img, None)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.codebook.cluster_centers_.shape[1], dtype=np.float32)

        labels = self.codebook.predict(descriptors)
        centroids = self.codebook.cluster_centers_
        k, dim = centroids.shape
        vlad = np.zeros((k, dim), dtype=np.float32)

        for i in range(k):
            if np.sum(labels == i) > 0:
                vlad[i] = np.sum(descriptors[labels == i, :] - centroids[i], axis=0)

        vlad = vlad.flatten()
        # Power normalization
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad + 1e-8))
        # L2 normalization
        norm = np.linalg.norm(vlad) + 1e-8
        vlad /= norm
        return vlad

    # Lifecycle hooks
    def pre_exploration(self):
        print("Pre-exploration. If needed, compute SIFT or codebook here.")

    def pre_navigation(self):
        print("Pre-navigation. If needed, load or compute codebook & VLAD data.")

    def see(self, fpv):
        """Display the current FPV on pygame, define w/h up front."""
        if fpv is None or fpv.shape[0] < 3:
            return

        # Always define w/h at the top to avoid local var errors
        h, w, _ = fpv.shape
        self.fpv = fpv

        if self.screen is None:
            self.screen = pygame.display.set_mode((w, h))

        # Convert BGR->RGB
        rgb = cv2.cvtColor(fpv, cv2.COLOR_BGR2RGB)
        py_img = pygame.image.frombuffer(rgb.tobytes(), (w, h), 'RGB')
        pygame.display.set_caption("SemiAutoTargetFinder FPV")
        self.screen.blit(py_img, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        filename='vis_nav_player.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )
    import vis_nav_game as vng

    logging.info("Starting SemiAutoTargetFinder with w/h fix.")
    vng.play(the_player=SemiAutoTargetFinder())
