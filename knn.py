###########################################################
# p2_threshold.py - SimpleNavPlayer with Goal ID, 
# Visual Feedback, and a Threshold-based "kNN" Graph
# to avoid O(N^2) full pairwise distances.
###########################################################

################################################
# Monkey-patch ntplib to avoid real NTP queries
################################################
import ntplib
import time

class DummyNTPResponse:
    def __init__(self):
        # Just return local system time
        self.tx_time = time.time()

class FakeNTPClient(ntplib.NTPClient):
    def request(self, *args, **kwargs):
        print("Pretending to fetch time from NTP; returning local time.")
        return DummyNTPResponse()

ntplib.NTPClient = FakeNTPClient


import pygame
import cv2
import os
import numpy as np
import pickle
from sklearn.cluster import KMeans
from natsort import natsorted
from tqdm import tqdm
import networkx as nx
import random

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
        self.goal_id = None        # Index of the closest exploration image to target.jpg
        self.goal_image = None     # The actual goal image (from exploration data)
        self.goal_vlad = None      # Precomputed VLAD descriptor for the goal image
        self.graph = None          # Our threshold-based adjacency graph

        ####################################################
        # 1) Load the data folder
        ####################################################
        self.save_dir = "data/images_subsample/"
        if not os.path.exists(self.save_dir):
            raise FileNotFoundError(f"Exploration data directory {self.save_dir} not found.")

        # Verify images exist
        self.all_exploration_images = natsorted(
            [x for x in os.listdir(self.save_dir) if x.endswith('.jpg')]
        )
        if not self.all_exploration_images:
            raise FileNotFoundError(f"No .jpg files found in {self.save_dir}.")

        print(f"Found {len(self.all_exploration_images)} images in {self.save_dir}.")

        # ------------------------------------------------------------
        # (A) OPTIONAL: Subsample the images to reduce the dataset size
        # e.g. take 1 out of every 5 images
        # ------------------------------------------------------------
        subsample_ratio = 5
        self.exploration_images = self.all_exploration_images[::subsample_ratio]
        print(f"Subsampled from {len(self.all_exploration_images)} to "
              f"{len(self.exploration_images)} images (ratio = {subsample_ratio}).")

        ####################################################
        # 2) Load precomputed SIFT, codebook, and VLAD
        ####################################################
        if not os.path.exists("sift_descriptors.npy"):
            raise FileNotFoundError("sift_descriptors.npy not found.")
        self.sift_descriptors = np.load("sift_descriptors.npy")
        print("Loaded existing SIFT descriptors.")

        if not os.path.exists("codebook.pkl"):
            raise FileNotFoundError("codebook.pkl not found.")
        self.codebook = pickle.load(open("codebook.pkl", "rb"))
        print("Loaded existing codebook.")

        if not os.path.exists("vlad_database.pkl"):
            raise FileNotFoundError("vlad_database.pkl not found.")
        full_vlad_db = pickle.load(open("vlad_database.pkl", "rb"))
        print("Loaded existing VLAD database of size:", len(full_vlad_db))

        # NOTE: Because we are subsampling images, we also need to 
        # subsample the vlad_database in the same way
        # vlad_database[i] corresponds to all_exploration_images[i],
        # so we must keep them in sync with the new self.exploration_images.
        # We do a fresh list comprehension:
        # We need the indices we took from all_exploration_images
        original_indices = list(range(0, len(self.all_exploration_images), subsample_ratio))
        self.vlad_database = [full_vlad_db[i] for i in original_indices]
        print(f"Subsampled the VLAD database to match the {len(self.exploration_images)} images.")

        ####################################################
        # 3) Build or load the threshold-based adjacency graph
        ####################################################
        self.graph_path = "threshold_graph.pkl"
        self.build_threshold_graph()

        ####################################################
        # 4) Load target.jpg and set the Goal ID
        ####################################################
        target_image = cv2.imread("target.jpg")
        if target_image is None:
            raise FileNotFoundError("target.jpg not found in project directory.")

        self.set_goal_id(target_image)

        # Action cycling
        self.action_sequence = [Action.FORWARD, Action.LEFT, Action.RIGHT, Action.BACKWARD]
        self.action_index = 0

    def build_threshold_graph(self):
        """
        Build or load a threshold-based adjacency graph.
        For each image, randomly pick a subset of other images, 
        and connect edges if distance < threshold.
        """
        if os.path.exists(self.graph_path):
            print(f"Loading threshold-based graph from {self.graph_path} ...")
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            return

        print("Building threshold-based adjacency graph (faster than O(N^2) full pass)...")
        self.graph = nx.Graph()

        # Add each image as a node
        n = len(self.vlad_database)
        for i in range(n):
            self.graph.add_node(i)

        # Parameters to tune
        threshold = 0.75     # If distance < 0.75, we add an edge
        random_candidates = 500  # how many random images to compare per node

        for i in tqdm(range(n), desc="Building threshold graph"):
            # We'll pick random_candidates other images to check
            # This is drastically fewer than n-1 for large n
            indices = random.sample(range(n), min(random_candidates, n))
            for j in indices:
                if j == i:
                    continue
                dist = np.linalg.norm(self.vlad_database[i] - self.vlad_database[j])
                if dist < threshold:
                    self.graph.add_edge(i, j, weight=dist)

        print(f"Saving threshold-based graph to {self.graph_path}...")
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)

    def set_goal_id(self, target_image):
        """Compute a VLAD descriptor for target and find the best matching exploration image."""
        print("Setting Goal ID from target.jpg ...")
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

        # Load the actual goal image from self.exploration_images
        goal_filename = self.exploration_images[goal_id]
        self.goal_image = cv2.imread(os.path.join(self.save_dir, goal_filename))
        self.goal_vlad = self.vlad_database[goal_id]

        print(f"Goal ID set to {self.goal_id}, image: {goal_filename}")

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()

        # Extra keys for WASD
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_w: Action.FORWARD,
            pygame.K_s: Action.BACKWARD,
            pygame.K_a: Action.LEFT,
            pygame.K_d: Action.RIGHT,
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
                    return self.navigate_to_goal()  # Press Q => navigate
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def get_VLAD(self, img):
        # same as your existing approach
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
        norm = np.linalg.norm(VLAD_feature) + 1e-8
        VLAD_feature /= norm
        return VLAD_feature

    def find_closest_exploration_image(self, vlad_descriptor):
        """Find the exploration image closest to the given VLAD descriptor (subsampled)."""
        min_dist = float('inf')
        closest_idx = None
        for idx, vlad in enumerate(self.vlad_database):
            dist = np.linalg.norm(vlad_descriptor - vlad)
            if dist < min_dist:
                min_dist = dist
                closest_idx = idx

        if closest_idx is None:
            return None, None

        closest_filename = self.exploration_images[closest_idx]
        closest_image = cv2.imread(os.path.join(self.save_dir, closest_filename))
        return closest_idx, closest_image

    def navigate_to_goal(self):
        """Compute current->goal path in the threshold-based graph, suggest next step, show images."""
        if self.fpv is None:
            print("[Navigate] No FPV available.")
            return Action.IDLE
        if self.goal_id is None or self.goal_image is None:
            print("[Navigate] No valid goal set.")
            return Action.IDLE

        # 1) Distance to the goal
        current_vlad = self.get_VLAD(self.fpv)
        dist_to_goal = np.linalg.norm(current_vlad - self.goal_vlad)
        print(f"[Navigate] Distance to goal: {dist_to_goal:.4f}")

        # 2) Where am I? => nearest exploration image
        current_idx, closest_image = self.find_closest_exploration_image(current_vlad)
        if current_idx is None:
            print("[Navigate] Could not find a closest image for current FPV.")
            return Action.IDLE
        print(f"[Navigate] Current ID = {current_idx} | Goal ID = {self.goal_id}")

        # 3) A* path in threshold-based graph
        try:
            path = nx.astar_path(self.graph, current_idx, self.goal_id, weight='weight')
            if len(path) > 1:
                print("[Navigate] Path:", " -> ".join(map(str, path)))
                next_idx = path[1]
            else:
                next_idx = current_idx
                print("[Navigate] You're at the goal node!")
        except nx.NetworkXNoPath:
            print("[Navigate] No path found. Possibly the threshold is too low or not enough edges. Using default action.")
            path = [current_idx]
            next_idx = current_idx

        # 4) Display the goal + the closest image
        if self.screen:
            h, w, _ = self.fpv.shape
            # resize images for side-by-side
            goal_display = cv2.resize(self.goal_image, (w // 3, h // 3))
            if closest_image is not None:
                closest_display = cv2.resize(closest_image, (w // 3, h // 3))
            else:
                closest_display = np.zeros((h // 3, w // 3, 3), dtype=np.uint8)

            def convert_opencv_img_to_pygame(opencv_image):
                opencv_image = opencv_image[:, :, ::-1]
                shape = opencv_image.shape[1::-1]
                return pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            combined_surface = pygame.Surface((w, h))
            # show the current FPV on left
            fpv_pygame = convert_opencv_img_to_pygame(self.fpv)
            combined_surface.blit(fpv_pygame, (0, 0))

            # top-left corner: goal
            goal_pygame = convert_opencv_img_to_pygame(goal_display)
            combined_surface.blit(goal_pygame, (0, 0))

            # top-right corner: closest
            closest_pygame = convert_opencv_img_to_pygame(closest_display)
            combined_surface.blit(closest_pygame, (w - w // 3, 0))

            font = pygame.font.Font(None, 36)
            goal_label = font.render("Goal", True, (255, 255, 255))
            combined_surface.blit(goal_label, (10, 10))
            closest_label = font.render("Closest", True, (255, 255, 255))
            combined_surface.blit(closest_label, (w - w // 3 + 10, 10))

            self.screen.blit(combined_surface, (0, 0))
            pygame.display.update()

        # 5) Action suggestion
        action = self.action_sequence[self.action_index]
        action_desc = {
            Action.FORWARD: "Move FORWARD",
            Action.LEFT: "Turn LEFT",
            Action.RIGHT: "Turn RIGHT",
            Action.BACKWARD: "Move BACKWARD"
        }[action]
        print(f"[Navigate] Suggestion: {action_desc}")
        print("[Navigate] Press Q again after moving to see if distance decreased.")

        self.action_index = (self.action_index + 1) % len(self.action_sequence)
        return action

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        # Just update the display
        def convert_opencv_img_to_pygame(opencv_image):
            opencv_image = opencv_image[:, :, ::-1]
            shape = opencv_image.shape[1::-1]
            return pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

        pygame.display.set_caption("SimpleNavPlayer:fpv")
        self.screen.blit(convert_opencv_img_to_pygame(fpv), (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    import sys
    try:
        vis_nav_game.play(the_player=SimpleNavPlayer())
    except Exception as e:
        print("Error during execution:", e)
        sys.exit(1)
