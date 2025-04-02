# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree, NearestNeighbors
from tqdm import tqdm
from natsort import natsorted
import networkx as nx

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()
        
        # Variables for reading exploration data
        self.save_dir = "data/images_subsample/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()
        # Load pre-trained SIFT features and codebook
        self.sift_descriptors, self.codebook = None, None
        if os.path.exists("sift_descriptors.npy"):
            self.sift_descriptors = np.load("sift_descriptors.npy")
        if os.path.exists("codebook.pkl"):
            self.codebook = pickle.load(open("codebook.pkl", "rb"))
        # Initialize database for storing VLAD descriptors of FPV
        self.database = None
        self.goal = None
        self.tree = None
        self.graph = None  # For navigation graph

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
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])
        w, h = concat_img.shape[:2]
        color = (0, 0, 0)
        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)
        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1
        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        path = self.save_dir + str(id) + ".jpg"
        if os.path.exists(path):
            img = cv2.imread(path)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        else:
            print(f"Image with ID {id} does not exist")

    def compute_sift_features(self):
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        sift_descriptors = list()
        for img in tqdm(files, desc="Processing images"):
            img = cv2.imread(os.path.join(self.save_dir, img))
            _, des = self.sift.detectAndCompute(img, None)
            sift_descriptors.extend(des)
        return np.asarray(sift_descriptors)
    
    def get_VLAD(self, img):
        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.codebook.cluster_centers_.shape[0] * self.codebook.cluster_centers_.shape[1], dtype=np.float32)
        pred_labels = self.codebook.predict(des)
        centroids = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])
        for i in range(k):
            if np.sum(pred_labels == i) > 0:
                VLAD_feature[i] = np.sum(des[pred_labels==i, :] - centroids[i], axis=0)
        VLAD_feature = VLAD_feature.flatten()
        VLAD_feature = np.sign(VLAD_feature) * np.sqrt(np.abs(VLAD_feature))
        VLAD_feature = VLAD_feature / np.linalg.norm(VLAD_feature)
        return VLAD_feature

    def get_neighbor(self, img):
        q_VLAD = self.get_VLAD(img).reshape(1, -1)
        _, index = self.tree.query(q_VLAD, 1)
        return index[0][0]

    def build_knn_graph(self):
        graph_path = "knn_graph.pkl"
        if os.path.exists(graph_path):
            print(f"Loading existing k-NN graph from {graph_path}...")
            with open(graph_path, 'rb') as f:
                return pickle.load(f)
        print("Building k-NN graph...")
        graph = nx.Graph()
        for i in range(len(self.database)):
            graph.add_node(i)
        k = 5 + 1  # Reduced from 10 to 5 for faster computation
        vlad_array = np.array(self.database, dtype=np.float32)
        # Use NearestNeighbors with 'auto' algorithm for better performance
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean', n_jobs=-1)
        nbrs.fit(vlad_array)
        distances, indices = nbrs.kneighbors(vlad_array)
        for i in tqdm(range(len(self.database)), desc="Adding edges"):
            for j_idx in range(1, k):  # Skip the first neighbor (itself)
                j = indices[i][j_idx]
                d = distances[i][j_idx]
                graph.add_edge(i, j, weight=d)
        print(f"Saving k-NN graph to {graph_path}...")
        with open(graph_path, 'wb') as f:
            pickle.dump(graph, f)
        return graph

    def pre_nav_compute(self):
        # Compute or load SIFT features
        if self.sift_descriptors is None:
            print("Computing SIFT features...")
            self.sift_descriptors = self.compute_sift_features()
            np.save("sift_descriptors.npy", self.sift_descriptors)
        else:
            print("Loaded SIFT features from sift_descriptors.npy")

        # Compute or load codebook
        if self.codebook is None:
            print("Computing codebook...")
            self.codebook = KMeans(
                n_clusters=64,
                init='k-means++',
                n_init=3,
                max_iter=100,
                verbose=1
            ).fit(self.sift_descriptors)
            pickle.dump(self.codebook, open("codebook.pkl", "wb"))
        else:
            print("Loaded codebook from codebook.pkl")
        
        # Load or compute VLAD embeddings
        vlad_path = "vlad_database.pkl"
        if os.path.exists(vlad_path):
            print(f"Loading existing VLAD database from {vlad_path}...")
            with open(vlad_path, 'rb') as f:
                self.database = pickle.load(f)
        else:
            self.database = []
            print("Computing VLAD embeddings...")
            exploration_observation = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
            for img in tqdm(exploration_observation, desc="Processing images"):
                img = cv2.imread(os.path.join(self.save_dir, img))
                VLAD = self.get_VLAD(img)
                self.database.append(VLAD)
            print(f"Saving VLAD database to {vlad_path}...")
            with open(vlad_path, 'wb') as f:
                pickle.dump(self.database, f)

        # Build a BallTree for fast nearest neighbor search
        print("Building BallTree...")
        tree = BallTree(self.database, leaf_size=40)
        self.tree = tree

        # Build k-NN graph for navigation
        self.graph = self.build_knn_graph()

    def pre_navigation(self):
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
        
    def display_next_best_view(self):
        current_idx = self.get_neighbor(self.fpv)
        try:
            path = nx.astar_path(self.graph, current_idx, self.goal, weight='weight')
            next_idx = path[1] if len(path) > 1 else current_idx
            print(f"Current ID: {current_idx}, Next ID towards goal: {next_idx}, Goal ID: {self.goal}")
            self.display_img_from_id(next_idx, 'Next Best View')
        except nx.NetworkXNoPath:
            print("No path found to the goal. Showing closest image.")
            self.display_img_from_id(current_idx + 3, 'Next Best View')

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

        pygame.display.set_caption("KeyboardPlayer:fpv")
        if self._state:
            if self._state[1] == Phase.EXPLORATION:
                pass
            elif self._state[1] == Phase.NAVIGATION:
                if self.goal is None:
                    targets = self.get_target_images()
                    index = self.get_neighbor(targets[0])
                    self.goal = index
                    print(f'Goal ID: {self.goal}')
                keys = pygame.key.get_pressed()
                if keys[pygame.K_q]:
                    self.display_next_best_view()

        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
