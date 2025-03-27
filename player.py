##############################
# player.py (Fully Automated)
##############################

from vis_nav_game import Player, Action
import pygame
import cv2

class FullyAutoPlayer(Player):
    def __init__(self):
        super(FullyAutoPlayer, self).__init__()
        self.fpv = None
        self.screen = None

        # Step counter to determine when to check in at the goal
        self.step_count = 0
        self.forward_limit = 200  # Number of steps to move forward before checking in

    def reset(self):
        """
        Resets the player state.
        """
        self.fpv = None
        self.screen = None
        self.step_count = 0
        print("FullyAutoPlayer reset.")

        pygame.init()

    def act(self):
        """
        Automatically moves forward for 'forward_limit' steps,
        then attempts to check in at the goal.
        """
        # Handle only QUIT events, ignore all else
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return Action.QUIT

        # If we haven't reached the forward limit, keep moving forward
        if self.step_count < self.forward_limit:
            self.step_count += 1
            return Action.FORWARD
        else:
            # After 'forward_limit' steps, try to check in
            print("Reached forward limit. Checking in at goal.")
            return Action.CHECKIN

    def show_target_images(self):
        """
        (Optional) If you still want to display the 4 target images,
        but since this is fully auto, it won’t be used.
        """
        targets = self.get_target_images()
        if not targets:
            return

        # Simple display example
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])
        cv2.imshow('FullyAutoPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Optional: automatically show target images if needed.
        """
        super(FullyAutoPlayer, self).set_target_images(images)
        self.show_target_images()

    def pre_exploration(self):
        """
        Called before exploration phase. Not used in fully auto.
        """
        K = self.get_camera_intrinsic_matrix()
        print(f'Camera Intrinsics: {K}')

    def pre_navigation(self):
        """
        Called before navigation phase. Not used in fully auto.
        """
        pass

    def see(self, fpv):
        """
        Receives a first-person-view image each frame.
        We just display it, but ignore keyboard controls.
        """
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # Initialize a display window if needed
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]
            return pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

        pygame.display.set_caption("FullyAutoPlayer:fpv")
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
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
    logging.info(f'player.py is using vis_nav_game {vng.core.__version__}')

    # Launch the game with FullyAutoPlayer
    vng.play(the_player=FullyAutoPlayer())
