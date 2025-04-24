import pygame
import os

# Only initialize once
pygame.mixer.init()

def play_sound(path):
    if not os.path.exists(path):
        print(f"[Error] Sound file not found: {path}")
        return
    try:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"[Sound Error] {e}")
