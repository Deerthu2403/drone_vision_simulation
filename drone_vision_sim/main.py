# main.py
import pygame
import cv2
import numpy as np
import os
import sys

WIDTH, HEIGHT = 800, 600
FPS = 30

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Vision Navigation Simulation")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE  = (0, 0, 255)

# Drone properties
drone_x, drone_y = 100, 100
drone_speed = 3

# Landing pad (target)
landing_pad = pygame.Rect(600, 400, 120, 120)

# Obstacles
obstacles = [
    pygame.Rect(300, 200, 100, 40),
    pygame.Rect(420, 320, 60, 140),
    pygame.Rect(160, 450, 200, 40)
]

# Load QR image if available and scale to landing pad size
QR_PATH = os.path.join("assets", "qr.png")
qr_image = None
if os.path.exists(QR_PATH):
    try:
        qr_image = pygame.image.load(QR_PATH).convert()
        qr_image = pygame.transform.smoothscale(qr_image, (landing_pad.width, landing_pad.height))
    except Exception as e:
        print("Failed to load QR image:", e)
        qr_image = None
else:
    print("No QR image found at", QR_PATH, "- running without visible QR.")

# Utility: convert pygame surface to OpenCV image (BGR)
def pygame_to_cvimage():
    data = pygame.surfarray.array3d(pygame.display.get_surface())
    data = np.rot90(data)  # rotate to correct orientation
    bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return bgr

# Simple movement: move drone towards target
def move_drone_towards(target_x, target_y):
    global drone_x, drone_y
    if abs(drone_x - target_x) > drone_speed:
        drone_x += drone_speed if drone_x < target_x else -drone_speed
    else:
        drone_x = target_x
    if abs(drone_y - target_y) > drone_speed:
        drone_y += drone_speed if drone_y < target_y else -drone_speed
    else:
        drone_y = target_y

def main():
    detector = cv2.QRCodeDetector()
    running = True

    while running:
        # Handle events (important for IDLE window responsiveness)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw environment
        screen.fill(WHITE)
        # Landing pad
        pygame.draw.rect(screen, GREEN, landing_pad)
        # Blit QR on landing pad if available
        if qr_image is not None:
            screen.blit(qr_image, landing_pad.topleft)
        # Obstacles
        for obs in obstacles:
            pygame.draw.rect(screen, RED, obs)
        # Drone
        pygame.draw.circle(screen, BLUE, (int(drone_x), int(drone_y)), 12)

        # Update drone movement (towards center of landing pad)
        move_drone_towards(landing_pad.centerx, landing_pad.centery)

        # Display pygame
        pygame.display.flip()

        # ---- Vision: capture screen and process with OpenCV ----
        frame = pygame_to_cvimage()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect green landing pad (color mask)
        lower_green = np.array([36, 50, 50])
        upper_green = np.array([86, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Landing Pad", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Detect QR code using OpenCV's built-in detector
        data, points, _ = detector.detectAndDecode(frame)
        if points is not None and data:
            pts = points.astype(int).reshape(-1, 2)
            # draw polygon
            for i in range(len(pts)):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i+1) % len(pts)])
                cv2.line(frame, pt1, pt2, (255, 0, 255), 2)
            cv2.putText(frame, f"QR: {data}", (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

        # Show vision window
        cv2.imshow("Drone Vision (OpenCV)", frame)

        # Frame rate control
        clock.tick(FPS)

        # Exit on 'q' in the OpenCV window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    pygame.quit()
    cv2.destroyAllWindows()
    sys.exit()

if __name__ == "__main__":
    main()
