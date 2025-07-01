import cv2

def draw_particles(frame, particles):
    for p in particles:
        cv2.circle(frame, (int(p.x), int(p.y)), 2, (0, 255, 0), -1)

def draw_tracking(frame, centers):
    for (x, y) in centers:
        cv2.circle(frame, (x, y), 20, (255, 0, 0), 2)  # bigger blue circle
        cv2.putText(frame, "Tracked", (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
