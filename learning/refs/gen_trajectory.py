import pygame
import math
import numpy as np
import pickle
import os
from pathlib import Path
from DATT.learning.refs.base_ref import BaseRef

RECORD = False
if RECORD:
    from PIL import Image
    import imageio


WIDTH, HEIGHT = 800, 600

class Trajectory(BaseRef):
    def __init__(self, altitude):
        self.altitude = altitude
        self.points = np.empty((0, 3))

    def add_point(self, x, y, t):
        new_point = np.array([[x, y, t]])
        self.points = np.append(self.points, new_point, axis=0)

    def reset(self):
        pass

    def pos(self, t):
        t_points = self.points[:, 2]
        x_points = self.points[:, 0]
        y_points = -self.points[:, 1]

        x = np.interp(t, t_points, x_points)
        y = np.interp(t, t_points, y_points)
        return np.array([x, y, t*0 + self.altitude])

    def vel(self, t):
        delta_t = 0.1
        x1, y1, _ = self.pos(t)
        x2, y2, _ = self.pos(t + delta_t)
        vel_x = (x2 - x1) / delta_t
        vel_y = (y2 - y1) / delta_t
        return np.array([vel_x, vel_y, t*0])

    def acc(self, t):
        delta_t = 0.1
        vel_x1, vel_y1, _ = self.vel(t)
        vel_x2, vel_y2, _ = self.vel(t + delta_t)
        acc_x = (vel_x2 - vel_x1) / delta_t
        acc_y = (vel_y2 - vel_y1) / delta_t
        return np.array([acc_x, acc_y, t*0])
    
    def jerk(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def snap(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yaw(self, t):
        if isinstance(t, np.ndarray):
            y = np.zeros_like(t)
        else:
            return 0
        return y

    def yawvel(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])

    def yawacc(self, t):
        return np.array([
            t*0,
            t*0,
            t*0
        ])


# Run the main program loop
def main_loop(x_min=-3, x_max=3, y_min=-3, y_max=3, saved_traj=None, parent=None, overwrite=False, rerender=False):
    if parent is None:
        parent = Path().absolute() / 'learning' / 'refs'
    print(parent, saved_traj)
    if saved_traj is not None and os.path.exists(parent / f'REF_{saved_traj}') and not overwrite:
        print('LOADING_REF')
        with open(parent / f'REF_{saved_traj}', 'rb') as f:
            trajectory = pickle.load(f)
        if not rerender:
            return trajectory
        else:
            trajectory_loaded = trajectory

    if RECORD:
        frames = []

    # Set the screen size
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    pygame.init()

    # Create an instance of Trajectory
    trajectory = Trajectory(0.0)  # Example altitude value

    running = True
    drawing = False
    start_time = pygame.time.get_ticks()  

    speed_window = []  # Speed window for calculating running average
    speed_window_duration = 0.5  # Duration of the speed window in seconds


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Start drawing trajectory
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                # Stop drawing trajectory
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                # Add points to the trajectory while drawing
                if drawing:
                    x, y = event.pos

                    # Rescale x and y to the range [-3, 3]
                    x = (x / WIDTH) * (x_max - x_min) + x_min
                    y = (y / HEIGHT) * (y_max - y_min) + y_min

                    t = pygame.time.get_ticks() / 1000.0
                    if not rerender:
                        trajectory.add_point(x, y, t)
            elif event.type == pygame.KEYDOWN:
                # Quit if the Q key is pressed
                if event.key == pygame.K_q:
                    running = False
        
        if rerender:
            t = pygame.time.get_ticks() / 1000.0
            x, y, _ = trajectory_loaded.pos(t)
            trajectory.add_point(x, -y, t)

        # Fill the screen with white color
        screen.fill((255, 255, 255))

        # Draw grid lines
        draw_grid_lines(screen, x_min, x_max, y_min, y_max)

        # Draw label at (0, 0) on the grid
        draw_label(screen, "0, 0", (0, 0), (x_min, x_max, y_min, y_max))

        # Draw the trajectory if there are at least two points
        if len(trajectory.points) >= 2:
            pygame.draw.lines(screen, (0, 0, 0), False, [((point[0] - x_min) * WIDTH / (x_max - x_min), (point[1] - y_min) * HEIGHT / (y_max - y_min)) for point in trajectory.points], 2)

        # Draw timer in the top-right corner
        elapsed_time = pygame.time.get_ticks() - start_time
        draw_timer(screen, elapsed_time)

        # Calculate and display current speed in m/s
        if len(trajectory.points) > 0:
            current_speed = calculate_running_average_speed(trajectory, speed_window, speed_window_duration)
        else:
            current_speed = 0.0
        draw_speed(screen, current_speed)

        # Draw reminder to quit in the bottom-right corner
        draw_quit_reminder(screen, "Press Q to finish trajectory", (WIDTH - 10, HEIGHT - 25))
        draw_quit_reminder(screen, "NOTE: trajectory should start at (0, 0)", (WIDTH - 10, HEIGHT - 10))

        # Update the display
        pygame.display.flip()

        if RECORD:
            frame_surface = screen.copy()
            frames.append(pygame.surfarray.array3d(frame_surface))
            # recording_surface = pygame.Surface((WIDTH, HEIGHT))
            # recording_surface.blit(screen, (0, 0))
            # frames.append(pygame.surfarray.array3d(recording_surface))
        clock.tick(60)

    pygame.quit()
    if saved_traj is not None and not rerender:
        print(parent)
        if saved_traj is not None and os.path.isdir(parent):
            print('SAVING REF')
            with open(parent / f'REF_{saved_traj}', 'wb') as f:
                pickle.dump(trajectory, f)

    if RECORD:
        gif_filename = 'api.gif'
        fps = 60
        pil_frames = [Image.fromarray(frame).rotate(90, expand=True) for frame in frames]

        # Save the frames as a GIF using PIL
        pil_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(10 / fps),
            loop=0,
            optimize=True,
            transparency=0,
            quality=1000
        )

 
    return trajectory

def calculate_running_average_speed(trajectory, speed_window, window_duration):
    current_time = pygame.time.get_ticks()
    speed_window.append(trajectory.points[-1])  # Add the latest point to the speed window

    if len(speed_window) == 0:
        return 0

    # Remove points outside the window duration
    while len(speed_window) > 0 and speed_window[0][2] < current_time / 1000 - window_duration:
        speed_window.pop(0)

    if len(speed_window) >= 2:
        total_distance = 0
        total_time = 0

        for i in range(len(speed_window) - 1):
            x_diff = speed_window[i + 1][0] - speed_window[i][0]
            y_diff = speed_window[i + 1][1] - speed_window[i][1]
            distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
            time_diff = speed_window[i + 1][2] - speed_window[i][2]
            total_distance += distance
            total_time += time_diff
 
        average_speed = total_distance / (total_time)
        return average_speed
    else:
        return 0

def draw_speed(screen, speed):
    speed_text = "Speed: {:.2f} m/s".format(speed)
    speed_font = pygame.font.SysFont(None, 20)
    speed_label = speed_font.render(speed_text, True, (0, 0, 0))
    speed_rect = speed_label.get_rect(topright=(WIDTH - 10, 40))
    screen.blit(speed_label, speed_rect)


def draw_label(screen, text, position, bounds):
    x_min, x_max, y_min, y_max = bounds
    x = ((position[0] - x_min) / (x_max - x_min)) * WIDTH
    y = HEIGHT - ((position[1] - y_min) / (y_max - y_min)) * HEIGHT
    label_font = pygame.font.SysFont(None, 20)
    label = label_font.render(text, True, (0, 0, 0))
    label_rect = label.get_rect(center=(x, y))
    screen.blit(label, label_rect)


def draw_grid_lines(screen, x_min, x_max, y_min, y_max):
    # Draw vertical grid lines
    for i in range(int(x_min), int(x_max) + 1):
        x = ((i - x_min) / (x_max - x_min)) * WIDTH
        pygame.draw.line(screen, (200, 200, 200), (x, 0), (x, HEIGHT), 1)

    # Draw horizontal grid lines
    for i in range(int(y_min), int(y_max) + 1):
        y = ((i - y_min) / (y_max - y_min)) * HEIGHT
        pygame.draw.line(screen, (200, 200, 200), (0, y), (WIDTH, y), 1)

def draw_timer(screen, elapsed_time):
    timer_font = pygame.font.SysFont(None, 20)
    timer_text = "Time: {:.2f}s".format(elapsed_time / 1000.0)
    timer_label = timer_font.render(timer_text, True, (0, 0, 0))
    timer_rect = timer_label.get_rect(topright=(WIDTH - 10, 10))
    screen.blit(timer_label, timer_rect)

def draw_quit_reminder(screen, reminder_text, position):
    reminder_font = pygame.font.SysFont(None, 20)
    reminder_label = reminder_font.render(reminder_text, True, (0, 0, 0))
    reminder_rect = reminder_label.get_rect(bottomright=position)
    screen.blit(reminder_label, reminder_rect)



if __name__ == "__main__":
    trajectory = main_loop(saved_traj='test_ref', overwrite=False, rerender=True)

    import matplotlib.pyplot as plt

    t_values = np.linspace(0, 10, 500)

    pos = trajectory.pos(t_values)
    x_values = pos[0]
    y_values = pos[1]

    vel = trajectory.vel(t_values)
    xv = vel[0]
    yv = vel[1]

    # Create subplots for x, y, and z components
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    # Plot x component
    ax1.plot(t_values, x_values)
    ax1.set_ylabel('x')

    # Plot y component
    ax2.plot(t_values, y_values)
    ax2.set_ylabel('y')

    # Plot z component (altitude)
    ax3.axhline(y=trajectory.altitude, color='r', linestyle='--', label='Altitude')
    ax3.set_ylabel('z')
    ax3.legend()

    # Set the x-axis label
    ax3.set_xlabel('Time')

    # Adjust the layout
    plt.tight_layout()

    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    ax1.plot(t_values, xv)
    ax1.set_ylabel('x vel')

    # Plot y component
    ax2.plot(t_values, yv)
    ax2.set_ylabel('y vel')

    # Plot z component (altitude)
    ax3.axhline(y=trajectory.altitude, color='r', linestyle='--', label='Altitude')
    ax3.set_ylabel('z')
    ax3.legend()

    # Set the x-axis label
    ax3.set_xlabel('Time')

    # Show the plot
    plt.show()
