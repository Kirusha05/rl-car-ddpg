import json
import math
from typing import Literal
import pygame
import sys
import random

WIDTH, HEIGHT = 1500, 800
FPS = 60
BG_COLOR = (34, 40, 49)

MAX_SENSOR_DISTANCE = 500
START_POS = (100, 470)


class Track:
    def __init__(self):
        self.outer_points = []
        self.inner_points = []
        self.smooth_outer_points = []
        self.smooth_inner_points = []
        self.boundaries = []
        self._load_from_file("track_points3.json")
    
    def _load_from_file(self, filename: str):
        with open(filename, "r") as f:
            track_data = json.load(f)
            # self.smooth_outer_points = track_data["outer_points"]
            self.smooth_outer_points = [
                (x - 40, y + 50)
                for x, y in
                track_data["outer_points"]
            ]
            # self.smooth_inner_points = track_data["inner_points"]
            self.smooth_inner_points = [
                (x - 40, y + 50)
                for x, y in
                track_data["inner_points"]
            ]
            self.boundaries = [self.smooth_outer_points, self.smooth_inner_points]

    # Ray Casting test
    def _point_in_polygon(self, px, py, polygon):
        inside = False
        n = len(polygon)
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


    def is_on_track(self, px, py):
        return (
            self._point_in_polygon(px, py, self.smooth_outer_points)
            and not self._point_in_polygon(px, py, self.smooth_inner_points)
        )

    def _ray_segment_intersect(self, ray_origin, ray_dir, p1, p2):
        """Returns the distance along the ray to intersection with segment p1-p2, or None."""
        ox, oy = ray_origin
        dx, dy = ray_dir
        ax, ay = p1
        bx, by = p2

        # Segment direction
        sx, sy = bx - ax, by - ay

        denom = dx * sy - dy * sx
        if abs(denom) < 1e-9:  # parallel
            return None

        t = ((ax - ox) * sy - (ay - oy) * sx) / denom
        u = ((ax - ox) * dy - (ay - oy) * dx) / denom

        if t >= 0 and 0 <= u <= 1:  # t>=0 means forward along ray, u in [0,1] means on segment
            return t
        return None


    def _cast_ray(self, origin, angle, polygons):
        """Cast a ray, return distance to nearest intersection across all polygon edges."""
        ray_dir = (math.cos(angle), math.sin(angle))
        min_dist = float('inf')

        for polygon in polygons:
            n = len(polygon)
            for i in range(n):
                p1 = polygon[i]
                p2 = polygon[(i + 1) % n]
                dist = self._ray_segment_intersect(origin, ray_dir, p1, p2)
                if dist is not None:
                    min_dist = min(min_dist, dist)

        return min_dist if min_dist != float('inf') else None

    def get_dist_to_track_boundaries(self, pos, angle):
        return self._cast_ray(pos, angle, self.boundaries)

    def draw_track(self, screen):
        # outer track
        if len(self.smooth_outer_points) >= 3:
            pygame.draw.polygon(screen, "black", self.smooth_outer_points, width = 0)

        # inner track
        if len(self.smooth_inner_points) >= 3:
            pygame.draw.polygon(screen, BG_COLOR, self.smooth_inner_points, width = 0)

    # Catmull-Rom sampler
    def _smooth_polygon(self, points, num_points=20):
        n = len(points)
        if n < 3:
            return points
        smooth = []
        for i in range(n):
            p0 = points[(i - 1) % n]
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]
            for j in range(num_points):
                t = j / num_points
                t2, t3 = t**2, t**3
                x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
                y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
                smooth.append((x, y))
        return smooth

    def register_track_points(self, event):
        click_pos = event.dict['pos']

        # Left click
        if event.dict['button'] == 1:
            self.outer_points.append(click_pos)
            if len(self.outer_points) >= 3:
                self.smooth_outer_points = self._smooth_polygon(self.outer_points)

        # Right click
        elif event.dict['button'] == 3:
            self.inner_points.append(click_pos)
            if len(self.inner_points) >= 3:
                self.smooth_inner_points = self._smooth_polygon(self.inner_points)

        with open("track_points4.json", "w") as f:
            track_data = {"outer_points": self.smooth_outer_points, "inner_points": self.smooth_inner_points}
            json.dump(track_data, f, indent=2)


class NeuralNetAnimation:
    max_neurons_per_layer = 32

    def __init__(self, layers: list[int]):
        # ex: layers = [5, 64, 3]
        self.layers = layers
        self.positions = []

    def compute_positions(self, start_x, start_y, layer_spacing, neuron_spacing):
        for layer_idx, layer_size in enumerate(self.layers):
            layer_size = min(layer_size, self.max_neurons_per_layer)
            layer_positions = []

            total_height = (layer_size - 1) * neuron_spacing
            y_offset = start_y - (total_height / 2)

            for neuron_idx in range(min(layer_size, self.max_neurons_per_layer)):
                x = start_x + (layer_spacing * layer_idx) # each layer will move to the right
                y = y_offset + neuron_idx * neuron_spacing

                layer_positions.append((x, y))

            self.positions.append(layer_positions)

    def draw_neurons(self, screen, activations):
        if not activations:
            return
            
        for layer_idx, layer_positions in enumerate(self.positions):
            if layer_idx == 0:
                acts = activations["input"]
            elif layer_idx == 1:
                acts = activations["h1"]
            elif layer_idx == 2:
                acts = activations["h2"]
            else:
                acts = activations["out"]

            neuron_values: list = acts[0].flatten().tolist() if acts.dim() > 1 else acts.flatten().tolist()
            # random.shuffle(neuron_values)

            for neuron_idx, (x, y) in enumerate(layer_positions):
                val = float(neuron_values[neuron_idx])
                intensity = min(255, int(abs(val**3) * 255))  # normalize

                color = (intensity, intensity, 0)
                pygame.draw.circle(screen, color, (int(x), int(y)), 8)
                # pygame.draw.circle(screen, (122, 122, 0), (int(x), int(y)), 6, 1)

    def draw_connections(self, screen, pos1, pos2):
        for j, (x2, y2) in enumerate(pos2):
            for i, (x1, y1) in enumerate(pos1):
                color = (77, 77, 77)

                width = 1

                pygame.draw.line(
                    screen,
                    color,
                    (x1, y1),
                    (x2, y2),
                    width
                )


class ActionSpace:
    def __init__(self, action_min: float, action_max: float):
        self.action_min = action_min
        self.action_max = action_max
        self.n = 1 # only one continuous action

    def sample(self):
        return random.random() * (self.action_max - self.action_min) + self.action_min


class ObservationSpace:
    def __init__(self, for_actor, for_critic):
        self.for_actor = for_actor
        self.for_critic = for_critic


class CarEnv:
    def __init__(self):
        self.velocity = 2
        self.MAX_STEERING_RATE = math.pi * 1.5
        self.MAX_STEERING_ANGLE = 8*math.pi
        self.steering_angle = 0

        self.car_width = 30
        self.car_height = 10
        self.track = Track()
        self.render_mode = False
        self.episode = 0
        self.FPS = FPS

        # state consisting of 5 sensors + steering_angle and 3 possible actions
        self.observation_space = ObservationSpace(6, 7)
        self.action_space = ActionSpace(-1, 1)

        self.reset()

    def reset(self):
        self.x, self.y = START_POS
        self.angle = -math.pi / 2 # 90 deg
        self.steering_angle = 0
        self.done = False
        self.steps = 0

        _, sensors, steering_angle = self._get_observations()
        state = [*sensors, steering_angle]
        return state

    def step(self, action: float, truncate_at=3000):
        self.steps += 1
        
        # 1. Apply action
        self._apply_action(action)

        # 2. Update physics
        self._update_position()

        # 3. Get observations
        car_is_on_track, sensors, steering_angle = self._get_observations()

        # 4. Compute rewards
        reward, self.done = self._compute_reward(car_is_on_track, sensors)

        # Truncate at 3000 max steps. If it can survive for so many steps, it should have learned a pretty good policy
        truncated = False
        if self.steps > truncate_at:
            truncated = True

        # return observations/state, reward, done, truncated, info
        state = [*sensors, steering_angle]
        return state, reward, self.done, truncated, {}
    
    # steering_velocity = [-1, 1]
    def _apply_action(self, steering_velocity: float):
        dt = 0.1
        steering_rate = steering_velocity * self.MAX_STEERING_RATE
        self.steering_angle += steering_rate * dt
        if abs(self.steering_angle) > self.MAX_STEERING_ANGLE:
            self.steering_angle = math.copysign(self.MAX_STEERING_ANGLE, self.steering_angle)

    def _get_axis_velocities(self, angle, velocity):
        dx = math.cos(angle) * velocity
        dy = math.sin(angle) * velocity
        return dx, dy

    def _update_position(self):
        self.angle += self.steering_angle * 0.005
        dx, dy = self._get_axis_velocities(self.angle, self.velocity)
        self.x += dx
        self.y += dy

    def _get_rect_corners(self, x, y, width, height, angle):
        hw = width / 2
        hh = height / 2

        corners_local = [
            (-hw, -hh),
            ( hw, -hh),
            ( hw,  hh),
            (-hw,  hh)
        ]

        corners_world = []

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        for lx, ly in corners_local:
            wx = x + lx * cos_a - ly * sin_a
            wy = y + lx * sin_a + ly * cos_a
            corners_world.append((wx, wy))

        return corners_world

    def _get_car_corners(self):
        return self._get_rect_corners(self.x, self.y, self.car_width, self.car_height, self.angle)

    def _normalize_dist(self, dist):
            if dist is None:
                return 1.0
            return min(dist, MAX_SENSOR_DISTANCE) / MAX_SENSOR_DISTANCE

    def _get_observations(self):
        car_corners = self._get_car_corners()
        corners_are_on_track = [self.track.is_on_track(cx, cy) for cx, cy in car_corners]
        car_is_on_track = all(corners_are_on_track)

        car_pos = (self.x, self.y)
        forward_dist = self.track.get_dist_to_track_boundaries(car_pos, self.angle)
        left_dist = self.track.get_dist_to_track_boundaries(car_pos, self.angle - math.pi / 2)  # 90 deg left
        right_dist = self.track.get_dist_to_track_boundaries(car_pos, self.angle + math.pi / 2)  # 90 deg right
        front_left_dist = self.track.get_dist_to_track_boundaries(car_pos, self.angle - math.pi / 4)  # 45 deg left
        front_right_dist = self.track.get_dist_to_track_boundaries(car_pos, self.angle + math.pi / 4)  # 45 deg right        

        # space ordered, ChatGPT says that spatial ordering matters for neural nets
        sensors = [
            self._normalize_dist(left_dist),
            self._normalize_dist(front_left_dist),
            self._normalize_dist(forward_dist),
            self._normalize_dist(front_right_dist),
            self._normalize_dist(right_dist)
        ]

        steering_angle_normalized = self.steering_angle / self.MAX_STEERING_ANGLE

        return car_is_on_track, sensors, steering_angle_normalized
    
    # will return reward, is_done
    def _compute_reward(self, car_is_on_track, sensors) -> (int, bool):
        if not car_is_on_track:
            # penalty for boom boom
            return -10.0, True

        # promote centered driving, meaning left dist and right dist should be pretty much equal
        left_dist, _, forward_dist, _, right_dist = sensors
        diff = abs(left_dist - right_dist)
        complement = 1 - diff  # the greater the diff, the smaller the reward
        return  0.5 * forward_dist + 0.5 * complement, False
        
        # small reward for surviving
        # return 0.1, False

    def _draw_car(self, corners, color):
        pygame.draw.polygon(self.screen, color, corners)

    def _draw_steering_wheel(self):
        original_image = pygame.image.load("steering_wheel.png").convert_alpha()
        image_center = (800, 200)

        scaled_img = pygame.transform.smoothscale(original_image, (250, 250))

        rotated_surface = pygame.transform.rotate(scaled_img, -self.steering_angle * 20)
        new_rect = rotated_surface.get_rect(center=scaled_img.get_rect(center=image_center).center)
        self.screen.blit(rotated_surface, new_rect)

    def init_neural_net(self, layers):
        self.neural_net = NeuralNetAnimation(layers)
        self.neural_net.compute_positions(950, 400, 120, 20)

    def animate_neural_net(self, activations):
        if self.render_mode:
            for pos_idx in range(len(self.neural_net.positions) - 1):
                self.neural_net.draw_connections(
                    self.screen, 
                    self.neural_net.positions[pos_idx], 
                    self.neural_net.positions[pos_idx+1]
                )
                
            self.neural_net.draw_neurons(self.screen, activations)
            pygame.display.flip()
            self.clock.tick(self.FPS)

    def init_display(self):
        if not self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
            pygame.display.set_caption("RL Car")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 20)
            self.render_mode = True

    def close_display(self):
        if self.render_mode:
            pygame.display.quit()  # This explicitly kills the window
            pygame.quit()          # This uninitializes PyGame
            self.render_mode = False

    def render(self):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_LEFT]:
        #     self._apply_action(-1)
        # elif keys[pygame.K_RIGHT]:
        #     self._apply_action(1)

        # self._update_position()

        self.screen.fill(BG_COLOR)

        self.track.draw_track(self.screen)
        self._draw_car(self._get_car_corners(), "orange")
        self._draw_steering_wheel()
        
        # Divider
        # pygame.draw.line(
        #     self.screen,
        #     (0,0,0),
        #     (750, 0),
        #     (750, 800),
        #     1
        # )
        
        # Draw info text
        left = 250
        top = 450

        text_surface = self.font.render(f"Steps: {self.steps}", True, (255, 255, 255))
        self.screen.blit(text_surface, (left, top))

        if self.episode is not None:
            text_surface = self.font.render(f"Episode: {self.episode}", True, (255, 255, 255))
            self.screen.blit(text_surface, (left, top - 100))
            text_surface = self.font.render("Exploration: ON", True, (255, 255, 255))
            self.screen.blit(text_surface, (left, top - 70))
        else:
            text_surface = self.font.render("Training done", True, (255, 255, 255))
            self.screen.blit(text_surface, (left, top - 100))
            text_surface = self.font.render("Exploration: OFF", True, (255, 255, 255))
            self.screen.blit(text_surface, (left, top - 70))


        # pygame.display.flip()
        # self.clock.tick(FPS)


if __name__ == "__main__":
    # Example loop to be used for RL
    # Allows you to run computations off-screen and ocasionally render the game to check the progress
    env = CarEnv()
    env.init_neural_net([5, 16, 16, 3])

    act = ActionSpace(-1, 1)

    episodes = 500

    for episode in range(episodes):
        print(f"Episode: {episode}")

        # Determine if we should render THIS episode
        # render_this_episode = (30 <= episode <= 40) or (200 <= episode <= 210)
        render_this_episode = True

        if render_this_episode:
            env.init_display()
        else:
            env.close_display()

        state = env.reset()

        done = False
        while not done:
            # Render if needed
            if render_this_episode:
                env.episode = episode
                env.render()

            # Take action
            state, reward, done, truncated, info = env.step(act.sample())

            env.animate_neural_net(None)


# pygame.init()
# screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED)
# pygame.display.set_caption("RL Car")
# clock = pygame.time.Clock()

# outer_points = []
# smooth_outer_points = []
# inner_points = []
# smooth_inner_points = []


# font = pygame.font.SysFont("Arial", 12)


# with open("track_points.json", "r") as f:
#     track_data = json.load(f)
#     smooth_outer_points = track_data["outer_points"]
#     smooth_inner_points = track_data["inner_points"]


# # Catmull-Rom sampler
# def smooth_polygon(points, num_points=20):
#     n = len(points)
#     if n < 3:
#         return points
#     smooth = []
#     for i in range(n):
#         p0 = points[(i - 1) % n]
#         p1 = points[i]
#         p2 = points[(i + 1) % n]
#         p3 = points[(i + 2) % n]
#         for j in range(num_points):
#             t = j / num_points
#             t2, t3 = t**2, t**3
#             x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
#             y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
#             smooth.append((x, y))
#     return smooth

# # Ray Casting test
# def point_in_polygon(px, py, polygon):
#     inside = False
#     n = len(polygon)
#     j = n - 1
#     for i in range(n):
#         xi, yi = polygon[i]
#         xj, yj = polygon[j]
#         if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
#             inside = not inside
#         j = i
#     return inside


# def is_on_track(px, py):
#     return (point_in_polygon(px, py, smooth_outer_points)
#         and not point_in_polygon(px, py, smooth_inner_points))


# def register_track_points(event):
#     global smooth_outer_points
#     global smooth_inner_points

#     click_pos = event.dict['pos']

#     # Left click
#     if event.dict['button'] == 1:
#         outer_points.append(click_pos)
#         if len(outer_points) >= 3:
#             smooth_outer_points = smooth_polygon(outer_points)

#     # Right click
#     elif event.dict['button'] == 3:
#         inner_points.append(click_pos)
#         if len(inner_points) >= 3:
#             smooth_inner_points = smooth_polygon(inner_points)

#     with open("track_points.json", "w") as f:
#         track_data = {"outer_points": smooth_outer_points, "inner_points": smooth_inner_points}
#         json.dump(track_data, f, indent=2)


# def check_if_clicked_on_track(event):
#     click_pos = event.dict['pos']
#     x, y = click_pos
#     print("On track:", is_on_track(x, y))


# def register_click(event):
#     click_pos = event.dict['pos']
#     print(click_pos)


# def draw_outer_track():
#     if len(smooth_outer_points) >= 3:
#         pygame.draw.polygon(screen, "black", smooth_outer_points, width = 0)

# def draw_inner_track():
#     if len(smooth_inner_points) >= 3:
#         pygame.draw.polygon(screen, BG_COLOR, smooth_inner_points, width = 0)


# # Boundaries
# def ray_segment_intersect(ray_origin, ray_dir, p1, p2):
#     """Returns the distance along the ray to intersection with segment p1-p2, or None."""
#     ox, oy = ray_origin
#     dx, dy = ray_dir
#     ax, ay = p1
#     bx, by = p2

#     # Segment direction
#     sx, sy = bx - ax, by - ay

#     denom = dx * sy - dy * sx
#     if abs(denom) < 1e-9:  # parallel
#         return None

#     t = ((ax - ox) * sy - (ay - oy) * sx) / denom
#     u = ((ax - ox) * dy - (ay - oy) * dx) / denom

#     if t >= 0 and 0 <= u <= 1:  # t>=0 means forward along ray, u in [0,1] means on segment
#         return t
#     return None


# def cast_ray(origin, angle, polygons):
#     """Cast a ray, return distance to nearest intersection across all polygon edges."""
#     ray_dir = (math.cos(angle), math.sin(angle))
#     min_dist = float('inf')

#     for polygon in polygons:
#         n = len(polygon)
#         for i in range(n):
#             p1 = polygon[i]
#             p2 = polygon[(i + 1) % n]
#             dist = ray_segment_intersect(origin, ray_dir, p1, p2)
#             if dist is not None:
#                 min_dist = min(min_dist, dist)

#     return min_dist if min_dist != float('inf') else None


# def draw_ray(origin, angle, dist, color="yellow"):
#     if dist is None:
#         return

#     ox, oy = origin
#     end_x = ox + math.cos(angle) * dist
#     end_y = oy + math.sin(angle) * dist

#     pygame.draw.line(screen, color, origin, (end_x, end_y), 2)


# car_start_pos = (795, 469)
# general_velocity = 100
# car_angle = -math.pi / 2 # 90 deg
# car_angle_d = math.pi / 15 # 6 deg
# car_width = 30
# car_height = 10

# def get_axis_velocities(angle, velocity):
#     dx = math.cos(angle) * velocity
#     dy = math.sin(angle) * velocity
#     return dx, dy


# def get_car_corners(x, y, width, height, angle):
#     hw = width / 2
#     hh = height / 2

#     corners_local = [
#         (-hw, -hh),
#         ( hw, -hh),
#         ( hw,  hh),
#         (-hw,  hh)
#     ]

#     corners_world = []

#     cos_a = math.cos(angle)
#     sin_a = math.sin(angle)

#     for lx, ly in corners_local:
#         wx = x + lx * cos_a - ly * sin_a
#         wy = y + lx * sin_a + ly * cos_a
#         corners_world.append((wx, wy))

#     return corners_world


# x, y = car_start_pos
# dx, dy = get_axis_velocities(car_angle, general_velocity)


# def draw_car(corners, color):
#     pygame.draw.polygon(screen, color, corners)

# def steer_left():
#     global car_angle
#     car_angle -= car_angle_d

# def steer_right():
#     global car_angle
#     car_angle += car_angle_d


# dt = 1

# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             sys.exit()
        
#         if event.type == pygame.MOUSEBUTTONDOWN:
#             # register_track_points(event)
#             # check_if_clicked_on_track(event)
#             register_click(event)

#         keys = pygame.key.get_pressed()
#         if keys[pygame.K_LEFT]:
#             steer_left()
#         elif keys[pygame.K_RIGHT]:
#             steer_right()

#     dx, dy = get_axis_velocities(car_angle, general_velocity)

#     screen.fill(BG_COLOR)  # background color

#     # --- draw stuff here ---
#     draw_outer_track()
#     draw_inner_track()

#     car_corners = get_car_corners(x, y, car_width, car_height, car_angle)
#     corners_are_on_track = [is_on_track(cx, cy) for cx, cy in car_corners]
#     car_is_on_track = all(corners_are_on_track)

#     car_color = "blue"
#     if not car_is_on_track:
#         car_color = "red"

#     # sensor info
#     boundaries = [smooth_outer_points, smooth_inner_points]

#     def get_dist(angle):
#         return cast_ray((x, y), car_angle + angle, boundaries)

#     forward_dist = get_dist(angle=0)
#     left_dist = get_dist(angle=-math.pi / 2)  # 90 deg left
#     right_dist = get_dist(angle=math.pi / 2)  # 90 deg right
#     front_left_dist = get_dist(angle=-math.pi / 4)  # 45 deg left
#     front_right_dist = get_dist(angle=math.pi / 4)  # 45 deg right

#     def normalize_dist(dist):
#         if dist is None:
#             return 1.0
#         return dist / MAX_SENSOR_DISTANCE if dist is not None else dist

#     sensors = {
#         "forward": normalize_dist(forward_dist),
#         "left": normalize_dist(left_dist),
#         "right": normalize_dist(right_dist),
#         "front_left": normalize_dist(front_left_dist),
#         "front_right":  normalize_dist(front_right_dist),
#     }

#     draw_ray((x, y), car_angle, cast_ray((x, y), car_angle, boundaries))
#     draw_ray((x, y), car_angle - math.pi / 2, cast_ray((x, y), car_angle - math.pi / 2, boundaries))
#     draw_ray((x, y), car_angle + math.pi / 2, cast_ray((x, y), car_angle + math.pi / 2, boundaries))
#     draw_ray((x, y), car_angle - math.pi / 4, cast_ray((x, y), car_angle - math.pi / 4, boundaries))
#     draw_ray((x, y), car_angle + math.pi / 4, cast_ray((x, y), car_angle + math.pi / 4, boundaries))

#     # draw sensor info
#     for idx, sensor in enumerate(sensors):
#         text_surface = font.render(f"{sensor}: {sensors[sensor]}", True, (255, 255, 255))
#         screen.blit(text_surface, (20, 20 * (idx + 1)))

#     # draw car
#     draw_car(car_corners, car_color)

#     x += dx * dt
#     y += dy * dt

#     # Display and wait a bit by corresponding FPS
#     pygame.display.flip()
#     dt = clock.tick(FPS) / 1000.0