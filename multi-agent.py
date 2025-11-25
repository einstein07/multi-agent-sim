import csv
import json
import os
import random
import math
import time
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --------------------------------------------------------------------
# 1. States
# --------------------------------------------------------------------
NO_TURN   = "NO_TURN"
SOFT_TURN = "SOFT_TURN"
HARD_TURN = "HARD_TURN"

class WheelTurningParams:
    def __init__(self, turning_mechanism,
                 BaseSpeed,
                 hard_turn_on_angle_threshold,
                 soft_turn_on_angle_threshold,
                 no_turn_angle_threshold):
        self.BaseSpeed = BaseSpeed
        self.turning_mechanism = turning_mechanism
        self.HardTurnOnAngleThreshold = hard_turn_on_angle_threshold
        self.SoftTurnOnAngleThreshold = soft_turn_on_angle_threshold
        self.NoTurnAngleThreshold = no_turn_angle_threshold


class Agent:
    """Represents a single agent in the simulation -- with occlusion capabilities."""

    def __init__(self, id, x, y, speed, track_width, direction, commitment, eta, light_ids,
                 fov, update_offset, thresholds, turning_mechanism, radius, all_agents=None):
        self.id = id
        self.x = x
        self.y = y
        self.speed = speed
        self.track_width = track_width
        self.direction = direction  # Initial direction in radians
        self.commitment = commitment
        self.eta = eta
        self.light_ids = light_ids
        self.fov = fov
        self.trajectory = [(x, y)]
        self.received_broadcasts = {}
        self.update_offset = update_offset
        self.broadcast = None
        self.my_opinions = []
        self.number_of_neighbors = []
        self.ids_of_neighbors = []
        self.visible_agents = []
        self.visible_targets = []
        self.log_file = None
        self.radius = radius  # New attribute for agent radius
        self.all_agents = all_agents  # Reference to all agents for occlusion checks
        self.wheel_turning_params = WheelTurningParams(
            BaseSpeed=self.speed,
            turning_mechanism=turning_mechanism,
            hard_turn_on_angle_threshold=thresholds['hard'],
            soft_turn_on_angle_threshold=thresholds['soft'],
            no_turn_angle_threshold=thresholds['none']
        )
        self.is_within_termination_radius = False  # Track if agent is within termination radius
        self.has_reached_target = False  # Track if agent has reached the target

    def initialize_log_file(self, simulation_start_time, run_folder, experiment_name):
        """Initialize the agent's log file."""
        filename = os.path.join(run_folder, f"{experiment_name}_bot{self.id}_{simulation_start_time}.csv")
        self.log_file = open(filename, "w", newline="")
        writer = csv.writer(self.log_file)
        writer.writerow(["Time", "Commitment", "Opinion", "Neighbors", "LightSources", "VisibleAgents", "IDsOfNeighbors"])
        self.csv_writer = writer

    def close_log_file(self):
        """Close the agent's log file."""
        if self.log_file:
            self.log_file.close()

    def normalize_angle(sef, angle):
        """
        Returns angle in [-pi, pi].
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle <= -math.pi:
            angle += 2.0 * math.pi
        return angle

    # --------------------------------------------------------------------
    # State Transition Function
    # --------------------------------------------------------------------
    def update_turning_mechanism(self, c_heading_angle, params):
        #print('init TM: ', params.turning_mechanism)
        """
        Update the turning mechanism based on the absolute heading angle
        and threshold values, replicating the logic from the original C++ code.

        :param current_mechanism: One of (NO_TURN, SOFT_TURN, HARD_TURN)
        :param c_heading_angle: (float) Current heading angle difference we need to correct
        :param params: WheelTurningParams instance
        :return: new_mechanism (string)
        """
        abs_angle = abs(c_heading_angle)

        # The original code did these checks in sequence.
        # We replicate that structure:

        # 1) If currently HARD_TURN, check possible switch to SOFT_TURN
        if params.turning_mechanism == HARD_TURN:
            if abs_angle <= params.SoftTurnOnAngleThreshold:
                params.turning_mechanism = SOFT_TURN

        # 2) If currently SOFT_TURN, check possible switch to HARD_TURN or NO_TURN
        if params.turning_mechanism == SOFT_TURN:
            if abs_angle > params.HardTurnOnAngleThreshold:
                params.turning_mechanism = HARD_TURN
            elif abs_angle <= params.NoTurnAngleThreshold:
                params.turning_mechanism = NO_TURN

        # 3) If currently NO_TURN, check possible switch to HARD_TURN or SOFT_TURN
        if params.turning_mechanism == NO_TURN:
            if abs_angle > params.HardTurnOnAngleThreshold:
                params.turning_mechanism = HARD_TURN
            elif abs_angle > params.NoTurnAngleThreshold:
                params.turning_mechanism = SOFT_TURN
        #print('new turning mechanism: ', params.turning_mechanism)
        return params.turning_mechanism

    # --------------------------------------------------------------------
    # Compute Wheel Speeds for Each Mechanism
    # --------------------------------------------------------------------
    def compute_wheel_speeds(self, turning_mechanism, c_heading_angle, c_heading_magnitude, params):
        """
        Given the turning mechanism (NO_TURN, SOFT_TURN, HARD_TURN),
        compute the left/right wheel linear speeds in m/s.

        :param turning_mechanism: One of (NO_TURN, SOFT_TURN, HARD_TURN)
        :param c_heading_angle: The heading angle difference to correct
        :param params: WheelTurningParams
        :return: (v_left, v_right) in m/s
        """
        abs_angle = abs(c_heading_angle)
        # Clamp the speed so that it's not greater than MaxSpeed
        # Spoke with Giovanni about this, and he said that the speed is better kept constant
        #fBaseAngularWheelSpeed = min(c_heading_magnitude/100, params.BaseSpeed)
        fBaseAngularWheelSpeed = params.BaseSpeed
        #print('TM, ANGLE, PARAMS: ', turning_mechanism, c_heading_angle, params)
        if turning_mechanism == NO_TURN:
            # Both wheels run at the same base speed => go straight
            fSpeed1 = fBaseAngularWheelSpeed
            fSpeed2 = fBaseAngularWheelSpeed

        elif turning_mechanism == HARD_TURN:
            # Turn in place: left = -MaxSpeed, right = +MaxSpeed (pivot turn)
            fSpeed1 = -fBaseAngularWheelSpeed
            fSpeed2 = fBaseAngularWheelSpeed

        elif turning_mechanism == SOFT_TURN:
            # Turn while moving forward
            # Reproduce the "soft turn" logic from your snippet:
            #   fSpeedFactor = (HardTurnOnAngleThreshold - abs_angle) / HardTurnOnAngleThreshold

            fSpeedFactor = (params.HardTurnOnAngleThreshold - abs_angle) / params.HardTurnOnAngleThreshold

            # For clarity:
            #   fSpeed1 = base - base * (1 - factor) = base * factor
            #   fSpeed2 = base + base * (1 - factor) = base * (2 - factor)
            # We'll treat left = fSpeed1, right = fSpeed2
            fSpeed1 = fBaseAngularWheelSpeed * fSpeedFactor
            fSpeed2= fBaseAngularWheelSpeed * (2.0 - fSpeedFactor)

        else:
            # Default fallback (should not happen if code is correct)
            fSpeed1 = 0.0
            fSpeed2 = 0.0
            print('something wrong. turning mech: ', turning_mechanism)
        fLeftWheelSpeed = 0
        fRightWheelSpeed = 0
        if c_heading_angle > 0:
            #Turn Left * /
            fLeftWheelSpeed  = fSpeed1
            fRightWheelSpeed = fSpeed2
        else:
            #Turn Right * /
            fLeftWheelSpeed  = fSpeed2
            fRightWheelSpeed = fSpeed1

        return fLeftWheelSpeed, fRightWheelSpeed

    # --------------------------------------------------------------------
    # Differential Drive Pose Update
    # --------------------------------------------------------------------
    def update_pose(self, x, y, theta, v_left, v_right, track_width, dt):
        """
        Given the current pose (x, y, theta) and the left/right wheel speeds,
        update the robot's pose over a time step dt using the standard
        differential drive equations.

        :param x:       current x position
        :param y:       current y position
        :param theta:   current heading (radians)
        :param v_left:  left wheel linear speed (m/s)
        :param v_right: right wheel linear speed (m/s)
        :param track_width: distance between the two wheels (m)
        :param dt:      time step (s)
        :return: (x_new, y_new, theta_new)
        """

        # 1) Forward speed of the robot's center
        v = 0.5 * (v_left + v_right)

        # 2) Yaw rate
        dot_theta = (v_right - v_left) / track_width

        # 3) Update heading
        theta_new = theta + dot_theta * dt

        # 4) Update position
        x_new = x + v * math.cos(theta) * dt
        y_new = y + v * math.sin(theta) * dt

        return x_new, y_new, theta_new

    def update_position(self, target_x, target_y, dt):
        """Update the agent's position based on movement rules."""
        target_distance = math.sqrt((target_x - self.x)**2 + (target_y - self.y)**2)
        """TODO: Better way to check if agent has reached the target."""
        if target_distance <= 0.1:  # Agent has reached the target
            self.has_reached_target = True
            return
        target_direction = math.atan2(target_y - self.y, target_x - self.x)
        
        angle_diff = (target_direction - self.direction + math.pi) % (2 * math.pi) - math.pi
        #print('angle diff: ', angle_diff)
        # (A) Update turning mechanism
        self.wheel_turning_params.turning_mechanism = self.update_turning_mechanism(
            angle_diff,
            self.wheel_turning_params
        )
        # (B) Compute wheel speeds for that mechanism
        v_left, v_right = self.compute_wheel_speeds(
            self.wheel_turning_params.turning_mechanism,
            angle_diff,
            target_distance,
            self.wheel_turning_params
        )
        # (C) Do the differential-drive pose update
        self.x, self.y, self.direction = self.update_pose(
            self.x,
            self.y, self.direction,
            v_left,
            v_right,
            self.track_width,
            dt
        )

        self.trajectory.append((self.x, self.y))

    def determine_broadcast(self, hard_turn_threshold, angle_diff):
        """Determine the broadcast value based on turn type."""
        self.broadcast = 0 if abs(angle_diff) > hard_turn_threshold else self.commitment
        #if self.id == 0:
        #    print("ID: ", self.id, " Commitment: ", self.commitment, " broadcast: ", self.broadcast)
        self.my_opinions.append(self.broadcast)

    def receive_broadcast(self, sender_id, broadcast_value):
        """Store received broadcasts only if different from previous."""
        if self.received_broadcasts.get(sender_id, None) != broadcast_value:
            self.received_broadcasts[sender_id] = broadcast_value

    def update_commitment(self, light_sources, occlusion):
        if self.is_within_termination_radius:
            return  # Do not update commitment if within termination radius
        if occlusion:
            # If occlusion is enabled, append perceived light sources
            # to the list of received social broadcasts/information
            visible_targets_ids = []
            for light_id in self.light_ids:
                # Check if light source is within FOV
                if self.is_in_fov(light_sources[light_id], self.fov, occlusion, check_type="edge"):
                    visible_targets_ids.append(light_id)
            completeInputs = list(self.received_broadcasts.values())
            completeInputs.extend(visible_targets_ids)  # Changed from append to extend
            #if visible_targets_ids:
            #print("ID: ", self.id, " Received broadcasts: ", self.received_broadcasts, " Visible targets: ", visible_targets_ids)
            if len(completeInputs) > 0:
                newCommitment = random.choice(completeInputs)
            else:
                newCommitment = 0
            if newCommitment != 0:
                self.commitment = newCommitment

        else:
            if random.random() < self.eta:
                """Update the agent's commitment based on perception."""
                #self.commitment = random.choice(self.light_ids)
                # Filter available targets that lie within the FOV
                visible_targets = []
                for light_id in self.light_ids:
                    #print(light_sources)
                    # Check if light source is within FOV
                    if self.is_in_fov(light_sources[light_id], self.fov, occlusion):
                        #visible_targets.append(light_id)
                        visible_targets.append(light_sources[light_id])
                    """else:
                        print('Light ', light_id, ' not in fov')"""

                if visible_targets:  # Only choose if there's at least one visible light
                    # before quality consideration
                    #self.commitment = random.choice(visible_targets)
                    # Extract quality values as weights
                    weights = [target.quality for target in visible_targets]
                    # Select one LightSource object based on weights
                    selected_target = random.choices(visible_targets, weights=weights, k=1)[0]
                    # Assign the id of the selected light source to self.commitment
                    self.commitment = selected_target.id
            else:
                """Update the agent's commitment based on received messages."""
                if self.received_broadcasts:
                    #valid_commitments = [v for v in self.received_broadcasts.values() if v != 0]
                    newCommitment = random.choice(list(self.received_broadcasts.values()))
                    if newCommitment != 0:
                        self.commitment = newCommitment
        self.received_broadcasts.clear()

    def agent_log_data(self, time_step):
        """Log the agent's current state."""
        opinions = ";".join(map(str, self.my_opinions))
        neighbors = ";".join(map(str, self.number_of_neighbors))
        ids_of_neighbors = ";".join(map(str, self.ids_of_neighbors))
        visible_targets = ";".join(map(str, self.visible_targets))
        visible_agents = ";".join(map(str, self.visible_agents))
        # Log the data to the CSV file
        self.csv_writer.writerow([time_step, self.commitment, opinions, neighbors, visible_targets, visible_agents, ids_of_neighbors])
        self.my_opinions.clear()
        self.number_of_neighbors.clear()
        self.ids_of_neighbors.clear()
        self.visible_targets.clear()
        self.visible_agents.clear()

    def is_in_fov(self, light_source, fov, occlusion, check_type="center"):
        """
        Returns True if light_source is within FOV and not occluded.
        check_type: 'center' for line from agent's center, 'edge' for line from agent's near edge.
        """

        if check_type not in ["center", "edge"]:
            raise ValueError("check_type must be 'center' or 'edge'")
        dx = light_source.x - self.x
        dy = light_source.y - self.y

        # Angle to the light source
        angle_to_light = math.atan2(dy, dx)

        # Signed difference between agent's direction and angle to light
        angle_diff = (angle_to_light - self.direction + math.pi) % (2 * math.pi) - math.pi

        if abs(angle_diff) > fov:
            return False

        if occlusion:
            # Step 2: Check for occlusion by other agents
            if check_type == "edge":
                # Line from agent's near edge to light source
                direction = np.array([light_source.x - self.x, light_source.y - self.y])
                dist = np.linalg.norm(direction)
                if dist == 0:
                    return True
                unit_direction = direction / dist
                p1 = np.array([self.x, self.y]) - self.radius * unit_direction
                p2 = np.array([light_source.x, light_source.y])
                for other in self.all_agents:
                    if other.id == self.id:
                        continue
                    # Check if line segment p1-p2 intersects other's circle
                    c = np.array([other.x, other.y])
                    r = other.radius
                    d = p2 - p1
                    seg_len = np.linalg.norm(d)
                    if seg_len == 0:
                        continue
                    d_unit = d / seg_len
                    f = c - p1
                    t = np.dot(f, d_unit)
                    if t < 0 or t > seg_len:
                        continue
                    closest = p1 + t * d_unit
                    dist_to_center = np.linalg.norm(c - closest)
                    if dist_to_center < r:
                        return False
            else:
                # Original center-based check
                for other in self.all_agents:
                    if other.id == self.id:
                        continue
                    if self.is_occluded(light_source, other, check_type="center"):
                        return False

        return True
    
    def is_occluded(self, target, other_agent, check_type="center"):
        """
        Returns True if 'other_agent' occludes the line of sight to 'target'.
        check_type: 'center' for line to target's center, 'edge' for line to target's near edge.
        """
        # Determine p1 (viewer position)
        if isinstance(target, LightSource) and check_type == "edge":
            # Line from agent's near edge to light source
            direction = np.array([target.x - self.x, target.y - self.y])
            dist = np.linalg.norm(direction)
            if dist == 0:
                return False
            unit_direction = direction / dist
            p1 = np.array([self.x, self.y]) - self.radius * unit_direction
        else:
            # Default: line from agent's center
            p1 = np.array([self.x, self.y])

        # Determine p2 (target position)
        if isinstance(target, Agent) and check_type == "edge":
            # Line to target's near edge
            direction = np.array([target.x - self.x, target.y - self.y])
            dist = np.linalg.norm(direction)
            if dist == 0:
                return False
            unit_direction = direction / dist
            p2 = np.array([target.x, target.y]) - target.radius * unit_direction
        else:
            # Line to target's center (agent or light source)
            p2 = np.array([target.x, target.y])
        
        # Center of the other agent
        c = np.array([other_agent.x, other_agent.y])
        r = other_agent.radius

        # Vector from p1 to p2
        d = p2 - p1
        # Vector from p1 to circle center
        f = c - p1
        # Length of the line segment
        seg_len = np.linalg.norm(d)
        if seg_len == 0:
            return False

        # Normalize direction vector
        d_unit = d / seg_len
        # Project f onto d
        t = np.dot(f, d_unit)
        # Closest point on line segment
        if t < 0:
            closest = p1
        elif t > seg_len:
            closest = p2
        else:
            closest = p1 + t * d_unit

        # Distance from circle center to closest point
        dist = np.linalg.norm(c - closest)
        # Occlusion occurs if distance is less than radius
        return dist < r
    
    def get_visible_agents(self):
        #self.visible_agents = []
        for other in self.all_agents:
            if other.id == self.id:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            angle_to_agent = math.atan2(dy, dx)
            angle_diff = (angle_to_agent - self.direction + math.pi) % (2 * math.pi) - math.pi
            if abs(self.fov - 2 * math.pi) > 1e-6 and abs(angle_diff) > self.fov:
                continue
            distance = math.hypot(dx, dy)
            if distance < 2 * self.radius:
                if other.id not in self.visible_agents:
                    self.visible_agents.append(other.id)
                continue
            occluded = False
            for occluder in self.all_agents:
                if occluder.id == self.id or occluder.id == other.id:
                    continue
                if self.is_occluded(other, occluder, check_type="edge"):
                    occluded = True
                    break
            if not occluded:
                if other.id not in self.visible_agents:
                    self.visible_agents.append(other.id)

class LightSource:
    """Represents a target light source in the environment."""

    def __init__(self, id, x, y, color, quality):
        self.id = id
        self.x = x
        self.y = y
        self.color = color
        self.quality = quality


class Simulation:
    """Main simulation controller."""

    def __init__(self, config):
        self.start_time = datetime.now().strftime("%Y%m%d-%Hh%Mm%Ss%fus")
        self.config = config
        self.validate_config()
        self.initialize_from_config()
        self.agents = []
        self.history = []
        self.position_log = None

    def validate_config(self):
        """Ensure all required configuration parameters are present."""
        required_keys = ['num_agents', 'x_width', 'y_height', 'center',
                         'time_steps', 'light_sources', 'init_robot_bounds',
                         'robots_speed', 'track_width', 'eta', 'robots_direction',
                         'no_turn_threshold', 'soft_turn_threshold', 'hard_turn_threshold',
                         'steps_per_second', 'termination_radius', 'commitment_update_time',
                         'robot_radius', 'occlusion']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def initialize_from_config(self):
        """Initialize simulation parameters from config."""
        # Light sources
        self.light_sources = {light['id']: LightSource(**light)
                              for light in self.config['light_sources']}
        self.light_ids = list(self.light_sources.keys())
        self.fov = math.radians(self.config['fov'])
        self.communication_range = float('inf') if self.config.get('communication_range', -1) == -1 else self.config['communication_range']

        # Simulation parameters
        self.num_agents = self.config['num_agents']
        self.num_runs = self.config.get('num_runs', 1)
        self.base_log_dir = self.config.get('log_directory', './logs')
        self.experiment_name = self.config.get('experiment_name', 'experiment')

        # Movement thresholds
        self.thresholds = {
            'none': math.radians(self.config['no_turn_threshold']),
            'soft': math.radians(self.config['soft_turn_threshold']),
            'hard': math.radians(self.config['hard_turn_threshold'])
        }

        # Timing control
        self.steps_per_second = self.config['steps_per_second']
        self.step_duration = 1.0 / self.steps_per_second

        # Agent parameters
        self.track_width = self.config['track_width']
        self.robots_speed = self.config['robots_speed']
        self.eta = self.config['eta']
        self.robot_radius = self.config['robot_radius']
        self.debug = self.config.get('debug', False)

        # Arena bounds
        half_x = self.config['x_width'] / 2
        half_y = self.config['y_height'] / 2
        self.arena_bounds = {
            'x_min': self.config['center'][0] - half_x,
            'x_max': self.config['center'][0] + half_x,
            'y_min': self.config['center'][1] - half_y,
            'y_max': self.config['center'][1] + half_y
        }

    def create_run_folder(self, run_index):
        """Create a directory for the current run's logs."""
        run_folder = os.path.join(self.base_log_dir, f"run{run_index}")
        os.makedirs(run_folder, exist_ok=True)
        return run_folder

    def initialize_agents(self, run_folder):
        """Initialize agents in a dense grid centered at the origin."""
        self.agents = []
        commitments = random.choices(self.light_ids, k=self.num_agents)

        # Calculate grid dimensions
        spacing = 2 * self.robot_radius  # Minimum distance to avoid overlap
        num_agents = self.num_agents
        # Estimate grid size: try to make it as square as possible
        cols = int(math.ceil(math.sqrt(num_agents)))
        rows = int(math.ceil(num_agents / cols))
        
        # Calculate grid width and height
        grid_width = (cols - 1) * spacing
        grid_height = (rows - 1) * spacing
        
        # Center the grid at (0, 0)
        center_x, center_y = 0, 0
        x_start = center_x - grid_width / 2
        y_start = center_y - grid_height / 2

        # Ensure the grid fits within arena bounds
        if (x_start < self.arena_bounds['x_min'] or 
            x_start + grid_width > self.arena_bounds['x_max'] or
            y_start < self.arena_bounds['y_min'] or
            y_start + grid_height > self.arena_bounds['y_max']):
            raise ValueError("Grid dimensions exceed arena bounds. Reduce num_agents or robot_radius.")

        agent_idx = 0
        for row in range(rows):
            for col in range(cols):
                if agent_idx >= num_agents:
                    break
                x = x_start + col * spacing
                y = y_start + row * spacing
                agent = Agent(
                    id=agent_idx,
                    x=x,
                    y=y,
                    speed=self.robots_speed,
                    track_width=self.track_width,
                    direction=math.radians(self.config['robots_direction']),
                    commitment=commitments[agent_idx],
                    eta=self.eta,
                    light_ids=self.light_ids,
                    fov=self.fov,
                    update_offset=random.randint(1, self.config["commitment_update_time"]),
                    thresholds=self.thresholds,
                    turning_mechanism=NO_TURN,
                    radius=self.robot_radius,
                    all_agents=self.agents
                )
                agent.initialize_log_file(self.start_time, run_folder, self.experiment_name)
                self.agents.append(agent)
                agent_idx += 1

        # Update all_agents reference for all agents after creation
        for agent in self.agents:
            agent.all_agents = self.agents


    def initialize_position_log(self, run_folder):
        """Initialize the position log file."""
        filename = os.path.join(run_folder, f"{self.experiment_name}_positions_{self.start_time}.csv")
        self.position_log = open(filename, "w", newline="")
        writer = csv.writer(self.position_log)
        writer.writerow(["Time", "ID", "x", "y"])
        self.position_writer = writer

    def log_positions_data(self, time_step):
        """Log all agents' positions for the current timestep."""
        for agent in self.agents:
            self.position_writer.writerow([time_step, agent.id, agent.x, agent.y])

    def setup_visualization(self):
        """Initialize the visualization window."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(self.arena_bounds['x_min'], self.arena_bounds['x_max'])
        ax.set_ylim(self.arena_bounds['y_min'], self.arena_bounds['y_max'])
        ax.set_ylabel("y (m)", fontsize=16)
        ax.set_xlabel("x (m)", fontsize=16)

        # Plot light sources
        for light in self.light_sources.values():
            ax.scatter(light.x, light.y, color=light.color, s=100, label=f"Target {light.id}")

        if self.config["occlusion"]:
            # Initialize agent markers as circles
            self.agent_markers = [
                ax.add_patch(plt.Circle((0, 0), self.robot_radius, alpha=0.7, fc='gray'))
                for _ in range(self.num_agents)
            ]
        else:
            # Initialize agent markers
            self.agent_markers = [
                ax.add_patch(plt.Polygon([(0, 0)], closed=True, alpha=0.7))
                for _ in range(self.num_agents)
            ]
        # Debug lines: agent-to-agent and agent-to-light
        if self.debug:
            self.agent_to_agent_lines = []
            self.agent_to_light_lines = []
            # For each agent, create lines to all other agents
            for i in range(self.num_agents):
                agent_lines = []
                for j in range(self.num_agents):
                    if i != j:
                        line, = ax.plot([], [], 'b--', alpha=0.3, lw=1)  # Dashed blue for agents
                        agent_lines.append(line)
                self.agent_to_agent_lines.append(agent_lines)
            # For each agent, create lines to all light sources
            for i in range(self.num_agents):
                light_lines = []
                for _ in self.light_sources:
                    line, = ax.plot([], [], 'r-', alpha=0.3, lw=1)  # Solid red for lights
                    light_lines.append(line)
                self.agent_to_light_lines.append(light_lines)
        

        # Initialize trajectories
        self.trajectories = [ax.plot([], [], lw=1, alpha=0.6)[0]
                             for _ in range(self.num_agents)]

        self.time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        plt.legend(prop={'size': 16})
        return fig, ax

    def update_visualization_triangles(self, ax, timestep):
        """Update the visualization elements."""
        positions = self.history[-1]

        # Update agent markers
        for marker, (x, y, commitment), agent in zip(self.agent_markers, positions, self.agents):
            color = self.light_sources[commitment].color
            size_long = 1.0
            size_short = 0.1
            direction = agent.direction

            triangle_vertices = [
                (x + size_long * math.cos(direction), y + size_long * math.sin(direction)),
                (x + size_short * math.cos(direction + 2.5 * math.pi / 3),
                 y + size_short * math.sin(direction + 2.5 * math.pi / 3)),
                (x + size_short * math.cos(direction - 2.5 * math.pi / 3),
                 y + size_short * math.sin(direction - 2.5 * math.pi / 3)),
            ]
            marker.set_xy(triangle_vertices)
            marker.set_facecolor(color)

        # Update trajectories
        for traj, agent in zip(self.trajectories, self.agents):
            x_traj, y_traj = zip(*agent.trajectory)
            traj.set_data(x_traj, y_traj)

        # Update time display
        self.time_text.set_text(f"Time: {timestep / self.steps_per_second:.1f}s")
        plt.pause(0.001)

    def update_visualization_circles(self, ax, timestep):
        """Update the visualization elements."""
        positions = self.history[-1]
        for marker, (x, y, commitment), agent in zip(self.agent_markers, positions, self.agents):
            color = self.light_sources[commitment].color
            marker.set_center((x, y))  # Update circle center
            marker.set_facecolor(color)
        for traj, agent in zip(self.trajectories, self.agents):
            x_traj, y_traj = zip(*agent.trajectory)
            traj.set_data(x_traj, y_traj)
        
        # Update debug lines if enabled
        if self.debug:
            for i, agent in enumerate(self.agents):
                # Agent-to-agent lines
                line_idx = 0
                for j, other in enumerate(self.agents):
                    if i == j:
                        continue
                    line = self.agent_to_agent_lines[i][line_idx]
                    if other.id in agent.visible_agents:
                        # Determine start and end points based on check_type
                        check_type = "edge"  # As used in get_visible_agents
                        if check_type == "edge":
                            direction = np.array([other.x - agent.x, other.y - agent.y])
                            dist = np.linalg.norm(direction)
                            if dist > 0:
                                unit_direction = direction / dist
                                start = [agent.x, agent.y]
                                end = [other.x, other.y] - other.radius * unit_direction
                            else:
                                start, end = [agent.x, agent.y], [other.x, other.y]
                        else:
                            start = [agent.x, agent.y]
                            end = [other.x, other.y]
                        line.set_data([start[0], end[0]], [start[1], end[1]])
                    else:
                        line.set_data([], [])
                    line_idx += 1
                # Agent-to-light lines
                for j, (light_id, light) in enumerate(self.light_sources.items()):
                    line = self.agent_to_light_lines[i][j]
                    if agent.is_in_fov(light, agent.fov, True, check_type="edge"):
                        # Determine start point based on check_type
                        check_type = "edge"  # As used in is_in_fov
                        if check_type == "edge":
                            direction = np.array([light.x - agent.x, light.y - agent.y])
                            dist = np.linalg.norm(direction)
                            if dist > 0:
                                unit_direction = direction / dist
                                start = [agent.x, agent.y] - agent.radius * unit_direction
                            else:
                                start = [agent.x, agent.y]
                        else:
                            start = [agent.x, agent.y]
                        end = [light.x, light.y]
                        line.set_data([start[0], end[0]], [start[1], end[1]])
                    else:
                        line.set_data([], [])
        self.time_text.set_text(f"Time: {timestep / self.steps_per_second:.1f}s")
        plt.pause(0.001)

    def update_visualization(self, ax, timestep):
        if self.config["occlusion"]:
            self.update_visualization_circles(ax, timestep)
        else:
            self.update_visualization_triangles(ax, timestep)

    def check_termination_center_of_mass(self):
        #Check if center of mass is within termination radius of any light.
        x_com = sum(a.x for a in self.agents) / len(self.agents)
        y_com = sum(a.y for a in self.agents) / len(self.agents)
        return any(math.hypot(x_com - l.x, y_com - l.y) <= self.config['termination_radius']
                   for l in self.light_sources.values())
   
    def check_termination(self):
        """Check if all agents are within termination radius of their committed targets."""
        terminate = True
        for agent in self.agents:
            light = self.light_sources[agent.commitment]
            distance = math.hypot(agent.x - light.x, agent.y - light.y)
            if distance <= self.config['termination_radius']:
                agent.is_within_termination_radius = True
                #print(f"Agent {agent.id} reached target {agent.commitment} at ({light.x}, {light.y})")
            else:
                terminate = False
        return terminate

    def enforce_timing(self, step_start):
        """Maintain consistent timestep duration."""
        elapsed = time.perf_counter() - step_start
        sleep_time = self.step_duration - elapsed
        if sleep_time > 0 and not self.config.get('visualize', False):
            time.sleep(sleep_time)

    def cleanup(self):
        """Close all open resources."""
        # Close agent logs
        for agent in self.agents:
            agent.close_log_file()

        # Close position log
        if self.position_log:
            self.position_log.close()

        # Keep visualization open if enabled
        if self.config.get('visualize', False) and self.num_runs <= 1:
            plt.show()

    def run_experiments(self):
        """Run all configured experiment runs."""
        for run_idx in range(self.num_runs):
            run_folder = self.create_run_folder(run_idx)
            print(f"Starting run {run_idx + 1}/{self.num_runs}")
            self.run(run_folder)
            print(f"Completed run {run_idx + 1}")

    def run(self, run_folder):
        """Execute a single simulation run."""
        self.initialize_agents(run_folder)
        self.initialize_position_log(run_folder)

        if self.config.get('visualize', False):
            fig, ax = self.setup_visualization()

        try:
            for t in range(self.config['time_steps']):
                step_start = time.perf_counter()

                # Process simulation step
                self.process_timestep(t)

                # Update visualization if enabled
                if self.config.get('visualize', False):
                    self.update_visualization(ax, t)

                # Check termination condition
                if self.check_termination():
                    print(f"Termination condition met at step {t}")
                    break

                # Maintain timing
                #self.enforce_timing(step_start)

        finally:
            self.cleanup()

    def process_timestep(self, t):
        """Process a single simulation timestep."""
        # Determine broadcasts
        if self.config["occlusion"]:
            # If occlusion is enabled, update visibility of agents
            for agent in self.agents:
                agent.get_visible_agents()
        for agent in self.agents:
            light = self.light_sources[agent.commitment]
            target_dir = math.atan2(light.y - agent.y, light.x - agent.x)
            angle_diff = (target_dir - agent.direction + math.pi) % (2 * math.pi) - math.pi
            agent.determine_broadcast(self.thresholds['hard'], angle_diff)

        # Broadcast to neighbors
        self.broadcast_commitments()

        for agent in self.agents:
            target_ids = []
            for light_id in self.light_ids:
                # Check if light source is within FOV
                if agent.is_in_fov(self.light_sources[light_id], self.fov, self.config["occlusion"], check_type="edge"):
                    target_ids.append(light_id)
                else:
                    target_ids.append(0)  # Append 0 if not in FOV
            # Update visible targets
            agent.visible_targets.extend(target_ids)

        self.log_data(t)

        # Update commitments asynchronously
        for agent in self.agents:
            # Check if this agent's update time has come
            if (t) % self.config['commitment_update_time'] == 0:
                agent.update_commitment(self.light_sources, self.config["occlusion"])

        # Move agents and log data
        self.move_agents()

    """def broadcast_commitments(self):
        #Exchange broadcasts between all agents.
        for agent in self.agents:
            for neighbor in self.agents:
                if agent.id != neighbor.id:
                    neighbor.receive_broadcast(agent.id, agent.broadcast)"""

    def broadcast_commitments(self) -> None:
        """
        Broadcast commitments to agents that are both visible and within communication range.

        Agents send their broadcast data to neighbors within self.communication_range.
        Assumes each agent has attributes: id, x, y, broadcast, and a receive_broadcast method.

        Attributes:
            self.agents: List of agent objects.
            self.communication_range: Float defining the maximum distance for communication.
        """
        # Pre-compute agent positions for vectorized distance calculation
        positions = np.array([[agent.x, agent.y] for agent in self.agents])
        n_agents = len(self.agents)

        # Compute pairwise distances efficiently
        distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)

        # Broadcast within range
        for i, sender in enumerate(self.agents):
            # Log number of neighbors within communication range
            in_range = (distances[i] <= self.communication_range) & (np.arange(n_agents) != i)
            num_neighbors = np.sum(in_range)
            sender.number_of_neighbors.append(num_neighbors)
            
            neighbor_indices = np.where(in_range)[0]  # Get indices of neighbors
            # Get actual agent IDs (assuming agent.id exists)
            neighbor_ids = [self.agents[idx].id for idx in neighbor_indices]
            # Convert to comma-separated string (e.g., "1,2,3,6")
            neighbor_ids_str = ",".join(map(str, neighbor_ids))
            sender.ids_of_neighbors.append(neighbor_ids_str)  # Store string of neighbor IDs

            if self.config["occlusion"]:
                # Send broadcasts to visible agents within communication range
                for receiver_id in sender.visible_agents:
                    receiver_idx = next((j for j, agent in enumerate(self.agents) if agent.id == receiver_id), None)
                    if receiver_idx is not None and in_range[receiver_idx]:
                        receiver = self.agents[receiver_idx]
                        receiver.receive_broadcast(sender.id, sender.broadcast)
            else:
                # Broadcast to neighbors
                for j in np.where(in_range)[0]:
                    receiver = self.agents[j]
                    receiver.receive_broadcast(sender.id, sender.broadcast)
        """for i in range(n_agents):
            sender = self.agents[i]
            # Find neighbors within range (excluding self)
            in_range = (distances[i] <= self.communication_range) & (np.arange(n_agents) != i)
            # Count neighbors in range
            num_neighbors = np.sum(in_range)
            # Log the number of neighbors for this broadcast
            sender.number_of_neighbors.append(num_neighbors)"""
    

    def move_agents(self):
        """Update positions of all agents."""
        positions = []
        for agent in self.agents:
            if not agent.has_reached_target:
                light = self.light_sources[agent.commitment]
                agent.update_position(
                    light.x, light.y,
                    (1/self.steps_per_second)
                )
            positions.append((agent.x, agent.y, agent.commitment))
        self.history.append(positions)

    def log_data(self, time_step):
        """Log data for all agents."""
        self.log_positions_data(time_step)
        if time_step % self.config['commitment_update_time'] == 0:
            for agent in self.agents:
                agent.agent_log_data(time_step)
                


if __name__ == "__main__":
    with open("configs/config_2_targets_geometry.json") as f:
        config = json.load(f)

    simulation = Simulation(config)
    simulation.run_experiments()