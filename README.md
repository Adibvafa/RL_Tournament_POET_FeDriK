# Warehouse Brawl: Comprehensive Guide to Building an AI for Platform Fighting Games

[![Agents Fighting](https://img.youtube.com/vi/zOmkgw6jru4/0.jpg)](https://www.youtube.com/watch?v=zOmkgw6jru4)
<br>

## Table of Contents

1. **Introduction**
   - What is UTMIST AI2
   - Warehouse Brawl Overview
   - Project Goals
   - Repository Structure

2. **Environment Deep Dive**
   - Game Mechanics and Rules
   - Physics Engine Implementation
   - Movement System
   - Coordinate System
   - Player States and State Machine
   - Combat Mechanics
   - Animation and Rendering

3. **Player Implementation**
   - Player Class Architecture
   - State Transition Logic
   - Input Handling System
   - Movement Controls
   - Combat Controls
   - Animation System
   - Hitbox and Hurtbox System
   - Collisions and Damage Calculation

4. **Reinforcement Learning Fundamentals**
   - RL Basics for Fighting Games
   - Observation Space
   - Action Space
   - Markov Decision Process in Warehouse Brawl
   - Proximal Policy Optimization (PPO)

5. **Agent Design**
   - SubmittedAgent Architecture
   - Neural Network Structure
   - Hyperparameter Selection and Tuning
   - Safety Mechanisms and Heuristics
   - Platform Edge Detection
   - Integrating NN Predictions with Game Logic

6. **Training Methodology**
   - Reward Function Design Philosophy
   - Component Reward Functions
   - Signal-Based Rewards
   - Opponent Selection Strategy
   - Self-Play Implementation
   - Curriculum Learning Approach
   - Checkpointing and Model Management

7. **Advanced Topics**
   - Multi-Agent Learning Dynamics
   - Exploration vs. Exploitation Balance
   - Overfitting and Generalization
   - Hyperparameter Optimization
   - Agent Evaluation Metrics

8. **Running the Code**
   - Environment Setup
   - Dependencies Installation
   - Training Configuration
   - Evaluation Setup
   - Visualizing Results

9. **Credits and Acknowledgments**

---

## 1. Introduction

### What is UTMIST AI2

UTMIST AI2 (University of Toronto Machine Intelligence Student Team AI2) is an initiative focused on applying artificial intelligence techniques to competitive gaming environments. The project provides students and researchers with a platform to explore reinforcement learning, multi-agent systems, and game AI in a competitive and visually engaging context.

### Warehouse Brawl Overview

Warehouse Brawl is a 1v1 platform fighting game inspired by popular titles like Super Smash Bros. The game takes place on a single platform where two characters battle to knock each other off the stage. Unlike traditional fighting games with health bars, Warehouse Brawl uses a damage percentage system where characters become more vulnerable to knockback as their damage increases.

The game features:
- A robust physics engine using PyMunk
- Complex character state transitions
- A variety of attacks and movements
- Dynamic hitbox and hurtbox interactions
- Animation systems for visual feedback

### Project Goals

This project aims to:
1. Provide a comprehensive environment for AI research in competitive gaming
2. Demonstrate the application of reinforcement learning to complex, dynamic environments
3. Explore multi-agent interactions and self-play for skill development
4. Create a platform for comparing different AI approaches and architectures
5. Serve as an educational resource for students learning about game AI and RL

### Repository Structure

The codebase is organized into several key components:

- **Malachite Framework**: The foundational multi-agent reinforcement learning framework
- **WarehouseBrawl Environment**: The game implementation including physics, rendering, and rules
- **Player Implementation**: Character controls, states, and combat mechanics
- **Agent Architecture**: The AI implementation using PPO
- **Training Framework**: Reward functions, opponent selection, and training loop
- **Evaluation Tools**: Methods for testing and comparing agents

## 2. Environment Deep Dive

### Game Mechanics and Rules

The core objective in Warehouse Brawl is to knock your opponent off the stage while avoiding being knocked off yourself. Each player starts with 3 stocks (lives) and loses one when they fall off the stage or are knocked beyond the boundaries.

Key rules:
- **Damage System**: As players take damage, their percentage increases (displayed as 0-100%)
- **Knockback Scaling**: Higher damage percentages result in characters flying further when hit
- **Stock System**: Players start with 3 stocks and lose when all are depleted
- **Time Limit**: Standard matches last 90 seconds
- **Victory Conditions**: Win by depleting all opponent stocks or having more stocks when time expires

### Physics Engine Implementation

Warehouse Brawl uses PyMunk (a Python wrapper for the Chipmunk2D physics engine) integrated with PyGame for rendering:

```python
# Physics world setup
self.space = pymunk.Space()
self.dt = 1 / 30.0  # 30 FPS physics update
self.space.gravity = 0, 17.808  # Gravity vector (y is down)
```

The physics engine handles:
- Character movement and momentum
- Gravity and falling
- Collisions between characters and platforms
- Knockback from attacks
- Boundary detection for stage limits

### Movement System

Characters can perform a variety of movements:

- **Walking/Running**: Basic horizontal movement on platforms
- **Jumping**: Vertical movement with initial velocity boost
- **Air Movement**: Limited horizontal control while airborne
- **Dashing**: Quick horizontal movement burst
- **Dodging**: Brief invulnerability frames to avoid attacks
- **Turning**: Changing facing direction

Movement parameters are configured in the Player class:

```python
# Movement parameters
self.move_speed = 6.75         # Base walking speed
self.jump_speed = 8.9          # Initial jump velocity
self.in_air_ease = 6.75 / 30   # Air control factor
self.run_speed = 8             # Running speed
self.dash_speed = 10           # Dash speed
self.backdash_speed = 4        # Backdash speed
```

### Coordinate System

The game uses a 2D coordinate system:
- Origin (0,0) is at the center of the stage
- Positive X is right, negative X is left
- Positive Y is down, negative Y is up
- Stage boundaries are defined relative to the origin

Platform dimensions:
```python
self.stage_width_tiles = 29.8
self.stage_height_tiles = 16.8
```

### Player States and State Machine

The character behavior is governed by a state machine with transitions between different states:

```python
self.states = {
    'walking': WalkingState(self),
    'standing': StandingState(self),
    'turnaround': TurnaroundState(self),
    'air_turnaround': AirTurnaroundState(self),
    'sprinting': SprintingState(self),
    'stun': StunState(self),
    'in_air': InAirState(self),
    'dodge': DodgeState(self),
    'attack': AttackState(self),
    'dash': DashState(self),
    'backdash': BackDashState(self),
    'KO': KOState(self),
    'taunt': TauntState(self),
}
```

Each state handles:
- Physics processing specific to that state
- Input handling and control responses
- Transition logic to other states
- Animation selection
- Special properties (e.g., vulnerability during attacks)

### Combat Mechanics

The combat system involves several components:

- **Attacks**: Different moves with varying damage, knockback, and frame data
- **Hitboxes**: Areas that deal damage when contacting an opponent's hurtbox
- **Hurtboxes**: Vulnerable areas of characters that can receive damage
- **Frame Data**: Timing for startup, active, and recovery frames of attacks
- **Damage Calculation**: Formulas for determining damage and knockback based on move properties and opponent state
- **Stun System**: Temporary incapacitation after being hit

Move types include:
- Light attacks (faster but weaker)
- Heavy attacks (slower but stronger)
- Aerial attacks (performed while airborne)
- Ground attacks (performed while on the platform)
- Directional variants (neutral, side, down)

### Animation and Rendering

The game uses a frame-based animation system with sprites:

- **AnimationSprite2D**: Handles loading, scaling, and displaying character animations
- **KeyIconPanel**: Shows pressed keys for debugging purposes
- **UIHandler**: Manages user interface elements like damage percentages and stock indicators
- **Camera**: Controls viewport and coordinate transformations

Animations are synchronized with game states:
```python
def animate_player(self, camera) -> None:
    player_anim, attack_anim = self.p.attack_anims[self.move_type]
    current_power = self.move_manager.current_power
    if isinstance(player_anim, str):
        self.p.animation_sprite_2d.play(player_anim)
    # More animation logic...
```

## 3. Player Implementation

### Player Class Architecture

The `Player` class is the central component handling character behavior:

```python
class Player(GameObject):
    def __init__(self, env, agent_id: int, start_position=[0,0], color=[200, 200, 0, 255]):
        # Initialize physics body
        self.shape = pymunk.Poly.create_box(None, size=(width, height))
        self.body = pymunk.Body(self.mass, self.moment)
        
        # Initialize states
        self.states = {...}  
        self.state = self.states['in_air']
        
        # Character stats
        self.facing = Facing.RIGHT
        self.damage = 0
        self.stocks = 3
        
        # Animation components
        self.animation_sprite_2d = AnimationSprite2D(...)
        self.attack_sprite = AnimationSprite2D(...)
```

The Player class integrates:
- Physics body for collision and movement
- State machine for behavior control
- Input processing
- Animation management
- Combat properties
- Game statistics tracking

### State Transition Logic

State transitions follow specific rules based on current state, inputs, and environment conditions:

```python
def physics_process(self, dt: float) -> PlayerObjectState:
    new_state = super().physics_process(dt)
    if new_state is not None:
        return new_state

    # Example transition logic for StandingState
    direction: float = self.p.input.raw_horizontal
    if Facing.turn_check(self.p.facing, direction):
        return self.p.states['turnaround']
    if abs(direction) > 1e-2:
        self.p.facing = Facing.from_direction(direction)
        return self.p.states['walking']
```

Each state's `physics_process` method checks transition conditions and returns either:
- A new state to transition to
- None to remain in the current state

This allows for complex behavior chains like:
1. Standing → Walking → Dashing → Attacking
2. Standing → Jumping → In-Air → Aerial Attack

### Input Handling System

Player inputs are processed through the `PlayerInputHandler` class:

```python
class PlayerInputHandler():
    def __init__(self):
        self.key_names = ["W", "A", "S", "D", "space", 'h', 'l', 'j', 'k', 'g']
        self.prev_state = {key: False for key in self.key_names}
        self.key_status = {key: KeyStatus() for key in self.key_names}
        self.raw_vertical = 0.0
        self.raw_horizontal = 0.0

    def update(self, action: np.ndarray):
        # Update key statuses (just_pressed, held, just_released)
        for i, key in enumerate(self.key_names):
            current = action[i] > 0.5
            previous = self.prev_state[key]
            self.key_status[key].just_pressed = (not previous and current)
            self.key_status[key].just_released = (previous and not current)
            self.key_status[key].held = current
            self.prev_state[key] = current

        # Compute raw axes
        self.raw_vertical = (1.0 if self.key_status["W"].held else 0.0) + 
                           (-1.0 if self.key_status["S"].held else 0.0)
        self.raw_horizontal = (1.0 if self.key_status["D"].held else 0.0) + 
                             (-1.0 if self.key_status["A"].held else 0.0)
```

This system tracks:
- Currently held buttons
- Just pressed buttons (for single-frame actions)
- Just released buttons (for release triggers)
- Derived directional inputs (horizontal and vertical axes)

### Movement Controls

Movement is handled through a combination of input processing and physics forces:

- **Walking**: Applied when horizontal input is detected in ground states
  ```python
  self.p.body.velocity = pymunk.Vec2d(direction * self.p.move_speed, self.p.body.velocity.y)
  ```

- **Jumping**: Applied when Space is pressed while grounded
  ```python
  self.p.body.velocity = pymunk.Vec2d(self.p.body.velocity.x, -self.p.jump_speed)
  ```

- **Air Movement**: Limited horizontal control while airborne
  ```python
  vel_x = self.p.move_toward(self.p.body.velocity.x, direction * self.p.move_speed, self.p.in_air_ease)
  self.p.body.velocity = pymunk.Vec2d(vel_x, self.p.body.velocity.y)
  ```

- **Dashing**: Quick burst of speed in the facing direction
  ```python
  self.p.body.velocity = pymunk.Vec2d(int(self.p.facing) * self.p.dash_speed, self.p.body.velocity.y)
  ```

### Combat Controls

Combat is initiated through J (light attacks) and K (heavy attacks) buttons, with direction modifiers:

```python
def get_move(self) -> MoveType:
    # Determine move types
    heavy_move = self.input.key_status['k'].held
    light_move = (not heavy_move) and self.input.key_status['j'].held
    
    # Determine directional keys
    up_key = self.input.key_status["W"].held
    down_key = self.input.key_status["S"].held
    side_key = self.input.key_status["A"].held or self.input.key_status["D"].held
    
    # Calculate move direction
    neutral_move = ((not side_key) and (not down_key)) or up_key
    down_move = (not neutral_move) and down_key
    side_move = (not neutral_move) and (not down_key) and side_key
    
    # Determine final move type
    cms = CompactMoveState(self.is_on_floor(), heavy_move, 
                           0 if neutral_move else (1 if down_move else 2))
    move_type = m_state_to_move[cms]
    return move_type
```

Attack execution is handled by the `AttackState` and supporting classes:
- `MoveManager`: Controls overall attack flow and data
- `Power`: Handles specific attack properties and frame data
- `Cast`: Manages hitbox activation and collision detection
- `CastFrameChangeHolder`: Tracks frame-by-frame changes during attacks

### Animation System

The animation system uses sprite sheets loaded from GIF files:

```python
class AnimationSprite2D(GameObject):
    def __init__(self, camera, scale, animation_folder, agent_id):
        self.animations: dict[str, Animation] = {}
        self.current_animation = None
        self.frames = []
        self.current_frame_index = 0
        
    def load_animation(self, file_path):
        # Load GIF and extract frames
        gif = Image.open(file_path)
        frames = []
        frame_durations = []
        
        for frame in ImageSequence.Iterator(gif):
            pygame_frame = pygame.image.fromstring(frame.convert("RGBA").tobytes(), 
                                                  frame.size, "RGBA")
            frames.append(pygame_frame)
            
            # Extract frame duration
            duration = frame.info.get('duration', 100)
            frame_durations.append(duration)
            
        # Calculate frames per game step for timing
        frames_per_step = [max(1, round((duration / 1000) * self.ENV_FPS)) 
                           for duration in frame_durations]
        
        return Animation(frames, frame_durations, frames_per_step)
```

Animations are synchronized with game states through the `animate_player` method in each state class.

### Hitbox and Hurtbox System

The collision system uses capsule colliders for precise hit detection:

```python
class CapsuleCollider():
    def __init__(self, center, width, height, is_hurtbox=False):
        self.center = pygame.Vector2(center)
        self.width = width
        self.height = height
        self.radius = min(width, height) / 2
        self.is_circle = width == height
        
    def intersects(self, other):
        # Complex collision detection between capsules
        # Handles circle-circle, circle-rectangle, and capsule-capsule collisions
```

Hitboxes are created during attacks with specific properties:
- Position relative to the character
- Size and shape
- Active frames
- Damage and knockback values

```python
# Example hitbox creation during an attack
hitbox_offset = Capsule.get_hitbox_offset(hitbox['xOffset'], hitbox['yOffset'])
hitbox_offset = (hitbox_offset[0] * int(mm.move_facing_direction), hitbox_offset[1])
hitbox_pos = (self.p.body.position[0] + hitbox_offset[0], 
              self.p.body.position[1] + hitbox_offset[1])
hitbox_size = Capsule.get_hitbox_size(hitbox['width'], hitbox['height'])
capsule1 = CapsuleCollider(center=hitbox_pos, width=hitbox_size[0], 
                          height=hitbox_size[1])
```

### Collisions and Damage Calculation

When a hitbox and hurtbox collide, damage and knockback are calculated:

```python
# Damage calculation (simplified)
force_magnitude = (current_cast.fixed_force + 
                  current_cast.variable_force * hit_agent.damage * 0.02622)

# Apply damage and knockback
hit_agent.apply_damage(damage_to_deal, self.stun_time,
                      (hit_vector[0] * force_magnitude, 
                       hit_vector[1] * force_magnitude))
```

The damage system accounts for:
- Base damage of the move
- Variable scaling based on target's current damage
- Knockback direction based on hit angle
- Stun duration to prevent immediate retaliation

## 4. Reinforcement Learning Fundamentals

### RL Basics for Fighting Games

Reinforcement Learning (RL) is particularly well-suited for fighting games due to:

1. **Clear Reward Structure**: Win/loss conditions and damage provide natural rewards
2. **Complex State Space**: Character positions, velocities, animations create rich observations
3. **Strategic Depth**: Balance between offense, defense, positioning creates interesting policies
4. **Emergence**: Agents can discover strategies beyond explicit programming

The fundamental RL components in Warehouse Brawl are:

- **Agent**: The AI controlling one player
- **Environment**: The Warehouse Brawl game with its physics and rules
- **State**: The game state including positions, velocities, animations, etc.
- **Action**: The 10-dimensional control vector for player inputs
- **Reward**: Signals indicating success or failure
- **Policy**: The strategy that maps observations to actions

### Observation Space

The observation space provides all necessary information for decision-making:

```python
def get_obs(self) -> list[float]:
    obs = []
    # Player position (normalized to [-1, 1])
    obs.extend([max(-18, min(18, pos.x)), max(-7, min(7, pos.y))])
    
    # Player velocity (normalized to [-10, 10])
    obs.extend([max(-10.0, min(10.0, vel.x)), max(-10.0, min(10.0, vel.y))])
    
    # Facing direction (0 for left, 1 for right)
    obs.append(1.0 if self.facing == Facing.RIGHT else 0.0)
    
    # State information (grounded, aerial, jumps left, etc.)
    obs.append(float(self.is_on_floor()))
    obs.append(0.0 if grounded == 1.0 else 1.0)
    obs.append(float(self.state.jumps_left) if hasattr(self.state, 'jumps_left') else 0.0)
    
    # Current state type
    current_state_name = type(self.state).__name__
    state_index = self.state_mapping.get(current_state_name, 0)
    obs.append(float(state_index))
    
    # Combat information
    obs.append(float(self.damage) / 700.0)
    obs.append(float(self.stocks))
    obs.append(float(self.state.move_type) if hasattr(self.state, 'move_type') else 0.0)
    
    return obs
```

The observation includes both player and opponent information, totaling 26 dimensions:
- Positions (4D)
- Velocities (4D)
- Facing directions (2D)
- Grounded state (2D)
- Aerial state (2D)
- Jumps remaining (2D)
- State indices (2D)
- Recoveries remaining (2D)
- Dodge timers (2D)
- Stun frames (2D)
- Damage percentages (2D)
- Stock counts (2D)
- Move types (2D)

### Action Space

The action space is a 10-dimensional binary vector corresponding to game controls:

```python
def get_action_space(self):
    act_helper = ActHelper()
    act_helper.add_key("w")     # W (Aim up)
    act_helper.add_key("a")     # A (Left)
    act_helper.add_key("s")     # S (Aim down/fastfall)
    act_helper.add_key("d")     # D (Right)
    act_helper.add_key("space") # Space (Jump)
    act_helper.add_key("h")     # H (Pickup/Throw)
    act_helper.add_key("l")     # L (Dash/Dodge)
    act_helper.add_key("j")     # J (Light Attack)
    act_helper.add_key("k")     # K (Heavy Attack)
    act_helper.add_key("g")     # G (Taunt)
    
    return act_helper.get_as_box()
```

This space allows for 2^10 = 1024 possible action combinations, though many are impractical or counterproductive (e.g., pressing opposite directions simultaneously).

### Markov Decision Process in Warehouse Brawl

The game environment can be modeled as a Markov Decision Process (MDP):

- **States (S)**: The observation space containing character positions, velocities, etc.
- **Actions (A)**: The 10-dimensional control vector
- **Transition Function (P)**: The game physics and logic determining state changes
- **Reward Function (R)**: The rewards provided for good actions and outcomes
- **Discount Factor (γ)**: The weighting of future rewards (typically 0.99)

The goal is to find a policy π that maximizes expected cumulative rewards:

E[Σγᵗ * R(sₜ, aₜ)]

### Proximal Policy Optimization (PPO)

PPO is a policy gradient method that offers stable learning by constraining policy updates:

```python
self.model = PPO(
    "MlpPolicy",
    self.env,
    learning_rate=self.learning_rate,
    n_steps=self.n_steps,
    batch_size=self.batch_size,
    n_epochs=self.n_epochs,
    gamma=self.gamma,
    gae_lambda=self.gae_lambda,
    ent_coef=self.ent_coef,
    clip_range=self.clip_range,
    policy_kwargs=policy_kwargs,
    verbose=self.verbose,
    normalize_advantage=True,
    max_grad_norm=self.max_grad_norm,
    device=self.device,
)
```

Key PPO components:
1. **Policy Network**: Maps observations to action probabilities
2. **Value Network**: Estimates expected rewards from states
3. **Advantage Estimation**: Uses Generalized Advantage Estimation (GAE)
4. **Clipped Objective**: Prevents too large policy updates
5. **Multiple Epochs**: Reuses collected transitions for efficient learning

PPO is well-suited for Warehouse Brawl because:
- It's sample efficient (important for complex environments)
- It has stable learning dynamics
- It handles continuous state spaces and discrete action spaces naturally
- It balances exploration and exploitation effectively

## 5. Agent Design

### SubmittedAgent Architecture

The `SubmittedAgent` class implements a PPO-based reinforcement learning agent:

```python
class SubmittedAgent(Agent):
    def __init__(
        self,
        file_path: Optional[str] = None,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 128,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ent_coef: float = 0.01,
        clip_range: float = 0.2,
        verbose: int = 1,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        # Initialize parameters
        # ...
        super().__init__(file_path)

    def _initialize(self) -> None:
        if self.file_path is None:
            # Create new model with deep network architecture
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[64, 128, 256, 128, 64],  # Policy network
                    vf=[64, 128, 256, 128, 64],  # Value network
                ),
                activation_fn=torch.nn.ReLU,
                ortho_init=True,
            )
            self.model = PPO("MlpPolicy", self.env, ...)
        else:
            # Load existing model
            self.model = PPO.load(self.file_path, ...)

    def predict(self, obs):
        # Call model prediction with safety logic
        # ...
```

The agent integrates several components:
1. The core PPO model from Stable-Baselines3
2. Custom network architecture
3. Safety mechanisms for platform edge detection
4. Loading/saving functionality for model persistence

### Neural Network Structure

The agent uses a deep multi-layer perceptron with symmetric architecture:

```python
policy_kwargs = dict(
    net_arch=dict(
        pi=[64, 128, 256, 128, 64],  # Policy network (action prediction)
        vf=[64, 128, 256, 128, 64],  # Value network (state evaluation)
    ),
    activation_fn=torch.nn.ReLU,
    ortho_init=True,  # Orthogonal initialization for better convergence
)
```

This architecture features:
- **Symmetric Hourglass Design**: Expanding then contracting layer sizes
- **Deep Middle Layers**: 256 neurons in the middle for complex pattern recognition
- **ReLU Activation**: Provides non-linearity without vanishing gradients
- **Orthogonal Initialization**: Helps maintain gradient magnitudes during backpropagation

The network processes 26-dimensional observation vectors and outputs:
- **Policy Head**: 10-dimensional vector of action probabilities
- **Value Head**: Scalar value estimating the expected future reward

### Hyperparameter Selection and Tuning

Critical hyperparameters for PPO training:

```python
learning_rate = 3e-4        # Adam optimizer step size
n_steps = 2048              # Steps per update batch
batch_size = 128            # Minibatch size for gradient updates
n_epochs = 10               # Number of passes over each batch
gamma = 0.99                # Discount factor for future rewards
gae_lambda = 0.95           # GAE smoothing parameter
ent_coef = 0.01             # Entropy bonus for exploration
clip_range = 0.2            # Policy update clipping range
max_grad_norm = 0.5         # Gradient clipping threshold
```

These parameters were selected to balance:
- **Learning Speed**: Controlled by learning_rate and n_steps
- **Sample Efficiency**: Influenced by batch_size and n_epochs
- **Exploration**: Tuned with ent_coef
- **Stability**: Maintained by clip_range and max_grad_norm

### Safety Mechanisms and Heuristics

The agent implements platform edge detection and safety behaviors:

```python
def predict(self, obs):
    # Extract relevant observations
    player_pos = self.obs_helper.get_section(obs, "player_pos")
    player_vel = self.obs_helper.get_section(obs, "player_vel")
    player_grounded = self.obs_helper.get_section(obs, "player_grounded")

    # Platform boundaries
    platform_left = -10.67 / 2
    platform_right = 10.67 / 2
    platform_top = 1.75

    # Initialize action
    action = np.zeros_like(self.action_space.sample())

    # Check if we're near the edge or above the platform
    danger_zone = 0.5  # Buffer zone from the edge

    # If we're in danger of falling off or already off
    if ((player_pos[0] < platform_left + danger_zone and player_vel[0] < 0) or
        (player_pos[0] > platform_right - danger_zone and player_vel[0] > 0) or
        (player_pos[1] > platform_top)):

        # Move toward center of platform
        if player_pos[0] < 0:
            action = self.act_helper.press_keys(["d"])
        else:
            action = self.act_helper.press_keys(["a"])

        # Jump if we're not on the ground
        if player_pos[1] > platform_top or player_grounded[0] == 0:
            action = self.act_helper.press_keys(["space"], action)

        return action

    # If we're safe, use the model's prediction
    model_action, _ = self.model.predict(obs, deterministic=True)
    return model_action
```

This safety mechanism:
1. Detects dangerous platform positions
2. Overrides the neural network when danger is detected
3. Forces movement toward platform center
4. Triggers jumping to recover when off platform
5. Falls back to model predictions when safe

### Integrating NN Predictions with Game Logic

The agent combines neural network outputs with game-specific knowledge:

1. **Action Selection**: The network outputs action probabilities, which are sampled (during training) or maximized (during evaluation)
2. **Safety Logic**: Platform edge detection overrides network decisions when necessary
3. **Action Execution**: Selected actions are passed to the game environment
4. **Observation Processing**: The resulting game state is observed and passed back to the network

This integration allows the agent to learn complex behaviors while preventing catastrophic failures from exploration.

## 6. Training Methodology

### Reward Function Design Philosophy

Effective reward functions for fighting games should:

1. Provide immediate feedback for good/bad actions
2. Reflect the ultimate goal (winning matches)
3. Balance short-term tactics and long-term strategy
4. Avoid reward hacking or unintended behaviors
5. Guide learning through sparse win/loss signals

The `RewardManager` class orchestrates multiple reward components:

```python
class RewardManager():
    def __init__(self,
                 reward_functions: Optional[Dict[str, RewTerm]]=None,
                 signal_subscriptions: Optional[Dict[str, Tuple[str, RewTerm]]]=None) -> None:
        self.reward_functions = reward_functions
        self.signal_subscriptions = signal_subscriptions
        self.total_reward = 0.0
        self.collected_signal_rewards = 0.0
        
    def process(self, env, dt) -> float:
        reward_buffer = 0.0
        
        # Process continuous reward functions
        if self.reward_functions is not None:
            for name, term_cfg in self.reward_functions.items():
                if term_cfg.weight == 0.0:
                    continue
                value = term_cfg.func(env, **term_cfg.params) * term

### Component Reward Functions

The agent uses a combination of reward components to shape behavior:

```python
reward_functions = {
    # Reward for having more lives than opponent (weight 1.5)
    "lives_advantage": RewTerm(func=RewardFunctions.lives_advantage, weight=1.5),
    
    # Reward for dealing more damage than opponent (weight 1.5)
    "damage_advantage": RewTerm(func=RewardFunctions.damage_advantage, weight=1.5),
    
    # Penalty for pressing too many buttons simultaneously (weight 0.25)
    "keyboard_efficiency": RewTerm(
        func=RewardFunctions.keyboard_efficiency, weight=0.25
    ),
    
    # Penalty for having equal or worse health than opponent (weight 0.5)
    "health_disadvantage": RewTerm(
        func=RewardFunctions.health_disadvantage_time, weight=0.5
    ),
    
    # Reward for maintaining good positioning (weight 1.0)
    "tactical_positioning": RewTerm(
        func=RewardFunctions.tactical_positioning, weight=1
    ),
}
```

Each reward function addresses specific aspects of fighting game strategy:

1. **Lives Advantage**: Encourages keeping stocks while depleting opponent stocks
   ```python
   def lives_advantage(env: WarehouseBrawl) -> float:
       player: Player = env.objects["player"]
       opponent: Player = env.objects["opponent"]
       stocks_diff = player.stocks - opponent.stocks
       return stocks_diff - 1
   ```

2. **Damage Advantage**: Rewards dealing damage while avoiding taking damage
   ```python
   def damage_advantage(env: WarehouseBrawl) -> float:
       player: Player = env.objects["player"]
       opponent: Player = env.objects["opponent"]
       damage_taken = player.damage_taken_this_frame
       damage_dealt = opponent.damage_taken_this_frame
       return (damage_dealt - damage_taken) / 140
   ```

3. **Keyboard Efficiency**: Discourages button mashing by penalizing excessive inputs
   ```python
   def keyboard_efficiency(env: WarehouseBrawl) -> float:
       player: Player = env.objects["player"]
       pressed = sum(1 for key in player.input.key_status.values() if key.held)
       if pressed > 3:
           return -0.1 * pressed
       return 0
   ```

4. **Health Disadvantage Time**: Penalizes staying at higher damage than opponent
   ```python
   def health_disadvantage_time(env: WarehouseBrawl) -> float:
       player: Player = env.objects["player"]
       opponent: Player = env.objects["opponent"]
       if player.damage >= opponent.damage:
           return -0.1 * env.dt
       return 0
   ```

5. **Tactical Positioning**: Rewards intelligent movement and positioning
   ```python
   def tactical_positioning(env: WarehouseBrawl) -> float:
       # Complex positioning logic including:
       # - Stock/damage advantage assessment
       # - Movement direction evaluation
       # - Input efficiency rewards
       # - Position relative to opponent
   ```

### Signal-Based Rewards

In addition to continuous rewards, the agent receives event-triggered rewards:

```python
signal_subscriptions = {
    # Large reward (2.5) when winning a match
    "on_win": ("win_signal", RewTerm(func=RewardFunctions.on_win_reward, weight=2.5)),
    
    # Large reward (2.5) when knocking out opponent
    "on_knockout": (
        "knockout_signal",
        RewTerm(func=RewardFunctions.on_knockout_reward, weight=2.5),
    ),
}
```

These signal-based rewards provide immediate feedback for critical game events:

1. **On Win**: Large positive reward when winning a match
   ```python
   def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
       if agent == 'player':
           return 1.0
       else:
           return -1.0
   ```

2. **On Knockout**: Reward for knocking out an opponent (taking a stock)
   ```python
   def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
       if agent == 'player':
           return -1.0
       else:
           return 1.0
   ```

Signal-based rewards are connected to the environment's event system:

```python
def subscribe_signals(self, env) -> None:
    if self.signal_subscriptions is None:
        return
    for _, (name, term_cfg) in self.signal_subscriptions.items():
        getattr(env, name).connect(partial(self._signal_func, term_cfg))
```

### Opponent Selection Strategy

Training against diverse opponents helps the agent generalize and avoid overfitting to specific strategies:

```python
opponents = {
    "self_play": (0.5, selfplay_handler),                # 50% chance of facing itself
    "constant_agent": (0.02, partial(ConstantAgent)),    # 2% chance of facing constant agent
    "based_agent": (0.02, partial(BasedAgent)),          # 2% chance of facing based agent
    "random_agent": (0.04, partial(RandomAgent)),        # 4% chance of facing random agent
    "good_agent": (
        0.04,
        partial(
            SubmittedAgent,
            file_path="checkpoints/sample/saved/rl_model_3487533_steps_good"
        )
    ),
    # ... more opponent types
}
```

The opponent selection strategy includes:

1. **Self-Play** (50%): Learning from competing against its own previous versions
2. **Baseline Agents** (8%): Simple rule-based agents that provide consistent behavior
3. **Pre-trained Agents** (42%): Different versions of trained agents with varying strengths and styles

This diversity ensures the agent:
- Doesn't overfit to a single opponent type
- Learns to handle a variety of strategies
- Continues to improve through increasingly challenging self-play
- Can generalize to unseen opponents

### Self-Play Implementation

Self-play is a critical component allowing the agent to improve by competing against its own previous versions:

```python
class SelfPlayHandler():
    def __init__(self, agent_partial: partial, mode: SelfPlaySelectionMode=SelfPlaySelectionMode.LATEST):
        self.agent_partial = agent_partial
        self.mode = mode

    def get_opponent(self) -> Agent:
        assert self.save_handler is not None, "Save handler must be specified for self-play"

        if self.mode == SelfPlaySelectionMode.LATEST:
            # Get the best model from the save handler
            self.best_model = self.save_handler.get_latest_model_path()
            if self.best_model:
                try:
                    opponent = self.agent_partial(file_path=self.best_model)
                    opponent.get_env_info(self.env)
                    return opponent
                except FileNotFoundError:
                    # Fallback to constant agent if model not found
                    opponent = ConstantAgent()
                    opponent.get_env_info(self.env)
            else:
                opponent = ConstantAgent()
                opponent.get_env_info(self.env)

        return opponent
```

The self-play system:
1. Loads previous model checkpoints as opponents
2. Uses a selection strategy (latest model by default)
3. Creates a constantly evolving challenge for the learning agent
4. Enables emergence of complex strategies without explicit programming

### Curriculum Learning Approach

The training process uses implicit curriculum learning through:

1. **Opponent Progression**: Starting with simple opponents and gradually introducing more complex ones
2. **Self-Play Evolution**: As the agent improves, self-play opponents become more challenging
3. **Reward Weights**: Balancing different reward components to guide learning progression

This curriculum approach helps the agent:
- Learn basic skills before attempting complex strategies
- Build on previously learned behaviors
- Avoid getting stuck in local optima
- Develop increasingly sophisticated play styles

### Checkpointing and Model Management

The `SaveHandler` class manages model persistence and versioning:

```python
class SaveHandler():
    def __init__(
            self,
            agent: Agent,
            save_freq: int=10_000,
            max_saved: int=20,
            run_name: str='experiment_1',
            save_path: str='checkpoints',
            name_prefix: str = "rl_model",
            mode: SaveHandlerMode=SaveHandlerMode.FORCE
        ):
        self.agent = agent
        self.save_freq = save_freq
        self.run_name = run_name
        self.max_saved = max_saved
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.mode = mode

        self.steps_until_save = save_freq
        self.history: List[str] = []
        
    def save_agent(self) -> None:
        print(f"Saving agent to {self._checkpoint_path()}")
        model_path = self._checkpoint_path('zip')
        self.agent.save(model_path)
        self.history.append(model_path)
        if len(self.history) > self.max_saved:
            os.remove(self.history.pop(0))
```

Checkpointing features:
1. **Regular Saving**: Models are saved every `save_freq` steps
2. **Version Management**: Maintains up to `max_saved` most recent checkpoints
3. **Named Experiments**: Organizes checkpoints by experiment name
4. **Modes**: Supports forcing new runs or resuming existing ones
5. **History Tracking**: Maintains a list of available checkpoint files

## 7. Advanced Topics

### Multi-Agent Learning Dynamics

The Warehouse Brawl environment creates interesting multi-agent learning dynamics:

1. **Co-evolution**: Self-play creates an arms race of increasingly sophisticated strategies
2. **Exploitation vs. Adaptation**: Agents learn to exploit weaknesses while adapting to counter-strategies
3. **Nash Equilibrium**: The training process gradually approaches a mixed strategy equilibrium
4. **Policy Cycling**: Strategies often cycle through rock-paper-scissors patterns
5. **Emergent Behaviors**: Complex tactics emerge without explicit programming

These dynamics lead to:
- More robust policies than training against fixed opponents
- Discovery of non-obvious strategies
- Better generalization to unseen opponents
- Protection against overfitting to specific policies

### Exploration vs. Exploitation Balance

Balancing exploration and exploitation is critical for effective learning:

```python
# Entropy coefficient controls exploration
ent_coef = 0.01

# Used in PPO initialization
self.model = PPO(
    "MlpPolicy",
    self.env,
    ent_coef=self.ent_coef,
    # Other parameters...
)
```

The entropy coefficient (`ent_coef`) controls how much the agent is encouraged to explore:
- **Higher values** (e.g., 0.05): More exploration, discovering new strategies but potentially unstable
- **Lower values** (e.g., 0.001): More exploitation, refining existing strategies but potentially suboptimal
- **Decreasing schedule**: Some implementations gradually reduce entropy during training

Other exploration mechanisms include:
- **Diverse opponents**: Different opponents encourage exploring different strategies
- **Stochastic action selection**: During training, actions are sampled probabilistically
- **Reward shaping**: Rewards for novel behaviors can encourage exploration

### Overfitting and Generalization

Fighting game agents face significant risks of overfitting to specific opponents or strategies:

1. **Opponent Overfitting**: Learning exploits that work only against certain opponents
2. **Pattern Overfitting**: Developing rigid patterns that are effective but predictable
3. **Environment Overfitting**: Strategies that rely on specific aspects of the environment

Techniques used to improve generalization:
- **Diverse opponent pool**: Training against many different opponent types
- **Environment randomization**: Varying initial positions and conditions
- **Domain randomization**: Altering physics parameters during training
- **Regularization**: Using network weight regularization and dropout
- **Self-play**: Continually adapting to an improving opponent

### Hyperparameter Optimization

Finding optimal hyperparameters for training involves balancing several factors:

1. **Learning Rate**: Controls step size during optimization
   - Too high: Unstable learning, inability to converge
   - Too low: Slow learning, getting stuck in local optima
   - Typical range: 1e-5 to 1e-3, with 3e-4 being a common default

2. **Batch Size**: Number of samples processed in each gradient update
   - Larger batches: More stable gradient estimates but less exploration
   - Smaller batches: More exploration but noisier updates
   - Typical range: 64 to 512

3. **Network Architecture**: Layer sizes and structure
   - Deeper networks: More capacity but harder to train
   - Wider networks: Better feature representation but more parameters
   - Design choices: [64, 128, 256, 128, 64] provides good balance

4. **Discount Factor** (gamma): Importance of future rewards
   - Higher values (near 1.0): More emphasis on long-term rewards
   - Lower values: More emphasis on immediate rewards
   - Fighting games typically use 0.99 to 0.995

Optimization approaches:
- **Grid search**: Trying combinations of parameters
- **Random search**: Sampling from parameter distributions
- **Bayesian optimization**: Using probabilistic models to guide search
- **Population-based training**: Evolving hyperparameters during training

### Agent Evaluation Metrics

Evaluating fighting game agents requires multiple metrics:

1. **Win Rate**: Percentage of matches won against benchmark opponents
2. **Average Damage Dealt**: Total damage dealt to opponents
3. **Average Damage Taken**: Total damage received
4. **Stock Differential**: Average difference in remaining stocks
5. **Match Duration**: Time taken to achieve victory
6. **Action Efficiency**: Ratio of successful actions to total actions
7. **Recovery Rate**: Success rate of returning to stage after being knocked off
8. **Combo Frequency**: Frequency of successful multi-hit combinations

Evaluation should include:
- **Multiple Opponents**: Testing against diverse strategies
- **Round-Robin Tournaments**: Comparing against all available agents
- **Ablation Studies**: Testing with and without specific components
- **Human Evaluation**: Subjective assessment of strategy and play style

## 8. Running the Code

### Environment Setup

Setting up the Warehouse Brawl environment requires:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/utmist/warehouse-brawl.git
   cd warehouse-brawl
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

### Dependencies Installation

Install required packages:

```bash
# Core dependencies
pip install torch==2.4.1 gymnasium pygame==2.6.1 pymunk==6.2.1 
pip install scikit-image scikit-video sympy==1.5.1

# Reinforcement learning libraries
pip install stable_baselines3 sb3-contrib

# Utilities
pip install jupyter gdown opencv-python
```

### Training Configuration

Configure and start training:

```python
# Create agent
my_agent = SubmittedAgent(device="cuda")

# Create reward manager
reward_manager = RewardManager(reward_functions, signal_subscriptions)

# Configure opponents
opponent_cfg = OpponentsCfg(opponents=opponents)

# Configure checkpointing
save_handler = SaveHandler(
    agent=my_agent,
    run_name="sample",
    mode=SaveHandlerMode.FORCE,
    save_freq=25_000,
    max_saved=3,
    save_path="checkpoints",
)

# Start training
train(
    my_agent,                   # Agent to train
    reward_manager,             # Reward configuration
    save_handler,               # Checkpoint configuration
    opponent_cfg,               # Opponent configuration
    CameraResolution.LOW,       # Resolution for faster training
    train_timesteps=200_000,    # Total timesteps to train
    train_logging=TrainLogging.PLOT,  # Display training progress
)
```

Key training parameters:
- **train_timesteps**: Total number of environment steps (200k-5M recommended)
- **CameraResolution**: Lower resolutions speed up training
- **train_logging**: PLOT displays learning curves, TO_FILE saves logs

### Evaluation Setup

Evaluate trained agents against each other:

```python
# Load trained agents
agent1 = SubmittedAgent(file_path="checkpoints/sample/rl_model_2185729_steps")
agent2 = SubmittedAgent(file_path="checkpoints/sample/rl_model_2185729_steps")

# Set match duration in seconds
match_time = 90

# Run a match
run_match(
    agent1,
    agent_2=agent2,
    video_path="match.mp4",
    resolution=CameraResolution.LOW,
    reward_manager=reward_manager,
    max_timesteps=30 * match_time,
)
```

Options for different opponents:
- **SubmittedAgent**: Your custom PPO-based agent
- **RandomAgent**: Takes random actions
- **BasedAgent**: Simple rule-based agent
- **ConstantAgent**: Does nothing
- **UserInputAgent**: Allows human control via keyboard

### Visualizing Results

The environment provides several visualization options:

1. **Match videos**: Saved as MP4 files for review
2. **Learning curves**: When using TrainLogging.PLOT
3. **Reward breakdowns**: Detailed in log files
4. **Interactive play**: Using UserInputAgent to compete against AI

## 9. Credits and Acknowledgments

This environment was developed by the University of Toronto Machine Intelligence Student Team (UTMIST):

- **General Event Organization**: Asad, Efe, Andrew, Matthew, Kaden
- **Notebook code**: Kaden, Martin, Andrew
- **Notebook art/animations**: EchoTecho, Andy
- **Website code**: Zain, Sarva, Adam, Aina
- **Workshops**: Jessica, Jingmin, Asad, Tyler, Wai Lim, Napasorn, Sara, San, Alden
- **Tournament Server**: Ambrose, Doga, Steven
- **Technical guide + Conference brochure**: Matthew, Caitlin, Lucie

### References

The environment draws inspiration from several sources:

- **[Shootout AI](https://github.com/ajwm8103/shootoutai/tree/main)**: Base game mechanics
- **[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)**: Animation system
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)**: Reinforcement learning framework
- **PyMunk and PyGame**: Physics and rendering libraries

### License

This project is licensed under [Add appropriate licensing information].

---

This README provides a comprehensive guide to the Warehouse Brawl environment, agent implementation, and training methodology. Whether you're a beginner looking to understand reinforcement learning in games or an experienced researcher exploring multi-agent dynamics, we hope this repository serves as a valuable resource.

For additional resources and updates, visit:
- [Technical Guide Notebook](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ)
- [Introductory RL Notebook](https://colab.research.google.com/drive/1JRQFLU5jkMrIJ5cWs3xKEO0e9QKuE0Hi#scrollTo=9UCawVuAI3k0)
- [Discord Server](https://discord.com/invite/TTGB62BE9U)
