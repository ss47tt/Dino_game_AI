import pygame
import random
import numpy as np
import threading
import pickle
from concurrent.futures import ThreadPoolExecutor

# Initialize pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 600
FPS = 60
GROUND_LEVEL = SCREEN_HEIGHT - 20  # Distance from the bottom of the screen to the ground


# Colors
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 128)  # Green background color
BROWN = (139, 69, 19)  # Brown ground color
YELLOW = (255, 255, 0)
GROUND_COLOR = BROWN  # Use brown for ground

# Game variables
GRAVITY = 1
JUMP_STRENGTH = -30
SPEED = 5

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dinosaur Game")

# Fonts
font = pygame.font.SysFont(None, 36)

# Load assets (optional: replace with your own images)
dino_image = pygame.Surface((25, 50))
dino_image.fill(GRAY)
dino_crouch_image = pygame.Surface((50, 25))  # Smaller height for crouching
dino_crouch_image.fill((100, 100, 100))  # Slightly darker gray for crouching
cactus_image = pygame.Surface((10, 50))
cactus_image.fill(GREEN)
bird_image = pygame.Surface((30, 20))
bird_image.fill((50, 50, 150))  # Dark blue for the bird

# Dinosaur class
class Dinosaur:
    def __init__(self, x=100, y=GROUND_LEVEL):
        self.image = dino_image  # Default image for standing
        self.rect = self.image.get_rect(midbottom=(x, y))
        self.velocity_y = 0
        self.is_jumping = False
        self.is_crouching = False
        self.is_standing = True  # Default state is standing
        self.speed = 0  # Initialize speed attribute
        self.last_action_time = pygame.time.get_ticks()  # Initialize with the current time

    def reset_states(self, y=GROUND_LEVEL):
        """ Reset dinosaur state to avoid conflicts. """
        self.rect.bottom = y  # Set the bottom to the provided y (default is GROUND_LEVEL)
        # Reset states based on the position
        if self.rect.bottom == GROUND_LEVEL:  # Only reset if the dinosaur is standing
            self.is_crouching = False  # Reset crouching only if standing
            self.is_jumping = False  # Ensure jumping is reset if standing
            self.is_standing = True
        else:
            # If in the air, reset only the jumping state
            self.is_crouching = False
            self.is_standing = False
            self.is_jumping = True

    def update_rect(self, new_image):
        """ Update the dinosaur's image and its rectangle. """
        current_x = self.rect.x
        current_bottom = self.rect.bottom
        self.image = new_image
        self.rect = self.image.get_rect()
        self.rect.x = current_x
        self.rect.bottom = current_bottom

        # Only adjust the rect.bottom if the dinosaur is supposed to be standing
        if self.rect.bottom != GROUND_LEVEL and self.is_standing:
            self.rect.bottom = GROUND_LEVEL  # Ensure it is aligned with the ground when standing

    def update(self):
        """ Update dinosaur state based on gravity and position. """
        if self.is_jumping:  # Only apply gravity if the dinosaur is jumping
            self.velocity_y += GRAVITY  # Apply gravity
            self.rect.y += self.velocity_y

        # Prevent falling below ground
        if self.rect.bottom >= GROUND_LEVEL:
            self.rect.bottom = GROUND_LEVEL
            self.is_jumping = False
            self.is_standing = True  # After landing, the dinosaur should stand
        elif self.rect.bottom < GROUND_LEVEL and self.is_standing:
            # If the dinosaur is mistakenly standing in mid-air, reset to ground level
            self.rect.bottom = GROUND_LEVEL
            self.is_standing = False  # Set to not standing if it's mid-air, maybe stand once it reaches the ground
    
    def jump(self):
        """ Make the dinosaur jump if it's not already jumping. """
        if self.rect.bottom == GROUND_LEVEL and not self.is_crouching and not self.is_jumping:  # Ensure it's not already jumping
            self.reset_states()
            self.is_jumping = True
            self.is_crouching = False
            self.velocity_y = JUMP_STRENGTH
            self.is_standing = False  # Transition from standing to jumping
            self.last_action_time = pygame.time.get_ticks()  # Initialize with the current time

    def crouch(self):
        """ Make the dinosaur crouch if it's not already jumping. """
        if self.rect.bottom == GROUND_LEVEL and not self.is_jumping:  # Check that the dinosaur is standing and not jumping
            self.reset_states()
            self.is_crouching = True
            self.is_jumping = False
            self.update_rect(dino_crouch_image)
            self.is_standing = False  # Transition from standing to crouching
            self.last_action_time = pygame.time.get_ticks()  # Initialize with the current time

    def stand(self):
        """ Make the dinosaur stand if it's crouching. """
        if self.is_crouching:
            self.reset_states()
            self.is_crouching = False
            self.is_jumping = False
            self.update_rect(dino_image)
            self.is_standing = True  # Transition back to standing
            self.rect.bottom = GROUND_LEVEL  # Ensure it stands on the ground
            self.last_action_time = pygame.time.get_ticks()  # Initialize with the current time
        else:
            # If the dinosaur is already standing, just update the rect
            self.update_rect(dino_image)  # Ensure standing position is updated
            self.is_standing = True  # Ensure the dinosaur is marked as standing
            self.rect.bottom = GROUND_LEVEL  # Confirm it's on the ground

    def draw(self):
        """ Draw the dinosaur on the screen. """
        screen.blit(self.image, self.rect)

class Cactus:
    def __init__(self, speed):
        self.image = cactus_image  # Use cactus image here
        # Choose between two fixed heights for the cactus
        cactus_height = random.choice([30, 60])  # Choose either 30 or 60 for the cactus height
        self.image = pygame.Surface((10, cactus_height))  # Set the cactus image with the chosen height
        self.image.fill(GREEN)  # Set the cactus color
        
        # Set height offset based on cactus height
        if cactus_height == 30:
            height_offset = 10
        else:
            height_offset = 40
        
        # Position the cactus so it touches the ground
        self.rect = self.image.get_rect(midbottom=(random.randint(SCREEN_WIDTH + 20, SCREEN_WIDTH + 300), 
                                                    SCREEN_HEIGHT - cactus_height + height_offset))  # Properly adjust the height
        
        self.speed = speed  # Set the speed

    def update(self):
        self.rect.x -= self.speed  # Move the cactus with speed
        # Reset cactus position if it moves out of screen
        if self.rect.right < 0:
            self.rect.left = SCREEN_WIDTH + random.randint(20, 300)
            # Randomly choose between the two possible heights again when cactus resets
            cactus_height = random.choice([30, 60])
            self.image = pygame.Surface((10, cactus_height))  # Adjust height of the cactus
            self.image.fill(GREEN)  # Reset the color of the cactus
            
            # Set height offset based on cactus height
            if cactus_height == 30:
                height_offset = 10
            else:
                height_offset = 40

            # Adjust position for the new height so it touches the ground
            self.rect = self.image.get_rect(midbottom=(self.rect.left, SCREEN_HEIGHT - cactus_height + height_offset))

    def draw(self):
        screen.blit(self.image, self.rect)

# Bird class
class Bird:
    def __init__(self, speed):
        self.image = bird_image
        self.image.fill(YELLOW)  # Set the bird color to yellow
        self.rect = self.image.get_rect(midbottom=(random.randint(SCREEN_WIDTH + 200, SCREEN_WIDTH + 600), 
                                                    random.randint(SCREEN_HEIGHT - 200, SCREEN_HEIGHT - 50)))  # Random height within screen bounds
        self.speed = speed  # Set the speed

    def update(self):
        self.rect.x -= self.speed  # Use dynamic speed
        # Reset bird position if it moves out of screen
        if self.rect.right < 0:
            self.rect.left = SCREEN_WIDTH + random.randint(200, 600)
            self.rect.bottom = random.randint(SCREEN_HEIGHT - 200, SCREEN_HEIGHT - 50)  # Random height within screen bounds

    def draw(self):
        screen.blit(self.image, self.rect)


# Q-learning Agent class
class QLearningAgent:
    def __init__(self, state_size, action_size, epsilon=0.1, learning_rate=0.5, discount_factor=0.5, epsilon_decay=0.99, epsilon_min=0.01):
        # Initialize parameters
        self.state_size = state_size  # Tuple representing the dimensions of the state space
        self.action_size = action_size  # Number of possible actions
        self.epsilon = epsilon  # Decay epsilon over time # Epsilon for epsilon-greedy policy

        # Parameters for decaying epsilon
        self.epsilon_decay = epsilon_decay  # Rate at which epsilon decays after each episode
        self.epsilon_min = epsilon_min  # Minimum value of epsilon, to avoid epsilon becoming 0

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Calculate the total number of states by taking the product of state_size dimensions
        total_state_size = np.prod(state_size)
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((total_state_size, action_size))  # Q-table with rows for states and columns for actions

    def state_to_index(self, state):
        try:
            # Ensure all values in state are valid
            state_index = hash(state) % self.q_table.shape[0]  # Or another suitable mapping
            if state_index < 0:
                raise ValueError(f"Invalid state index: {state_index}")
            return state_index
        except Exception as e:
            print(f"Error in state_to_index: {e}")
            return -1  # Invalid index

    def get_q_value(self, state, action):
        """Returns the Q-value for a given state-action pair."""
        state_index = self.state_to_index(state)  # Map state to a unique index
        return self.q_table[state_index, action]

    def choose_action(self, state):
        state_idx = self.state_to_index(state)  # Convert state to index

        # Ensure the state_idx is a valid positive integer
        if isinstance(state_idx, float):
            state_idx = int(state_idx)  # Convert to integer if it's a float

        if state_idx < 0 or state_idx >= self.q_table.shape[0]:
            print(f"Invalid state index: {state_idx}. Skipping action choice.")
            return random.choice(range(self.action_size))  # Return random action if index is invalid
        
        # Debugging: Print the Q-values for all actions (jump, crouch, stand)
        print(f"State Index: {state_idx}, Jump Q-Value: {self.q_table[state_idx, 0]}, Crouch Q-Value: {self.q_table[state_idx, 1]}, Stand Q-Value: {self.q_table[state_idx, 2]}")
        
        # Choose the action with the highest Q-value
        action = np.argmax(self.q_table[state_idx])  # Return the action with the highest Q-value
        return action

    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-table using the Q-learning update rule."""
        state_idx = self.state_to_index(state)  # Map state to a unique index
        
        # Ensure the state_idx is a valid positive integer
        if isinstance(state_idx, float):
            state_idx = int(state_idx)  # Convert to integer if it's a float
        
        if state_idx < 0 or state_idx >= self.q_table.shape[0]:
            print(f"Invalid state index: {state_idx}. Skipping Q-table update.")
            return  # Skip the update if the index is invalid

        current_q_value = self.get_q_value(state, action)
        
        # Estimate the maximum future reward
        max_future_q = max(
            [self.get_q_value(next_state, a) for a in range(self.action_size)],
            default=0.0
        )

        # Calculate the new Q-value using the Q-learning update rule
        new_q_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q_value
        )

        print(f"State Index: {state_idx}, Action: {action}, Current Q-Value: {current_q_value}, New Q-Value: {new_q_value}")  # Debugging line

        # Update Q-table with the new Q-value
        self.q_table[state_idx, action] = new_q_value

    def get_state(self, dinosaur, cacti, birds):
        # Discretize dinosaur's position and velocity
        x = min(max(dinosaur.rect.x // 100, 0), 3)  # 0-3 range
        y = min(max(dinosaur.rect.y // 100, 0), 3)  # 0-3 range
        velocity_y = min(max(dinosaur.velocity_y // 5, -2), 2)  # -2 to 2 range

        # Nearest cactus
        if cacti:
            cactus_distances = [(c.rect.x - dinosaur.rect.right) // 100 for c in cacti if c.rect.left > dinosaur.rect.right]
            closest_cactus_dist = min(cactus_distances) if cactus_distances else 2  # Default to far
            closest_cactus_dist = min(closest_cactus_dist, 2)

            cactus_heights = [(dinosaur.rect.bottom - c.rect.top) // 100 for c in cacti if c.rect.left > dinosaur.rect.right]
            closest_cactus_height = min(cactus_heights) if cactus_heights else 0
        else:
            closest_cactus_dist = 2
            closest_cactus_height = 0

        # Nearest bird
        if birds:
            bird_distances = [(b.rect.x - dinosaur.rect.right) // 100 for b in birds if b.rect.left > dinosaur.rect.right]
            closest_bird_dist = min(bird_distances) if bird_distances else 2
            closest_bird_dist = min(closest_bird_dist, 2)

            bird_heights = [(dinosaur.rect.bottom - b.rect.top) // 100 for b in birds if b.rect.left > dinosaur.rect.right]
            closest_bird_height = min(bird_heights) if bird_heights else 0
        else:
            closest_bird_dist = 2
            closest_bird_height = 0

        # Return the state as a tuple
        return (x, y, velocity_y, closest_cactus_dist, closest_cactus_height, closest_bird_dist, closest_bird_height)

    def is_valid_index(self, idx):
        # Check if the index is within the bounds of the Q-table's shape
        return all(0 <= i < dim for i, dim in zip(idx, self.q_table.shape))

    def decay_epsilon(self):
        """Decay the epsilon (exploration rate) over time."""
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)

def check_collision(dinosaur, cacti, birds):
    """ Check if the dinosaur collides with any cacti or birds. """
    # Check collision with cacti
    for cactus in cacti:
        if dinosaur.rect.colliderect(cactus.rect):
            return True  # Collision detected

    # Check collision with birds
    for bird in birds:
        if dinosaur.rect.colliderect(bird.rect):
            return True  # Collision detected

    return False  # No collision

def update_q_table_parallel(agent, batch):
    """Process multiple state-action updates in parallel."""
    for state, action, reward, next_state in batch:
        agent.update_q_table(state, action, reward, next_state)

def train_agent(agent, episodes, save_interval, model_path):
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")

        # Reset the game state for each episode
        dinosaur = Dinosaur(x=50, y=200)
        cacti = [Cactus(speed=SPEED)]
        birds = [Bird(speed=SPEED)]
        reward = 0
        clock = pygame.time.Clock()
        running = True

        # Store state-action-reward-next_state tuples for batch processing
        batch = []

        while running:
            screen.fill(BLUE)  # Set the background color to blue

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get the current state and choose an action
            state = agent.get_state(dinosaur, cacti, birds)

            # Validate state
            if not isinstance(state, (list, tuple)):
                print(f"Invalid state format at episode {episode}, skipping.")
                continue

            action = agent.choose_action(state)

            # Apply action based on agent's decision
            if action == 0:  # Jump
                dinosaur.jump()
            elif action == 1:  # Crouch
                dinosaur.crouch()
            elif action == 2:  # Do nothing
                if dinosaur.is_crouching:
                    dinosaur.stand()

            # Update game state
            dinosaur.update()
            for cactus in cacti:
                cactus.update()
            for bird in birds:
                bird.update()

            # Check for collision
            collision = check_collision(dinosaur, cacti, birds)
            collision_reward = -500 if collision else 0.5

            # Action-based reward (penalty or reward for jumping, crouching, or standing)
            action_reward = -10 if action == 0 else -10 if action == 1 else 1.5

            # Combine rewards
            total_reward = collision_reward + action_reward
            reward += total_reward

            # Store the state-action-reward-next_state in the batch
            next_state = agent.get_state(dinosaur, cacti, birds)
            batch.append((state, action, total_reward, next_state))

            # Draw objects
            dinosaur.draw()
            for cactus in cacti:
                cactus.draw()
            for bird in birds:
                bird.draw()

            # Draw the ground
            pygame.draw.rect(screen, GROUND_COLOR, (0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 20))

            # Update display
            pygame.display.flip()

            # Control frame rate
            clock.tick(FPS)

            # Check if collision occurred to end the episode
            if collision:
                print("Game Over!")
                running = False

        # After each episode, decay epsilon
        agent.decay_epsilon()

        # If batch is full or episode is over, process updates in parallel
        if len(batch) > 0:
            with ThreadPoolExecutor(max_workers=100) as executor:
                # Submit batch processing tasks to be done in parallel
                executor.submit(update_q_table_parallel, agent, batch)

            # Clear batch after processing
            batch.clear()

        # Periodically save the model
        if (episode + 1) % save_interval == 0:
            print(f"Episode {episode + 1}: Saving model...")
            agent.save_model(model_path)

        # Print episode statistics
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {reward}, Epsilon: {agent.epsilon}")


# Main game loop
def main():
    state_space = (4, 4, 5, 3, 3, 3, 3)
    action_space = 3
    agent = QLearningAgent(state_space, action_space)

    episodes = 1000
    save_interval = 10
    model_path = "q_learning_model.pkl"

    # Train the agent
    train_agent(agent, episodes, save_interval, model_path)

    print("Training completed for all agents.")


# Run the game
if __name__ == "__main__":
    main()