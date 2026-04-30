import vizdoom as vzd
import numpy as np

# Initialize Game
game = vzd.DoomGame()
game.load_config("scenarios/my_way_home.cfg") # Example scenario
game.set_mode(vzd.Mode.SPECTATOR) # HUMAN PLAYER MODE
game.init()

actions = []
states = []
rewards = []

for i in range(10): # Collect 10 episodes
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        
        # 1. Get Screen Buffer (Image Data)
        screen = state.screen_buffer
        
        # 2. Get Game Variables (Position, Health, Ammo)
        game_vars = state.game_variables
        
        # 3. Get Human Action (Keyboard/Mouse)
        # ViZDoom SPECTATOR mode automatically records keypresses
        # during game.make_action
        
        # In spectator mode, we don't pass an action, we read it.
        # This requires setting up input capture if not using
        # standard SPECTATOR mode recording tools.
        
        game.advance_action() 
        last_action = game.get_last_action()
        last_reward = game.get_last_reward()
        # Store state and action for training
        states.append(screen)
        actions.append(last_action)
        rewards.append(last_reward)

    print(f"Episode {i} finished. Reward: {game.get_total_reward()}")

# Save data for Imitation Learning
np.save("dataset/human_states.npy", np.array(states))
np.save("dataset/human_actions.npy", np.array(actions))
np.save("dataset/human_rewards.npy", np.array(rewards))

game.close()
