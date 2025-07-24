import numpy as np
from collections import deque
import vizdoom

class CustomRewardDoomEnv:
    def __init__(self, scenario_path, frame_stack=4, frame_skip=4, render=False):
        self.game = vizdoom.DoomGame()
        self.game.load_config(scenario_path)
        
        self.game.add_available_game_variable(vizdoom.GameVariable.KILLCOUNT)
        self.game.add_available_game_variable(vizdoom.GameVariable.HEALTH)
        self.game.add_available_game_variable(vizdoom.GameVariable.AMMO2)
        self.game.add_available_game_variable(vizdoom.GameVariable.DAMAGECOUNT)
        
        if not render:
            self.game.set_window_visible(False)
            self.game.set_m
        self.game.init()
        
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.frame_buffer = deque(maxlen=frame_stack)
        
        self.reward_weights = {
            'kill': 100.0,
            'damage_taken': -5.0,
            'ammo_gain': 1.0,
            'health_gain': 2.0,
            'survival': 10.0,
            'time_penalty': -0.1
        }
        
        self.prev_health = None
        self.prev_ammo = None
        self.prev_kills = None
        self.prev_damage = None
        self.episode_start_time = None
        
    def reset(self):
        self.game.new_episode()
        
        state = self.game.get_state()
        self.prev_health = state.game_variables[1]  # HEALTH
        self.prev_ammo = state.game_variables[2]    # AMMO2
        self.prev_kills = state.game_variables[0]   # KILLCOUNT
        self.prev_damage = state.game_variables[3]  # DAMAGECOUNT
        self.episode_start_time = self.game.get_episode_time()
        
        frame = self._preprocess_frame(state.screen_buffer)
        for _ in range(self.frame_stack):
            self.frame_buffer.append(frame)
            
        return np.stack(self.frame_buffer, axis=0)
    
    def step(self, action):
        reward = 0
        
        if isinstance(action, (int, np.integer)):
            action_list = [0] * self.game.get_available_buttons_size()
            action_list[action] = 1
        else:
            action_list = list(action)
            
        for _ in range(self.frame_skip):
            self.game.make_action(action_list)
            
            if self.game.is_episode_finished():
                break
        
        if self.game.is_episode_finished():
            return None, 0, True, {}
        
        custom_reward = self._calculate_custom_reward()
        
        state = self.game.get_state()
        frame = self._preprocess_frame(state.screen_buffer)
        self.frame_buffer.append(frame)
        new_state = np.stack(self.frame_buffer, axis=0)
        
        return new_state, custom_reward, False, {}
    
    def _calculate_custom_reward(self):
        state = self.game.get_state()
        game_vars = state.game_variables
        
        current_health = game_vars[1]
        current_ammo = game_vars[2]
        current_kills = game_vars[0]
        current_damage = game_vars[3]
        
        reward = 0
        
        if current_kills > self.prev_kills:
            reward += self.reward_weights['kill'] * (current_kills - self.prev_kills)
        
        if current_damage > self.prev_damage:
            reward += self.reward_weights['damage_taken'] * (current_damage - self.prev_damage)
        
        if current_ammo > self.prev_ammo:
            reward += self.reward_weights['ammo_gain'] * (current_ammo - self.prev_ammo)
        
        if current_health > self.prev_health:
            reward += self.reward_weights['health_gain'] * (current_health - self.prev_health)
        
        if self.game.get_episode_time() - self.episode_start_time > 50:
            reward += self.reward_weights['survival']
        
        reward += self.reward_weights['time_penalty']
        
        self.prev_health = current_health
        self.prev_ammo = current_ammo
        self.prev_kills = current_kills
        self.prev_damage = current_damage
        
        return reward
    
    def _preprocess_frame(self, frame):
        if len(frame.shape) == 3:
            frame = np.mean(frame, axis=0)  # Average across color channels
        
        # Resize to 84x84
        from scipy.ndimage import zoom
        if frame.shape != (84, 84):
            zoom_factor = 84 / frame.shape[0]
            frame = zoom(frame, zoom_factor)
            
        return frame.astype(np.float32) / 255.0
    
    def render(self):
        pass

    def close(self):
        self.game.close()
    
    @property
    def action_space(self):
        class ActionSpace:
            def __init__(self, n):
                self.n = n
        return ActionSpace(self.game.get_available_buttons_size())
    
    def get_available_actions(self):
        return [list(a) for a in self.game.get_available_actions()]

if __name__ == "__main__":
    env = CustomRewardDoomEnv("scenarios/basic_improved.cfg")
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    for i in range(10):
        action = np.random.randint(0, env.action_space)
        state, reward, done, _ = env.step([action])
        print(f"Step {i}: reward = {reward}")
        if done:
            break
    
    env.close()