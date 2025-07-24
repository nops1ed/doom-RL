import numpy as np
import cv2
from collections import deque
from vizdoom import DoomGame, ScreenResolution, ScreenFormat
import gym
from gym import spaces


class SimpleDoomEnv(gym.Env):
    
    def __init__(self, config_path='scenarios/basic.cfg', frame_stack=4, frame_skip=4, render=False):
        super(SimpleDoomEnv, self).__init__()
        
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        
        self.game = DoomGame()
        self.game.load_config(config_path)
        
        self.game.set_screen_resolution(ScreenResolution.RES_320X240) 
        self.game.set_screen_format(ScreenFormat.RGB24) 
        
        self.game.set_window_visible(render)
        self.game.set_render_hud(render)
        self.game.set_render_crosshair(render)
        self.game.set_render_weapon(render)
        self.game.set_render_decals(render)
        self.game.set_render_particles(render)
        
        self.game.init()
        
        self.n_actions = self.game.get_available_buttons_size()
        if self.n_actions == 0:
            self.n_actions = 3  # MOVE_LEFT, MOVE_RIGHT, ATTACK
            
        self.action_space = spaces.Discrete(self.n_actions)
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(frame_stack, 84, 84),
            dtype=np.float32
        )
        
        self.frames = deque(maxlen=frame_stack)
        
    def _preprocess_frame(self, frame):
        if frame is None:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        frame = cv2.resize(frame, (84, 84))
        
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def reset(self):
        self.game.new_episode()
        
        state = self.game.get_state()
        if state is not None:
            frame = state.screen_buffer
        else:
            frame = None
            
        frame = self._preprocess_frame(frame)
        
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        
        return np.stack(self.frames, axis=0)
    
    def step(self, action):
        actions = np.zeros(self.n_actions)
        actions[action] = 1
        
        reward = self.game.make_action(actions.tolist(), self.frame_skip)
        
        done = self.game.is_episode_finished()
        if not done:
            state = self.game.get_state()
            if state is not None:
                frame = state.screen_buffer
            else:
                frame = None
            frame = self._preprocess_frame(frame)
            self.frames.append(frame)
            next_state = np.stack(self.frames, axis=0)
        else:
            next_state = np.zeros((self.frame_stack, 84, 84), dtype=np.float32)
        
        return next_state, reward, done, {}
    
    def close(self):
        self.game.close()
    
    def render(self, mode='human'):
        if mode == 'human':
            state = self.game.get_state()
            if state is not None:
                cv2.imshow('Doom', state.screen_buffer)
                cv2.waitKey(1)


class SimpleDoomWrapper:
    def __init__(self, config_path='scenarios/basic.cfg', **kwargs):
        self.env = SimpleDoomEnv(config_path, **kwargs)
        
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        self.env.close()
    
    def render(self):
        self.env.render()
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space