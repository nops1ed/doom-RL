import numpy as np
import cv2
from collections import deque
from vizdoom import DoomGame, Mode, ScreenResolution, ScreenFormat, GameVariable
import gym
from gym import spaces


class DoomEnvironment(gym.Env):
    
    def __init__(self, config_path, frame_stack=4, frame_skip=4):
        super(DoomEnvironment, self).__init__()
        
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        
        self.game = DoomGame()
        self.game.load_config(config_path)
        
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(False)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        
        self.game.init()
        
        self.n_actions = self.game.get_available_buttons_size()
        self.action_space = spaces.Discrete(self.n_actions)
        
        self.screen_height = 480
        self.screen_width = 640
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(frame_stack, 84, 84),
            dtype=np.float32
        )
        
        self.frames = deque(maxlen=frame_stack)
        
    def _preprocess_frame(self, frame):
        frame = cv2.resize(frame, (84, 84))
        # 归一化
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def _get_state(self):
        state = self.game.get_state()
        if state is None:
            return np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
        return state.screen_buffer
    
    def reset(self):
        self.game.new_episode()
        
        frame = self._get_state()
        frame = self._preprocess_frame(frame)
        
        for _ in range(self.frame_stack):
            self.frames.append(frame)
        
        return np.stack(self.frames, axis=0)
    
    def step(self, action):
        doom_action = [0] * self.n_actions
        doom_action[action] = 1
        
        reward = self.game.make_action(doom_action, self.frame_skip)
        
        done = self.game.is_episode_finished()
        if not done:
            frame = self._get_state()
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


class DoomEnvironmentWrapper:
    def __init__(self, config_path, **kwargs):
        self.env = DoomEnvironment(config_path, **kwargs)
        
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