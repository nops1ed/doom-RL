import argparse
import os
import torch
import numpy as np
import cv2
from envs.custom_reward_env import CustomRewardDoomEnv as DoomEnvironmentWrapper
from agents.dqn_agent import DQNAgent


def test_doom_agent():
    parser = argparse.ArgumentParser(description='Test DQN agent for Doom')
    parser.add_argument('--config', type=str, default='scenarios/basic.cfg',
                       help='Doom config file path')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to test')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Save video to file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return
    
    config = {
        'dueling': True,
        'prioritized_replay': False
    }
    config_path = os.path.join(os.path.dirname(__file__), 'scenarios', 'basic.cfg')
    env = DoomEnvironmentWrapper(config_path, frame_stack=4, render=args.render)
    
    state_shape = (4, 84, 84) 
    
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions, config)
    
    agent.load(args.model)
    agent.epsilon = 0.0  # 测试时不使用随机探索
    
    print(f"Testing model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Environment: {args.config}")
    
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, 30.0, (640, 480)
        )
    
    scores = []
    
    try:
        for episode in range(args.episodes):
            state = env.reset()
            episode_score = 0
            
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            
            while True:
                if args.render:
                    env.render()
                
                if video_writer is not None:
                    frame = env.get_raw_frame()
                    if frame is not None:
                        frame = cv2.resize(frame, (640, 480))
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        video_writer.write(frame)
                
                action = agent.act(state, training=False)
                
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                episode_score += reward
                
                if done:
                    break
            
            scores.append(episode_score)
            print(f"Score: {episode_score:.2f}")
    
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    
    finally:
        if video_writer is not None:
            video_writer.release()
        
        env.close()
        
        if scores:
            print(f"\nTesting completed!")
            print(f"Average Score: {np.mean(scores):.2f}")
            print(f"Std Deviation: {np.std(scores):.2f}")
            print(f"Min Score: {np.min(scores):.2f}")
            print(f"Max Score: {np.max(scores):.2f}")


if __name__ == "__main__":
    test_doom_agent()