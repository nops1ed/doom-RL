import os
import time
import numpy as np
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from envs.custom_reward_env import CustomRewardDoomEnv as DoomEnvironmentWrapper
from agents.dqn_agent import DQNAgent


def plot_training_progress(scores, losses, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(scores)
    ax1.set_title('Episode Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    
    if losses:
        ax2.plot(losses)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_doom_agent():
    parser = argparse.ArgumentParser(description='Train DQN agent for Doom')
    parser.add_argument('--config', type=str, default='scenarios/basic_improved.cfg',
                       help='Doom config file path')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory to save logs')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"MPS: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA: {device}")
    else:
        device = torch.device("cpu")
        print(f"CPU: {device}")

    writer = SummaryWriter(args.log_dir)
    
    config = {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 100000,
        'target_update': 1000,
        'learning_starts': 1000,
        'dueling': True,
        'prioritized_replay': True,
        'device': device
    }
    
    env = DoomEnvironmentWrapper(args.config, render=args.render)
    
    state_shape = (4, 84, 84)  # 4帧堆叠
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions, config)
    
    scores = []
    losses = []
    best_score = -float('inf')
    
    print(f"Starting training for {args.episodes} episodes...")
    print(f"Environment: {args.config}")
    print(f"Action space: {n_actions}")
    
    try:
        for episode in tqdm(range(args.episodes), desc="Training"):
            state = env.reset()
            episode_score = 0
            episode_losses = []
            
            while True:
                if args.render:
                    env.render()
                
                action = agent.act(state)
                
                next_state, reward, done, _ = env.step(action)
                
                agent.remember(state, action, reward, next_state, done)
                
                loss = agent.update()
                if loss is not None:
                    episode_losses.append(loss)
                
                state = next_state
                episode_score += reward
                
                if done:
                    break
            
            scores.append(episode_score)
            if episode_losses:
                losses.extend(episode_losses)
            
            writer.add_scalar('Episode/Score', episode_score, episode)
            writer.add_scalar('Episode/Epsilon', agent.epsilon, episode)
            if episode_losses:
                writer.add_scalar('Episode/Loss', np.mean(episode_losses), episode)
            
            if episode_score > best_score:
                best_score = episode_score
                agent.save(os.path.join(args.save_dir, 'best_model.pth'))
            
            if (episode + 1) % 100 == 0:
                agent.save(os.path.join(args.save_dir, f'checkpoint_ep{episode+1}.pth'))
                
                plot_training_progress(
                    scores, losses,
                    os.path.join(args.log_dir, f'progress_ep{episode+1}.png')
                )
            
            if (episode + 1) % 50 == 0:
                avg_score = np.mean(scores[-50:])
                print(f"\nEpisode {episode+1}/{args.episodes}")
                print(f"Average Score (last 50): {avg_score:.2f}")
                print(f"Epsilon: {agent.epsilon:.4f}")
                print(f"Best Score: {best_score:.2f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        agent.save(os.path.join(args.save_dir, 'final_model.pth'))
        
        plot_training_progress(
            scores, losses,
            os.path.join(args.log_dir, 'final_progress.png')
        )
        
        env.close()
        writer.close()
        
        print(f"\nTraining completed!")
        print(f"Final model saved to: {os.path.join(args.save_dir, 'final_model.pth')}")
        print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    train_doom_agent()