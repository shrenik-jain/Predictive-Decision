import gym
import torch
import os
import csv
import numpy as np
import argparse
from planner import Planner
from predictor import Predictor, Decoder
from observation import observation_adapter
from smarts.core.utils.episodes import episodes

# build agent
class Policy:
    def __init__(self, model, use_interaction=False, render=False, decoder_type='transformer'):
        # set device based on availability of GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = Predictor(use_interaction)
        # load the model based on the decoder type
        if decoder_type in ['lstm', 'gru']:
            self.predictor.load_state_dict(torch.load(model, map_location=self.device))
        elif decoder_type == 'transformer':    
            self.predictor.decoder = Decoder(use_interaction)
            self.load_transformer_decoder(model)
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
        self.predictor.eval()
        self.planner = Planner(self.predictor, use_interaction, render)
    
    def load_transformer_decoder(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device)
        filtered_dict = {k: v for k, v in state_dict.items() if not (k.startswith('decoder.cell') or 'gru' in k.lower())}
        self.predictor.load_state_dict(filtered_dict, strict=False)

    def act(self, obs, env_input):
        actions = self.planner.plan(obs, env_input)
        return actions

scenarios = [
    "1_to_2lane_left_turn_c",
    "3lane_merge_single_agent",
    "3lane_overtake"
]

def main(args):
    log_path = f"./test_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    success_rate = []
    test_epoch = 0
    policy = Policy(args.model_path, use_interaction=args.use_interaction, decoder_type=args.decoder)

    with open(log_path+"test_log.csv", 'w') as csv_file: 
        writer = csv.writer(csv_file) 
        writer.writerow(['episodes', 'scenario', 'success', 'safety', 'episode_len', 'success_rate'])

    for scene in scenarios:
        print(f'============={scene}=============')
        env = gym.make('smarts.env:multi-scenario-v0', scenario=scene, 
                       headless=not args.envision_gui, sumo_headless=not args.sumo_gui)
        observer = observation_adapter(env=env, num_neighbors=5)

        eval_episodes = args.episodes

        for episode in episodes(n=eval_episodes):
            observations = env.reset()
            observer.reset()
            episode.record_scenario(env.scenario_log)
            episode_len = 0

            dones = {"__all__": False}
            while not dones["__all__"]:
                env_input = observer(observations['Agent_0'])
                actions = policy.act(observations['Agent_0'], env_input)

                # execute
                for t in range(5):
                    action = {'Agent_0': actions[t]}
                    observations, rewards, dones, infos = env.step(action)
                    observer(observations['Agent_0'])
                    episode.record_step(observations, rewards, dones, infos)
                    episode_len += 1
                    if dones["__all__"]:
                        break
                        
            test_epoch += 1

            if episode_len >= 5:
                with open(log_path+"test_log.csv", 'a') as csv_file: 
                    writer = csv.writer(csv_file) 
                    success = observations['Agent_0'].events.reached_goal
                    safety = any(observations['Agent_0'].events.collisions)
                    success_rate.append(1 if success else 0)
                    writer.writerow([test_epoch, scene, success, safety, episode_len, np.mean(success_rate)])

        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--name', type=str, help='log name (default: "Test1")', default="Test1")
    parser.add_argument('--episodes', type=int, help='test episodes (default: 50)', default=50)
    parser.add_argument('--model_path', type=str, help='path to the saved model')
    parser.add_argument('--use_interaction', action='store_true', help='whether using interaction-aware prediction', default=False)
    parser.add_argument('--envision_gui', action='store_true', help='visualize in envision', default=False)
    parser.add_argument('--sumo_gui', action='store_true', help='visualize in sumo', default=False)
    parser.add_argument('--decoder', type=str, default='transformer', choices=['gru', 'transformer', 'lstm'], help='Decoder architecture to use')
    args = parser.parse_args()

    main(args)
