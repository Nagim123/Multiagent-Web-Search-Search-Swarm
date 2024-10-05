"""
File that test implemented agent and computes final matrics.
"""

import argparse
import logging

from agents.test_primitive_agent import PrimitiveAgent
from agents.search_swarm_1 import SearchSwarm
from environments.webshop_env import WebAgentSiteEnv

def main():
    # Add arguments for running this file
    parser = argparse.ArgumentParser("Test implemented agent on the webshop environment.")
    parser.add_argument("-e", "--max_episodes", type=int, required=True, help="Maximum number of episodes for agent to find a product.")
    parser.add_argument("-c", "--instruction_count", type=int, required=True, help="Number of testing instructions.")
    parser.add_argument("-d", "--debug", action="store_true", default=False, help="Show debugging information")

    # Parse arguments
    args = parser.parse_args()
    max_episodes = args.max_episodes
    testing_instructions_count = args.instruction_count
    
    
    # Logging
    logger = logging.getLogger("testing_framework_test_py")
    if args.debug:
        print("!")
        logging.basicConfig(level=logging.INFO)

    logging.info("Start testing the agent!")
    logging.info(f"Max number of episodes is {max_episodes}")
    logging.info(f"Number of testing instructions is {testing_instructions_count}")

    agent = SearchSwarm() # Change testing agent here!
    env = WebAgentSiteEnv(observation_mode="text")

    try:
        # List to keep the reward of each instruction.
        all_rewards = []
        for i in range(testing_instructions_count):
            # Get initial observation and valid actions.
            observation, _ = env.reset(idx=i)
            valid_actions = env.get_available_actions()
            logger.info(f"OBSERVATION: {observation}")
            logger.info(f"VALID ACTIONS: {valid_actions}")
            
            episode = 0
            instruction_done = False
            # Run the agent until number of episodes exceeds limit or the task is done.
            while episode < max_episodes and not instruction_done:
                # Get action of agent.
                action = agent.act(observation, valid_actions)
                logger.info(f"Agent decided to {action}")
                # Update the environment.
                observation, reward, instruction_done, info = env.step(action)
                valid_actions = env.get_available_actions()
                logger.info(f"OBSERVATION: {observation}")
                logger.info(f"VALID ACTIONS: {valid_actions}")
                episode += 1
            if instruction_done:
                logger.info(f"Agent bought a product and got {reward} reward")
                all_rewards.append(reward)
            else:
                logger.info("Agent failed to find an appropriate product!")
                all_rewards.append(0)
    except Exception as e:
        raise e
    finally:
        # Be sure to stop chrome processes and the agent itself
        env.close()
        agent.stop()

    # Calculate metrics
    mean_reward = sum(all_rewards) / len(all_rewards)
    success_rate = sum([(reward == 1) for reward in all_rewards]) / len(all_rewards)

    print(f"RESULTS OVER {testing_instructions_count} INSTRUCTIONS:")
    print(f"REWARD: {mean_reward * 100}")
    print(f"SUCCESS RATE: {success_rate * 100}")


if __name__ == "__main__":
    main()