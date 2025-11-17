# This python file includes the training and testing functions for the GRL model
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time

def Training_GRLModels(GRL_model, n_episodes, max_episode_len, save_dir, debug,  gym_instance, warmup, server):
    """
        This function is a training function for the GRL model

        Parameter description:
        --------
        GRL_model: the GRL model to be trained
        env: the simulation environment registered to gym
        n_episodes: the number of training rounds
        max_episode_len: the maximum number of steps to train in a single step
        save_dir: path to save the model
        warmup: model free exploration steps (randomly selected actions)
        debug: model parameters related to debugging
    """
    # The following is the model training process
    Rewards = []  # Initialize the reward matrix for data saving
    Loss = []  # Initialize the Loss matrix for data storage
    Episode_Steps = []  # Initialize the step matrix to hold the step length at task completion for each episode
    writer = SummaryWriter(os.path.join(save_dir,'logs_train'))
    
    print("#------------------------------------#")
    print("#----------Training Begins-----------#")
    print("#------------------------------------#")

    gym = gym_instance.gym
    sim = gym_instance.sim
    dt  = gym_instance.dt
    gym_instance.step_physics() 
    action_scale = gym_instance.action_scale

    t_warmup        = 0
    R_warmup        = 0
    t               = 0
    R               = 0
    i               = 0
    success_count = 0
    fail    = 0
    total_steps = 0
    
    while i <= n_episodes :

        obs = gym_instance.init_episode()
        # previous_action = [0.0] * 6

        t_0 = gym.get_sim_time(sim)
        done    = False
        success = False
        R = 0
        t = 0
        
        while not done and not success:
            
            gym_instance.step_physics()  
            t_now = gym.get_sim_time(sim)
            
            action, prob, val = GRL_model.choose_action(obs)
            detached_action = action.detach().cpu().data.numpy().astype(np.float32)
            obs_next, reward, done = gym_instance.step(detached_action)
            
            
            R += reward
            t += 1
            total_steps += 1

            # ------Storing interaction results in PPOMemory------ #
            GRL_model.store_transition(obs, action, prob, val, reward, done)

            # ------Policy update------ #
            if total_steps % GRL_model.update_interval == 0:
                GRL_model.learn()

            # ------Observation update------ #
            obs = obs_next
            
            gym_instance.render()

        # ------ Records training data ------ #
        # Get the training data
        training_data = GRL_model.get_statistics()
        loss = training_data[0]
        # Recording training data
        Rewards.append(R)
        Episode_Steps.append(t)
        Loss.append(loss)
        
        writer.add_scalar('Reward/episode', R, i)
        writer.add_scalar('Loss/episode', loss, i)
    
        if i % 1 == 0:
            print('Training Episode:', i, 'Reward:', R, 'Loss:', loss)

        if i % 100 == 0 and i != 0:
            # Save model
            GRL_model.save_model(save_dir)
            # Save other data
            np.save(save_dir + "/Rewards_" + str(i), Rewards)
            np.save(save_dir + "/Episode_Steps_" + str(i), Episode_Steps)
            np.save(save_dir + "/Loss_"+ str(i), Loss)

        i += 1

        
    print('Training Finished.')

    
def Testing_GRLModels(GRL_model, env, test_episodes, load_dir, debug):
    """
        This function is a test function for a trained GRL model

        Parameters:
        --------
        GRL_Net: the neural network used in the GRL model
        GRL_model: the GRL model to be tested
        env: the simulation environment registered to gym
        test_episodes: the number of rounds to be tested
        load_dir: path to read the model
        debug: debug-related model parameters
    """
    # Here is how the model is tested
    Rewards = [] # Initialize the reward matrix for data storage

    GRL_model.load_model(load_dir)

    print("#-------------------------------------#")
    print("#-----------Testing Begins------------#")
    print("#-------------------------------------#")

    for i in range(1, test_episodes + 1):
        if debug:
            print("#------------------------------------#")
            for parameters in GRL_model.actormodel.parameters():
                print("param:", parameters)
            print("#------------------------------------#")
        obs = env.reset()
        R = 0
        t = 0
        done = False
        while not done:
            action, prob, val = GRL_model.choose_action(obs)
            obs, reward, done, info = env.step(action)
            R += reward
            t += 1

        # Record rewards
        Rewards.append(R)
        print('Evaluation Episode:', i, 'Reward:', R)
    print('Evaluation Finished')

    # Test data storage
    np.save(load_dir + "/Test_Rewards", Rewards)
