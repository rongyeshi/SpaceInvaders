#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, time

from keras.layers import Input
#from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten,Lambda, Add,MaxPooling2D
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model
from keras.optimizers import Adam
import random
from keras import backend as K
import random
import cv2


Learning_rate = 0.00001
Gama_glob =0.99
Targ_Update = 10000

input_dim =84
MINIBATCH = 32

class QNetwork():

    def __init__(self, environment_name):
        
        self.env = gym.make(environment_name)
        self.Num_input = len(self.env.observation_space.high)
        self.Num_output = self.env.action_space.__dict__['n']#get the number of action
        
        self.learning_rate = Learning_rate
        
        
        print(self.learning_rate,Gama_glob,Targ_Update)
        
        
        
        Flag = 2
        print("Flag",Flag)
        if Flag==0:
            
            
            inputs = Input(shape = (84,84,4))
            net = Conv2D(kernel_size=8, strides=4, filters=16, padding='same',
                         activation='relu')(inputs)#(resize)
            net = Flatten()(net)
            net = Dense(8, activation='relu')(net)
        
            self.outputs = Dense(self.Num_output, activation='linear')(net)
        if Flag==1:
            
            inputs = Input(shape = (84,84,4))
            net = Conv2D(kernel_size=8, strides=4, filters=16, padding='same',
                         activation='relu')(inputs)
            
            net = Conv2D(kernel_size=4, strides=2, filters=32, padding='same',
                         activation='relu')(net)
            
            net = Flatten()(net)
            net = Dense(256, activation='relu')(net)
        
            self.outputs = Dense(self.Num_output, activation='linear')(net)
        
        if Flag==2:
            
            inputs = Input(shape = (input_dim,input_dim,4))
            net = Conv2D(kernel_size=8, strides=4, filters=32, padding='same',
                         activation='relu')(inputs)
            net = Conv2D(kernel_size=4, strides=2, filters=64, padding='same',
                         activation='relu')(net)
            net = Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                         activation='relu')(net)
            net = Flatten()(net)
            net = Dense(512, activation='relu')(net)
        
            self.outputs = Dense(self.Num_output, activation='linear')(net)
        
        
        
        
        self.model = Model(inputs = inputs, outputs = self.outputs)
        self.model.compile(loss='mse', optimizer=Adam(lr=Learning_rate))
        print("network constructed")
        
        self.path_model = 'model_linear.h5'
        self.path_weight = 'weight_linear.h5'
        

    def save_model_weights(self, suffix):
        
        self.model.save(self.path_model)
        self.model.save_weights(self.path_weight)

    def load_model(self, model_file):
        
        self.model=load_model(model_file)
        

    def load_model_weights(self,weight_file):
        
        pass


        


class Replay_Memory():
    
    def __init__(self, memory_size=110000, burn_in=30000):
        self.memory_size = memory_size
        self.memory = []
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)
        return minibatch

    def append(self, transition):	
        self.memory.append(transition)
        if len(self.memory) > self.memory_size:
            self.memory=self.memory[1:]




class DQN_Agent():
    

    def __init__(self, environment_name, render=False):

        
        self.env = gym.make(environment_name)
        self.Num_input = len(self.env.observation_space.high)
        self.Num_output = self.env.action_space.__dict__['n']#get the number of action
        self.if_render = render
        self.environment_name = environment_name
        
            
        if environment_name == 'SpaceInvaders-v0':
            self.iteration = float("inf")
            self.episode = 50000
        
        self.QNet = QNetwork(environment_name)
        self.QNet2 = QNetwork(environment_name)
        
        if self.environment_name=='SpaceInvaders-v0':
            self.replay_memory = Replay_Memory()
        
        
        
        
        

    def epsilon_greedy_policy(self, q_values, epsilon):
        
        non_max = epsilon/self.Num_output
        max_greedy = 1 - epsilon + epsilon/self.Num_output
        max_inx = np.argmax(q_values)
        
        elements = list(range(self.Num_output))
        probabilities = [non_max] * self.Num_output
        probabilities[max_inx] = max_greedy
        action = np.random.choice(elements, 1, p=probabilities) # sample from the distribution
        return action[0]
        

    def greedy_policy(self, q_values):
        
        return np.argmax(q_values)

    def train(self):

        
        itera = 0
        episode = 0
        decay = (0.8 - 0.1)/300000
        epsilon= 0.8
        flag = 0
        perform_cross_time=[]
        max_one = 0.0
                
        old_Model = QNetwork(self.environment_name)
        old_Model2 = QNetwork(self.environment_name)
        model_list = [1,1]
        model_list2 = [1,1]
        average_reward_episode = []
        average_step = []
        average_loss = []
        BATCH_num = MINIBATCH 
        
        inputs = np.zeros((BATCH_num, input_dim,input_dim,4)) 
        targets = np.zeros((inputs.shape[0], self.Num_output)) 
        action_cont = 0
        MM = Targ_Update
        
        
        
        model_file = 'save/SpaceInvaders-burnin.h5'
        self.QNet.load_model(model_file)
        self.QNet2.load_model(model_file[:-3]+'_prime.h5')
        old_Model.load_model(model_file)
        old_Model2.load_model(model_file[:-3]+'_prime.h5')
        
        
        while True: ##episode
            
            episode +=1 
            initial_state = self.env.reset()
                        
            total_reward = 0
            num_steps = 0
            discount = 1
            gamma =Gama_glob#0.5#0.99
            action_cont = 0

            cur_state = initial_state
            frames_4 = [rgb2gray(cur_state)]
            reward_list = []
            cur_action = self.env.action_space.sample()
            M = 4
            state_buff = []
            old_Model.model.set_weights(self.QNet.model.get_weights())#Q1   
            old_Model2.model.set_weights(self.QNet.model.get_weights())#Q2          
            
            if itera > self.iteration or episode > self.episode:
                break
            
            
            while True:
                
                action_cont += 1
                nextstate, reward, is_terminal, debug_info = self.env.step(cur_action)
                
                frames_4.append(rgb2gray(nextstate))
                reward_list.append(reward)
            
                if len(frames_4) > 4:
                    
                    epsilon = max(epsilon-decay, 0.1)
                    itera += 1
                    
                    
                    
                    #-------------here is the state feature function S'
                    state_stack = np.asarray(frames_4[:-1])
                    frames_4 = frames_4[1:]
                    state_stack = np.swapaxes(state_stack,0,2)
                    state_stack = np.swapaxes(state_stack,0,1)#S'
                    
                    state_buff.append(state_stack)
                    
                    if len(state_buff)>1:# we save new transition into the memory
                        old_state_stack = state_buff[0]#S
                        
                        self.replay_memory.append([old_state_stack,cur_action,reward,state_stack,is_terminal]) #state, action, reward, next state, terminal flag tuples.
                        state_buff = state_buff[1:]
                        reward_list = reward_list[1:]
                        
                        #-------------here to find out next action given current state_stack A'
                        
                        q_values = self.QNet.model.predict(x = np.expand_dims(state_stack, axis=0))
                        q_values2 = self.QNet2.model.predict(x = np.expand_dims(state_stack, axis=0))
                        
                        
                        if action_cont%4==0:
                            next_action = self.epsilon_greedy_policy(np.add(q_values[0], q_values2[0]), epsilon = epsilon)# A'
                        
                        
                        ###....--> begin learning
                        
                        minibatch = self.replay_memory.sample_batch(BATCH_num)
                        coin_flip = random.uniform(0, 1)
                        
                        if coin_flip>=0.5:
                            
                            for i in list(range(0,len(minibatch))):
                                state_t, action_t,reward_t, nextstate_t, terminal_t = minibatch[i]
                                inputs[i] = state_t
                                targets[i] = self.QNet.model.predict(x = np.expand_dims(state_t, axis=0))[0]
                                Q_sa = self.QNet.model.predict(x = np.expand_dims(nextstate_t, axis=0))[0]#use previous weight for stability
                                Q_sa2= old_Model2.model.predict(x = np.expand_dims(nextstate_t, axis=0))[0][np.argmax(Q_sa)]
                                
                                if terminal_t:
                                    targets[i, action_t]= reward_t
                                if not terminal_t:
                                    targets[i, action_t]= reward_t + gamma * Q_sa2
                        
                        
                            loss = self.QNet.model.train_on_batch(inputs, targets)
                            model_list.append(1)
                        
                            
                            if len(model_list)>Targ_Update-1:
                            
                                old_Model.model.set_weights(self.QNet.model.get_weights())
                                model_list = []
                        else:
                            
                            for i in list(range(0,len(minibatch))):
                                state_t, action_t,reward_t, nextstate_t, terminal_t = minibatch[i]
                            
                                inputs[i] = state_t
                            
                            
                                targets[i] = self.QNet2.model.predict(x = np.expand_dims(state_t, axis=0))[0]
                                
                                Q_sa2 = self.QNet2.model.predict(x = np.expand_dims(nextstate_t, axis=0))[0]#use previous weight for stability
                                Q_sa= old_Model.model.predict(x = np.expand_dims(nextstate_t, axis=0))[0][np.argmax(Q_sa2)]
                                
                            
                    
                                if terminal_t:
                                    targets[i, action_t]= reward_t
                                if not terminal_t:
                                    targets[i, action_t]= reward_t + gamma * Q_sa
                        
                        
                            loss = self.QNet2.model.train_on_batch(inputs, targets)
                            model_list2.append(1)
                        
                            
                            if len(model_list2)>Targ_Update-1:
                            
                                old_Model2.model.set_weights(self.QNet.model.get_weights())
                                model_list = []
                        
                        
                        
                        ###....<--
                        if action_cont%4==0:
                            cur_action=next_action
                            action_cont=0
                        average_loss.append(loss)
                        
                        if num_steps % 30 ==0:
                            cpy_ave_loss = sum(average_loss) / float(len(average_loss))
                            print("Average loss:",cpy_ave_loss)                    
                            average_loss=[]
                        
                    else:
                        
                        pass
                        
                    
                
                
                total_reward += reward
                num_steps += 1
                cur_state = nextstate

                if is_terminal:
                    break
                
            average_reward_episode.append(total_reward)
            average_step.append(num_steps)
            
            if  episode % int(self.episode/3.0)==0:
                    self.QNet.model.save("save/"+self.environment_name+"DRM"+str(episode)+".h5")
                    self.QNet2.model.save("save/"+self.environment_name+"DRM"+str(episode)+"_prime.h5")
            
            
            
            if episode % 1==0:
                self.QNet.model.save("save/"+self.environment_name+"DRM.h5")
                self.QNet2.model.save("save/"+self.environment_name+"DRM_prime.h5")
                
                kk = self.test_in_train()
                if kk > max_one:
                    self.QNet.model.save("save/"+self.environment_name+str(MM)+"DRMbest.h5")
                    self.QNet2.model.save("save/"+self.environment_name+str(MM)+"DRMbest_prime.h5")
                    max_one = kk
                print("gamme = ",gamma,'target update=',MM,'max one:',max_one)
                
                
                
                    
                if False:
                    if episode > 1100 and kk >- 110:
                        print("Do decay!")
                        self.QNet.learning_rate = min(self.QNet.learning_rate/2.0, 0.00001)
                        self.QNet.model.compile(optimizer=Adam(lr=self.QNet.learning_rate),loss='mean_squared_error')
        
        
        
                if episode % 1==0:
                	perform_cross_time.append(kk)
                	
                	print(perform_cross_time)

                print("     Average reward:",sum(average_reward_episode) / float(len(average_reward_episode)),
                        "Average step:",sum(average_step) / float(len(average_step)),"epi:",episode,"itera:",itera,epsilon)
                
                
                
                average_reward_episode=[]
                average_step=[]
        print(perform_cross_time)
            
            
            
            
            
            
        
    
    
    
                
    
    def test_in_train(self):
        episode2=0
        average_reward_episode2 = []
        while True:#episode

            episode2 +=1 
            initial_state2 = self.env.reset()
            
            average_step2 = []
            average_loss2 = []

            total_reward2 = 0
            total_reward_t = 0
            num_steps2 = 0
            discount2 = 1
            nextstate2 = initial_state2# initial state
            cur_state2 = nextstate2
            frames_4 = [rgb2gray(cur_state2)]
            cur_action2 = self.env.action_space.sample()
            M =4
            action_cont = 0
            

            while True:
                
                
                action_cont += 1
                nextstate2, reward2, is_terminal2, debug_info2 = self.env.step(cur_action2)#S'
                frames_4.append(rgb2gray(nextstate2))
                
                
                if is_terminal2:
        
                    total_reward2 += discount2 * reward2
                    num_steps2 += 1
                    break
    
                
                if len(frames_4) > 4:
                    
                    state_stack = np.asarray(frames_4[:-1])
                    
                    frames_4 = frames_4[1:]
                    state_stack = np.swapaxes(state_stack,0,2)
                    state_stack = np.swapaxes(state_stack,0,1)#S'
                    
                    q_values = self.QNet.model.predict(x = np.expand_dims(state_stack, axis=0))
                    q_values2 = self.QNet2.model.predict(x = np.expand_dims(state_stack, axis=0))
                    
                    
                    
                    if action_cont %1==0:
                        cur_action2 = self.greedy_policy(np.add(q_values[0], q_values2[0]))# A'
                        action_cont=0
                                        
                
                total_reward2 += discount2 * reward2
                num_steps2 += 1

            
            average_reward_episode2.append(total_reward2)
            average_step2.append(num_steps2)
        


            if episode2 >= 10:
                kk = sum(average_reward_episode2) / float(len(average_reward_episode2))
                print("Average reward:",sum(average_reward_episode2) / float(len(average_reward_episode2)),
                    "Average step:",sum(average_step2) / float(len(average_step2)),"epi:",episode2)
                average_reward_episode2=[]
                average_step2=[]
                break
        return kk
        
    
    
    def test(self, model_file=None, numb=100):
        self.QNet.load_model(model_file)
        self.QNet2.load_model(model_file[:-3]+'_prime.h5')
        print(model_file[:-3]+'_prime.h5')
        episode=0
        average_reward_episode = []
        episode2=0
        average_reward_episode2 = []
        while True:#episode

            episode2 +=1 
            initial_state2 = self.env.reset()
            action_cont = 0
            
            average_step2 = []
            average_loss2 = []

            total_reward2 = 0
            num_steps2 = 0
            discount2 = 1
            nextstate2 = initial_state2# initial state
            cur_state2 = nextstate2
            frames_4 = [rgb2gray(cur_state2)]
            cur_action2 = self.env.action_space.sample()
            
            self.env.render()
            

            while True:
                
                action_cont += 1
                nextstate2, reward2, is_terminal2, debug_info2 = self.env.step(cur_action2)#S'
                self.env.render()
                frames_4.append(rgb2gray(nextstate2))
                
                
                if reward2<0:
                    print(reward2)
                
                
                if is_terminal2:
        
                    total_reward2 += discount2 * reward2
                    num_steps2 += 1
                    break
    
                
                if len(frames_4) > 4:
                    
                    state_stack = np.asarray(frames_4[:-1])
                    frames_4 = frames_4[1:]
                    state_stack = np.swapaxes(state_stack,0,2)
                    state_stack = np.swapaxes(state_stack,0,1)#S'
                    
                    q_values = self.QNet.model.predict(x = np.expand_dims(state_stack, axis=0))
                    q_values2 = self.QNet2.model.predict(x = np.expand_dims(state_stack, axis=0))
                                    
                    
                    if action_cont%1 == 0:
                        
                        cur_action2 = self.epsilon_greedy_policy(np.add(q_values[0], q_values2[0]), epsilon = 0)
                        action_cont = 0
                    
                    
                
                total_reward2 += discount2 * reward2
                num_steps2 += 1
                

            print(total_reward2)
            average_reward_episode2.append(total_reward2)
            average_step2.append(num_steps2)
        


            if episode2 >= 20:
                kk = sum(average_reward_episode2) / float(len(average_reward_episode2))
                print("Average reward:",sum(average_reward_episode2) / float(len(average_reward_episode2)),
                    "Average step:",sum(average_step2) / float(len(average_step2)),"epi:",episode2)
                average_reward_episode2=[]
                average_step2=[]
                break
    
    
    
    
    
        
    
    def burn_in_memory(self):
        
        num_transition = 0
        print(self.replay_memory.burn_in)
        
        
        
        while num_transition < self.replay_memory.burn_in:
                
            initial_state = self.env.reset()    
            total_reward = 0
            num_steps = 0
            cur_state = initial_state
            frames_4 = [rgb2gray(cur_state)]
            cur_action = self.env.action_space.sample()
            M = 4
            state_buff = []
            action_cont = 0
            
    
    
            while True:
                
                action_cont += 1
                nextstate, reward, is_terminal, debug_info = self.env.step(cur_action)
                frames_4.append(rgb2gray(nextstate))
            
                if len(frames_4) > 4:
                    
                    state_stack = np.asarray(frames_4[:-1])
                    
                    frames_4 = frames_4[1:]
                    state_stack = np.swapaxes(state_stack,0,2)
                    state_stack = np.swapaxes(state_stack,0,1)
                    
                    state_buff.append(state_stack)
                    
                    
                    if len(state_buff)>1:
                        num_transition +=1
                        if num_transition%10000 ==0:
                            print("Burn-in:",num_transition)
                        self.replay_memory.append([state_buff[0],cur_action,reward,state_stack,is_terminal]) #state, action, reward, next state, terminal flag tuples.
                        state_buff = state_buff[1:]
                    
                    
                    if True:
                        cur_action = self.env.action_space.sample()
                        
                
                
                
                cur_state = nextstate

                if is_terminal or num_transition >= self.replay_memory.burn_in:
                    break
            
            
           
            
        
    def burn_in_memory2(self):
        
        model_file = 'save/SpaceInvaders-burnin.h5'
        self.QNet.load_model(model_file)
        self.QNet2.load_model(model_file[:-3]+'_prime.h5')
        print(model_file[:-3]+'_prime.h5')
        
        num_transition = 0
        print(self.replay_memory.burn_in)
    
    
    
        while num_transition < self.replay_memory.burn_in:
            
            initial_state = self.env.reset()    
            total_reward = 0
            num_steps = 0
            cur_state = initial_state
            frames_4 = [rgb2gray(cur_state)]
            cur_action = self.env.action_space.sample()
            M = 4
            state_buff = []
            action_cont = 0
        


            while True:
            
                action_cont += 1
                nextstate, reward, is_terminal, debug_info = self.env.step(cur_action)
                frames_4.append(rgb2gray(nextstate))
        
                if len(frames_4) > 4:
                
                    state_stack = np.asarray(frames_4[:-1])
                    frames_4 = frames_4[1:]
                    state_stack = np.swapaxes(state_stack,0,2)
                    state_stack = np.swapaxes(state_stack,0,1)
                    state_buff.append(state_stack)
                
                
                    if len(state_buff)>1:
                        num_transition +=1
                        if num_transition%10000 ==0:
                            print("Burn-in:",num_transition)
                        self.replay_memory.append([state_buff[0],cur_action,reward,state_stack,is_terminal]) #state, action, reward, next state, terminal flag tuples.
                        state_buff = state_buff[1:]
                
                
                    
                    
                    q_values = self.QNet.model.predict(x = np.expand_dims(state_stack, axis=0))
                    q_values2 = self.QNet2.model.predict(x = np.expand_dims(state_stack, axis=0))
                                    
                    
                    if action_cont%1 == 0:
                        cur_action2 = self.epsilon_greedy_policy(np.add(q_values[0], q_values2[0]), epsilon = 0)
                        action_cont = 0
                    
                  
                cur_state = nextstate

                if is_terminal or num_transition >= self.replay_memory.burn_in:
                    break
        
        
       
    



def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()





def run_random_policy(env):
    
    initial_state = env.reset()
    env.render()
    time.sleep(0.1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    frames_4 = [rgb2gray(initial_state)]
    cur_action = env.action_space.sample()
    M = 4
    
    
    while True:
        
        nextstate, reward, is_terminal, debug_info = env.step(cur_action)
        env.render()
        frames_4.append(rgb2gray(nextstate))
            
        print(len(frames_4))
        if True and len(frames_4)> M:
            cur_action = env.action_space.sample()#take new action for next step
            state_stack = np.asarray(frames_4[:-1])
            frames_4 = frames_4[1:]
            state_stack = np.swapaxes(state_stack,0,2)
            state_stack = np.swapaxes(state_stack,0,1)
            print(state_stack.shape)
        
        
        

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(0.1)
    
    print(total_reward, num_steps)
    return total_reward, num_steps



def rgb2gray(rgb):
    
    rgb = cv2.resize(rgb,(input_dim,input_dim))
    color_img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return np.int8(color_img)


def main(args):

    args = parse_arguments()
    environment_name = args.env
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)
    keras.backend.tensorflow_backend.set_session(sess)
    
    
    
    agent = DQN_Agent(environment_name,render=args.render)
    
    
    if args.train>0:
        agent.burn_in_memory()
        #agent.burn_in_memory2()
        agent.train()
    
    if args.model_file != None:
        agent.test(args.model_file)
    
    
    


if __name__ == '__main__':
    main(sys.argv)

