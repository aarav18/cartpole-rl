# dependencies
import gym

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# build env
env = gym.make('CartPole-v0')
states = env.observation_space.shape[0]
actions = env.action_space.n

# build model
def build_model(states, actions):
    model = Sequential()
    
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    
    return model

model = build_model(states, actions)

# build agent
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    
    dqn = DQNAgent(model=model,
                   memory=memory, 
                   policy=policy, 
                   nb_actions=actions, 
                   nb_steps_warmup=10, 
                   target_model_update=1e-2)
    return dqn

# load trained model
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.load_weights('cartpole_weights.h5f')

# visualized test
_ = dqn.test(env, nb_episodes=5, visualize=True)