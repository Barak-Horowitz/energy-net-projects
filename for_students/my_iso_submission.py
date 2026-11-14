"""Construct one fresh trainable ISO model."""

# IMPORTANT CONSTANTS:

# PRICE RANGE (0-0.20/unit) - price = action * 0.20
# DISPATCH_SCALE = 6.0 - dispatch = action * 6.0
# GENERATION COST = 0.0375/unit - paid on ALL dispatched energy, used or not
# EMERGENCY_ENERGY_COST = 0.30/unit - checked per half-hour: if our won retail demand
#   (PCS grid imports * our market share) exceeds our dispatch, the shortfall is bought at 0.30.
#   we still collect our retail price on it, but max price is 0.20 so every short unit loses money
# EXTERNAL_RESALE_PRICE = 0.12 - what we get reselling energy bought from the PCS (not something we set)
# MINIMUM_PCS_EXPORT_PRICE = 0.06 - PCS only exports if the best buyback offer is >= 0.06 (raw action >= 0.3)
# TIED_MARKET_SHARE = 0.5 - exact price ties split the transaction between both ISOs
# TRAINING BUDGET = 300 CPU-seconds, alternating: us (16 days/phase) -> competitor -> PCS



# Training environment consists of a standard gym training env,
# for this exercise here are the important values contained.
# T - number of steps/day (48 half-hour intervals)
# observation_space. one/day contains an array of T values, predicted household consumption for each half-hour
#   NOTE: the "prediction" is just yesterday's actual consumption (all zeros on the first training day)
# action_space. one/day contains an array of T * 3 values in [0, 1] for each half-hour of the day:
#   [0:T]   buy price  - what WE PAY the PCS for its exported energy (buyback)
#   [T:2T]  sell price - what WE CHARGE the PCS (retail)
#   [2T:3T] dispatch   - energy we generate in advance
# step() - simulates an entire day and returns (next_day_obs, total_daily_profit, terminated=True, False, info)
# info contains the following info we can use to set our reward function.
# all are DAILY TOTALS, per-half-hour breakdowns are not in info
# (each also exist with opponent_iso_ prefix for the opponent)
# controlled_iso_profit - how much net profit we made (= reward)
# controlled_iso_retail_revenue - money from selling energy to the PCS (retail_energy * our retail price)
# controlled_iso_export_margin - net money from buying PCS energy and reselling it (export_energy * (0.12 - our buy price))
# controlled_iso_dispatch_cost - cost for normal energy generation (dispatch_energy * 0.0375)
# controlled_iso_emergency_dispatch_cost - cost for emergency energy (emergency_energy * 0.30)
# controlled_iso_retail_energy - total energy we SOLD to the PCS (won imports, may be partly covered by emergency energy)
# controlled_iso_export_energy - total energy we bought from the PCS (all resold at 0.12)
# controlled_iso_dispatch_energy - total energy we generated in advance
# controlled_iso_emergency_energy - total emergency energy we needed (sum of per-half-hour shortfalls)
# Additional environmental info (per-half-hour arrays of length T)
# realized_consumption - actual HOUSEHOLD consumption (not grid imports - battery shifts energy, and not our won demand)
# predicted - predicted household consumption (the observation we saw)
# dispatch - our dispatch after scaling (* 6)
# prices - market retail price = min(ours, competitor's), in $
#   where prices < our retail price, the competitor undercut us
# prices_raw - market buy price = max(ours, competitor's), raw [0, 1]
#   where prices_raw > our buy action, the competitor outbid us
# DON'T USE: money_earned, mae, action_raw - they describe the combined market, not our ISO
# reset() - returns the forecast for the next day (step already advanced the day pointer);
#   never rewinds, only wraps to the first day when the data runs out

# create_model contract:
# seed - use for ALL randomness (e.g. seed=seed on SB3 constructors)
# custom learn() must call the callback - that's how the CPU budget stops training
# changing the reward = custom replay buffer that transforms the reward when storing it (staff-approved on the forum),
#   never modify the shared env itself (e.g. replacing training_environment.step) - that fails the assignment

import gymnasium as gym 
import numpy as np 
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from for_students.internal.market_support import (
    EXTERNAL_RESALE_PRICE,
    MINIMUM_PCS_EXPORT_PRICE,
    DEFAULT_EMERGENCY_COST,
    DEFAULT_GENERATION_COST
)

# custom replay buffer: the env still returns real profit, we only change
# the reward our agent stores for its own training
class MyRewardBuffer(ReplayBuffer):
    EMERGENCY_USAGE_PENALTY_MODIFIER = DEFAULT_EMERGENCY_COST * 5
    UNUSED_ENERGY_PENALTY_MODIFIER = DEFAULT_GENERATION_COST * 3
    def add(self, obs, next_obs, action, reward, done, infos):
        # SB3 passes arrays with one entry per env, infos is a list of info dicts
        emergency = np.array([info['controlled_iso_emergency_energy'] for info in infos])
        dispatch = np.array([info['controlled_iso_dispatch_energy'] for info in infos])
        retail = np.array([info['controlled_iso_retail_energy'] for info in infos])
        unused_generated_energy = dispatch - (retail - emergency)
        training_reward = reward - (emergency * self.EMERGENCY_USAGE_PENALTY_MODIFIER) - (unused_generated_energy * self.UNUSED_ENERGY_PENALTY_MODIFIER) 
        super().add(obs, next_obs, action, training_reward, done, infos)
    
# custom wrapper around the env to allow for us to override actions
# by remapping them into "reasonable" intervals (as specified in decode)
class BoundedActions(gym.Wrapper):
    # remember the observation, decode needs the forecast
    def reset(self, **kwargs):
        self.last_obs, info = self.env.reset(**kwargs)
        return self.last_obs, info

    def step(self, action):
        env_action = decode(action, self.last_obs, self.env.generation_cost, self.env.dispatch_scale)
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        self.last_obs = obs
        return obs, reward, terminated, truncated, info


# SAC that trains on the bounded env and returns bounded actions to everyone else
class BoundedSAC(SAC):
    # trainer passes the RAW env before every phase -> wrap it every time
    def set_env(self, env, force_reset=True):
        return super().set_env(BoundedActions(env), force_reset=force_reset)

    # flag so predict() knows it's being called from inside training
    def learn(self, *args, **kwargs):
        self._in_learn = True
        try:
            return super().learn(*args, **kwargs)
        finally:
            self._in_learn = False
            
    # bound predictions before outputting
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action, state = super().predict(observation, state, episode_start, deterministic)
        if getattr(self, "_in_learn", False):
            return action, state
        return decode(action, observation, self.generation_cost, self.max_dispatch), state 
    

# agent outputs values for sale price, purchase price, energy generation 
# in [0, 1] space. A ton of values in this space make no sense at all
#(i.e. setting sale price lower then generation cost)
# so we bound the agent's outputs to the following ranges
# (each block is remapped as low + a * (high - low), so every agent value lands somewhere in the range)
# buyback: within [0.06, 0.12] - we sell for 0.12 buying for more makes no sense,
# retail: within [generation_cost, 0.20] - selling for less then generation cost makes no sense
# dispatch: within [forecasted, max_dispatch] - making less energy then forecasted for usage is obviously wrong choice as it leads to emergency costs
# bounds are computed in $ / energy units, then converted back to the env's raw [0, 1] scale
PRICE_MAX = 0.20 # price action 1.0 = 0.20/unit

def decode(action, observation, generation_cost, max_dispatch):
    action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), 0.0, 1.0)
    T = len(action) // 3
    forecast = np.asarray(observation, dtype=np.float32).reshape(-1)[:T]

    # rescale buyback into [0.06, 0.12]
    buyback = MINIMUM_PCS_EXPORT_PRICE + action[:T] * (EXTERNAL_RESALE_PRICE - MINIMUM_PCS_EXPORT_PRICE)
    # rescale retail into [generation_cost, 0.20]
    retail = generation_cost + action[T:2 * T] * (PRICE_MAX - generation_cost)
    # rescale dispatch into [forecast, max_dispatch] (clipped if the forecast is ever above the env's cap)
    low = np.minimum(forecast, max_dispatch)
    dispatch = low + action[2 * T:] * (max_dispatch - low)
    return np.concatenate([
        buyback / PRICE_MAX,
        retail / PRICE_MAX,
        dispatch / max_dispatch,
    ]).astype(np.float32)
    
def create_model(training_environment, seed):
    """Return a new model implementing set_env(), learn(), and predict()."""
    # SAC = soft actor critic library, standard RL library (TODO: watch tutorials to figure out how it works)
    model = BoundedSAC(
        "MlpPolicy", # network to use (options are mlpPolicy, CnnPolicy, MultiInputPolicy)
        BoundedActions(training_environment), # environment SAC trains off of
        replay_buffer_class=MyRewardBuffer, # stores our shaped reward instead of raw profit
        seed=seed, # random seed passed in to use for random events
        verbose=0, # log level 
        learning_rate=0.003, # size of weight changes applied to newtwork 
        buffer_size=100_000, # how many previous timesteps are held 
        learning_starts=32, # number of steps taken to establish baseline before training
        batch_size=64, # number of steps taken into account for each weight update
        train_freq=(1, "step"), # how often to train, what to call
        gradient_steps=1, # how many weight updates/step
        gamma=1, #discount factor (how much an immediate reward is worth compared to long term)
        tau=0.005, # how slowly target critics follow main critics? TODO: figure out what this means
        ent_coef="auto", # weight of exploration vs. exploitation
    )
    model.generation_cost = training_environment.generation_cost
    model.max_dispatch = training_environment.dispatch_scale
    return model
