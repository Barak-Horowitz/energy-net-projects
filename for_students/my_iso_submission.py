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
# changing the reward = wrapping the env inside our model's set_env - check with staff that this is allowed

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from energy_net.gym_envs.alternating_env import get_pred_window
from for_students.internal.market_support import (
    EXTERNAL_RESALE_PRICE,
    MINIMUM_PCS_EXPORT_PRICE,
)

PRICE_MAX = 0.20 # action 1.0 = 0.20/unit
MIN_PRICE_STEP = 0.0001 # "minimum amount" we undercut/outbid the opponent by


# custom wrapper around the training env to override the reward function.
# EXPERIMENT: reward = profit - (max_profit - profit), where max_profit is the hindsight best response:
#   retail  - price MIN_PRICE_STEP below the opponent, dispatch exactly the PCS imports (no emergency energy)
#   buyback - price MIN_PRICE_STEP above the opponent (at least 0.06 so the PCS exports)
#   skip any half-hour where winning would lose money
# assumes the PCS would have imported/exported the same amounts under our "best" prices (ignores battery reaction)
# NOTE: reads env internals (opponent model, pcs_env.day_steps) - only affects training reward, not evaluation
class MaxProfitRegretReward(gym.Wrapper):
    def step(self, action):
        market = self.env
        T = market.T

        # the env asks the opponent for its action with this same observation and deterministic=True,
        # so this reproduces exactly the offer the opponent makes this day
        opponent_observation = get_pred_window(market, market._next_start_idx, T)
        opponent_action, _ = market.opponent_iso_model.predict(opponent_observation, deterministic=True)
        opponent_action = np.asarray(opponent_action, dtype=np.float32).reshape(-1)
        opponent_buyback = opponent_action[:T] * PRICE_MAX
        opponent_retail = opponent_action[T:2 * T] * PRICE_MAX

        obs, profit, terminated, truncated, info = market.step(action)

        # per-half-hour PCS grid flow for the day just simulated (positive = imports, negative = exports)
        market_energy = np.array([step["market_energy_units"] for step in market.pcs_env.day_steps])
        imports = np.maximum(market_energy, 0.0)
        exports = np.maximum(-market_energy, 0.0)

        best_retail = opponent_retail - MIN_PRICE_STEP
        retail_profit = np.maximum(imports * (best_retail - market.generation_cost), 0.0)

        best_buyback = np.maximum(opponent_buyback + MIN_PRICE_STEP, MINIMUM_PCS_EXPORT_PRICE)
        export_profit = np.maximum(exports * (EXTERNAL_RESALE_PRICE - best_buyback), 0.0)

        max_profit = float(np.sum(retail_profit) + np.sum(export_profit))
        info["max_profit"] = max_profit

        new_reward = profit - (max_profit - profit)
        return obs, new_reward, terminated, truncated, info


# SAC that wraps every env the trainer hands it (the trainer calls set_env before every training phase)
class RegretSAC(SAC):
    def set_env(self, env, force_reset=True):
        super().set_env(MaxProfitRegretReward(env), force_reset=force_reset)


def create_model(training_environment, seed):
    """Return a new model implementing set_env(), learn(), and predict()."""
    # SAC = soft actor critic library, standard RL library (TODO: watch tutorials to figure out how it works)
    return RegretSAC(
        "MlpPolicy", # network to use (options are mlpPolicy, CnnPolicy, MultiInputPolicy)
        MaxProfitRegretReward(training_environment), # environment SAC trains off of
        seed=seed, # random seed passed in to use for random events
        verbose=0, # log level 
        learning_rate=0.0003, # size of weight changes applied to newtwork 
        buffer_size=100_000, # how many previous timesteps are held 
        learning_starts=32, # number of steps taken to establish baseline before training
        batch_size=64, # number of steps taken into account for each weight update
        train_freq=(1, "step"), # how often to train, what to call
        gradient_steps=1, # how many weight updates/step
        gamma=0.99, #discount factor (how much an immediate reward is worth compared to long term)
        tau=0.005, # how slowly target critics follow main critics? TODO: figure out what this means
        ent_coef="auto", # weight of exploration vs. exploitation
    )
