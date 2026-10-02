"""Gymnasium preparation layer for the simultaneous hospital MARL game."""

import random
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import outmigration_core as core


class ABMenv_gym(gym.Env):
    """One Gym step accepts all hospital actions and advances one ABM timestep.

    This is intentionally a Gymnasium joint-action environment. A future MARL
    runner can split the joint action and individual-reward vector by hospital
    without changing the ABM transition code.
    """

    metadata = {"render_modes": []}

    def __init__(self, data_dir, geography_dir, max_dt=None, investment_cost=0.0):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.geography_dir = Path(geography_dir)
        self._configured_max_dt = max_dt
        self.investment_cost = float(investment_cost)
        self.sim = None
        self.current_step = 0
        # 이번 step의 병원별 보상과 episode 누적 병원별 보상
        self.individual_rewards = None
        self.cumulative_individual_rewards = None

        df_city_info, _ = core.load_simulation_inputs(self.data_dir)
        self.num_hospitals = int(df_city_info["NUM_HOSPITAL"].sum())
        self.individual_rewards = np.zeros(self.num_hospitals, dtype=np.float32)
        self.cumulative_individual_rewards = np.zeros(self.num_hospitals, dtype=np.float32)

        # 우선 binary action (0: 투자 x, 1: 투자)
        self.action_space = spaces.MultiBinary(self.num_hospitals)

        self.observation_space = spaces.Dict({
            "ohq": spaces.Box(low=0.0, high=5.0, shape=(self.num_hospitals,), dtype=np.float32,),
            "demand": spaces.Box(low=0.0, high=np.inf, shape=(self.num_hospitals,), dtype=np.float32,)
        })
        
        self.last_actions = np.zeros(self.num_hospitals, dtype=np.int8) # 직전 action
        self.last_new_patient_counts = np.zeros(self.num_hospitals, dtype=np.int64) # 직전 수요

    def reset(self, seed=None, options=None):
        """Create a fresh episode and return the pre-transition market state."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        core.load_geographic_context(data_dir=self.data_dir, geography_dir=self.geography_dir)
        df_city_info, df_hospital_crd = core.load_simulation_inputs(self.data_dir)
        parameter_set = core.make_default_parameter_set(df_city_info, df_hospital_crd)
        if self._configured_max_dt is not None:
            parameter_set.max_dt = int(self._configured_max_dt)

        self.sim = core.Simulation(parameter_set)
        self.sim.initialize()
        self.sim.prepare_episode() # 추후 initalize와 prepare_episode 합쳐도될듯
        self.current_step = 0
        self.last_actions.fill(0)
        self.last_new_patient_counts.fill(0)
        self.individual_rewards.fill(0.0)
        self.cumulative_individual_rewards.fill(0.0)

        return self._get_observation(), self._get_info()

    def step(self, joint_action):
        joint_action = np.asarray(joint_action, dtype=np.int8)
        self.sim.advance_one_timestep(self.current_step)
        self.last_actions = joint_action.copy()
        self.last_new_patient_counts = self._new_patient_counts()

        # lines 82-84는 임의의 reward (추후 수정 필요, reward stepwise화) 
        self.individual_rewards = (
            self.last_new_patient_counts.astype(np.float32)
            - self.investment_cost * self.last_actions.astype(np.float32))
        
        self.cumulative_individual_rewards += self.individual_rewards
        self.current_step += 1

        obs = self._get_observation()
        info = self._get_info()

        terminated = self.current_step >= self.sim.parameter_set.max_dt

        return obs, 0.0, terminated, False, info

    def state(self): # 추후 병원이 decision시 고려할 정보가 확정되면 맞춰 수정
        return np.column_stack((
            self.sim.df_Hospital_info["OQ_objective_quality"].to_numpy(dtype=np.float32),
            np.mean(self.sim.arr_population_choice_prob, axis=0, dtype=np.float32),
            self.last_actions.astype(np.float32),
            self.last_new_patient_counts.astype(np.float32)))

    def _get_observation(self):
        return {
            "ohq": self.sim.df_Hospital_info["OQ_objective_quality"].to_numpy(dtype=np.float32, copy=True),
            "demand": self.last_new_patient_counts.astype(np.float32, copy=True)
        }

    # 병원별 인원 count
    def _new_patient_counts(self):
        chosen_hospitals = self.sim.df_Person_info.loc[self.sim.idx_new_patient, "idx_chosen_hospital"].to_numpy(dtype=np.int64)
        return np.bincount(chosen_hospitals, minlength=self.num_hospitals)

    def _get_info(self):
        return {
            "timestep": self.current_step,
            "last_actions": self.last_actions.copy(),
            "new_patient_counts": self.last_new_patient_counts.copy(),
            "individual_rewards": self.individual_rewards.copy(),
            "cumulative_individual_rewards": self.cumulative_individual_rewards.copy()}
