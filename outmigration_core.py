"""OHQ-only hospital-choice ABM core.

Direct all-hospital choice depends only on objective hospital quality (OHQ)
and person-to-hospital distance. Realized cancer choices are outputs only.
"""
from pathlib import Path
import random
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from easydict import EasyDict
from pyproj import Proj, transform
from scipy.stats import gamma, invgamma
from shapely.geometry import Point, Polygon
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
proj_tm_mid, proj_wgs84 = Proj(init="epsg:2097"), Proj(init="epsg:4326")


def random_points_in_multipolygon(shape, num_points):
    polygons = [shape] if isinstance(shape, Polygon) else list(shape.geoms)
    min_x, min_y = min(p.bounds[0] for p in polygons), min(p.bounds[1] for p in polygons)
    max_x, max_y = max(p.bounds[2] for p in polygons), max(p.bounds[3] for p in polygons)
    points = []
    while len(points) < num_points:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if any(p.contains(point) for p in polygons):
            points.append(point)
    return points


def load_geographic_context(data_dir=".", geography_dir="."):
    global gdf, outside_polygon_mask
    gdf = gpd.read_file(Path(geography_dir) / "Korea_shapefiles/TL_SCCO_CTPRVN.shp", encoding="euc-kr")
    # 2026-11-11: preserve the legacy explicit province-name decoding used by the city input workbook.
    gdf["CTP_KOR_NM"] = [
        "서울", "경기", "경남", "경북", "광주", "대구", "대전", "부산", "세종",
        "울산", "인천", "제주", "전남", "전북", "충남", "충북", "강원",
    ]
    outside_polygon_mask = np.load(Path(data_dir) / "outside_polygon_mask.npy")


def load_simulation_inputs(data_dir="."):
    data_dir = Path(data_dir)
    return (pd.read_excel(data_dir / "ABM_city_info.xlsx", index_col=0),
            pd.read_excel(data_dir / "ABM_hospital_crd.xlsx", index_col=0))


def make_default_parameter_set(df_city_info, df_hospital_crd):
    """Parameters retained by the independent OHQ-only experiment branch."""
    return EasyDict({
        "max_dt": 100, "n_p": 28, "cancer_duration": 10,
        "df_city_info": df_city_info, "df_hospital_crd": df_hospital_crd,
        "target_population_column": "POPULATION_1",
        "dict_crd_region": {"SM": Point(126.9707, 37.5551), "YN": Point(129.0415, 35.1151),
                            "HN": Point(126.8515, 35.1603), "CC": Point(127.4347, 36.3324),
                            "GW": Point(128.8961, 37.7644)},
        "max_quality": 5.0, "min_quality": 0.0,
        # 2026-11-11: beta_OHQ starts at 10.62; OHQ-scale recalibration is still required.
        "beta_OHQ": 10.62, "beta_d": 20.30, "delta": 1.85,
        "vor_K": 5, "gamma_alpha": 2.0, "gamma_beta": 2.0,
        "Q_vmax": 2.0, "S_half": 1 / 47,
        "action_effect_K": 5, "action_inv_gamma_alpha": 3.0,
        "action_inv_gamma_beta": 8.0, "Q_amax": 2.0,
    })


class Simulation:
    """OHQ-only simulation with VOR and delayed action-quality effects."""
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set
        self.df_city_info, self.df_hospital_crd = parameter_set.df_city_info, parameter_set.df_hospital_crd
        self.num_population = int(self.df_city_info[parameter_set.target_population_column].sum())
        self.num_hospital = int(self.df_city_info.NUM_HOSPITAL.sum())
        self.min_quality, self.max_quality = float(parameter_set.min_quality), float(parameter_set.max_quality)
        self.arr_population_choice_prob = np.zeros((self.num_population, self.num_hospital))
        self.arr_population_CancerOrNot = np.zeros(self.num_population, dtype=bool)
        self.arr_population_is_new_cancerpatient = np.zeros(self.num_population, dtype=bool)
        self.arr_active_cancer_end_dt = np.full(self.num_population, -1, dtype=int)
        self.arr_hospital_volume = np.zeros((parameter_set.max_dt, self.num_hospital), dtype=int)
        self.arr_hospital_action = np.zeros((parameter_set.max_dt, self.num_hospital), dtype=float)
        self.base_OHQ = np.ones(self.num_hospital, dtype=float)
        self.vor_OHQ = np.zeros(self.num_hospital, dtype=float)
        self.action_OHQ = np.zeros(self.num_hospital, dtype=float)
        self.idx_new_patient = np.empty(0, dtype=int)
        lags = np.arange(1, int(parameter_set.vor_K) + 1, dtype=float)
        self.vor_weights = gamma.pdf(lags, a=parameter_set.gamma_alpha, scale=parameter_set.gamma_beta)
        action_lags = np.arange(1, int(parameter_set.action_effect_K) + 1, dtype=float)
        self.action_effect_weights = invgamma.pdf(action_lags, a=parameter_set.action_inv_gamma_alpha,
                                                  scale=parameter_set.action_inv_gamma_beta)
        if self.vor_weights.sum() <= 0 or self.action_effect_weights.sum() <= 0:
            raise ValueError("VOR and action lag weights must have positive sums.")
        self.action_effect_weights /= self.action_effect_weights.sum()

    def initialize(self):
        self.create_Cities()
        self.create_Hospitals()
        self.create_People()
        self.calculate_distance_effect()

    def execute(self):
        self.prepare_episode()
        self.run_simulation()

    def prepare_episode(self):
        self.set_OHQ()
        self.calculate_choice_prob_all()
        self.create_Archive()

    def set_OHQ(self):
        """Initialize fixed base OHQ and the OHQ-only virtual VOR history."""
        # 2026-11-11: base OHQ -> virtual history -> initial VOR -> total OHQ is the OHQ-only initialization order.
        self.base_OHQ.fill(1.0)
        self.vor_OHQ.fill(0.0)
        self.action_OHQ.fill(0.0)
        self.arr_hospital_action.fill(0.0)
        self.arr_hospital_volume.fill(0)
        self.calculate_virtual_choice_prob()
        self.VOR_effect(self.parameter_set.Q_vmax, dt=0)
        self.update_total_OHQ()

    def create_Cities(self):
        self.arr_City = []
        for idx_city, row in self.df_city_info.reset_index(drop=True).iterrows():
            self.arr_City.append(City(idx_city, row.SIDO_NM, row.REGION,
                                      int(row[self.parameter_set.target_population_column]), int(row.NUM_HOSPITAL)))

    def create_Hospitals(self):
        records = []
        for city in self.arr_City:
            city_hospitals = self.df_hospital_crd.loc[self.df_hospital_crd.SIDO_NM == city.nm].iloc[:city.num_hospital]
            for _, row in city_hospitals.iterrows():
                records.append({"idx_hospital": len(records), "nm_hospital": row.INST_NM,
                                "nm_city": city.nm, "idx_city": city.idx, "REGION": city.nm_region,
                                "coord": Point(transform(proj_tm_mid, proj_wgs84, row.CRD_X, row.CRD_Y))})
        self.df_Hospital_info = pd.DataFrame(records)
        if len(self.df_Hospital_info) != self.num_hospital:
            raise ValueError("Hospital input does not match NUM_HOSPITAL totals.")
        self.df_Hospital_info["OQ_objective_quality"] = self.min_quality
        self.arr_Hospital = np.array([Hospital(r) for r in records], dtype=object)

    def create_People(self):
        if "gdf" not in globals():
            raise RuntimeError("Call load_geographic_context() before Simulation.initialize().")
        records, persons = [], []
        for city in self.arr_City:
            city_polygon = gdf.loc[gdf.CTP_KOR_NM == city.nm]
            if city_polygon.empty:
                raise ValueError(f"No geographic polygon found for city {city.nm!r}.")
            for coord in random_points_in_multipolygon(city_polygon.iloc[0].geometry, city.num_population):
                idx = len(records)
                records.append({"idx_person": idx, "nm_city": city.nm, "idx_city": city.idx,
                                "REGION": city.nm_region, "coord": coord, "is_cancerpatient": False,
                                "idx_chosen_hospital": -1})
                persons.append(Person(idx, city.idx, city.nm, coord, self.num_hospital))
        self.df_Person_info, self.arr_Person = pd.DataFrame(records), np.array(persons, dtype=object)

    def calculate_distance_effect(self):
        person_xy = np.array([(p.x, p.y) for p in self.df_Person_info.coord])
        hospital_xy = np.array([(p.x, p.y) for p in self.df_Hospital_info.coord])
        distance = np.sqrt(((person_xy[:, None, :] - hospital_xy[None, :, :]) ** 2).sum(axis=2))
        self.arr_distance_effect_hospital = np.exp(-self.parameter_set.delta * distance)

    @staticmethod
    def _softmax(utility):
        shifted = utility - utility.max(axis=1, keepdims=True)
        exp_utility = np.exp(shifted)
        return exp_utility / exp_utility.sum(axis=1, keepdims=True)

    def calculate_virtual_choice_prob(self):
        """OHQ-only virtual probabilities fill missing VOR lags before observed history exists."""
        utility = (self.base_OHQ[None, :] * self.parameter_set.beta_OHQ +
                   self.arr_distance_effect_hospital * self.parameter_set.beta_d)
        self.virtual_choice_prob = self._softmax(utility).mean(axis=0)
        self.prehistory_choice_prob = np.repeat(self.virtual_choice_prob[None, :], int(self.parameter_set.vor_K), axis=0)

    def calculate_choice_prob_all(self):
        """Compute direct all-hospital choice probabilities from current OHQ and distance effects."""
        ohq = self.df_Hospital_info.OQ_objective_quality.to_numpy(dtype=float)
        utility = ohq[None, :] * self.parameter_set.beta_OHQ + self.arr_distance_effect_hospital * self.parameter_set.beta_d
        self.arr_population_choice_prob[:] = self._softmax(utility)
        for person in self.arr_Person:
            person.choice_prob = self.arr_population_choice_prob[person.idx_person].copy()

    def create_Archive(self):
        t, p, h = int(self.parameter_set.max_dt), self.num_population, self.num_hospital
        self.achv_population_choice_prob = np.zeros((t, p, h))
        self.achv_mean_hospital_choice_prob = np.zeros((t, h))
        self.achv_OHQ = np.zeros((t, h))
        self.achv_base_OHQ = np.zeros((t, h))
        self.achv_vor_OHQ = np.zeros((t, h))
        self.achv_action_OHQ = np.zeros((t, h))
        self.achv_population_is_cancerpatient = np.zeros((t, p), dtype=bool)
        self.update_Archive(0)

    def update_Archive(self, dt):
        self.achv_population_choice_prob[dt] = self.arr_population_choice_prob
        self.achv_mean_hospital_choice_prob[dt] = self.arr_population_choice_prob.mean(axis=0)
        self.achv_OHQ[dt] = self.df_Hospital_info.OQ_objective_quality.to_numpy()
        self.achv_base_OHQ[dt], self.achv_vor_OHQ[dt], self.achv_action_OHQ[dt] = self.base_OHQ, self.vor_OHQ, self.action_OHQ
        self.achv_population_is_cancerpatient[dt] = self.arr_population_CancerOrNot

    def VOR_effect(self, q_vmax, dt):
        observed = self.achv_mean_hospital_choice_prob[:dt][::-1][:int(self.parameter_set.vor_K)] if dt else np.empty((0, self.num_hospital))
        missing = max(0, int(self.parameter_set.vor_K) - len(observed))
        history = np.vstack([observed, self.prehistory_choice_prob[::-1][:missing]])
        weights = self.vor_weights[:len(history)].copy()
        weights /= weights.sum()
        S_h = (history * weights[:, None]).sum(axis=0)
        self.vor_OHQ = q_vmax * S_h / (self.parameter_set.S_half + S_h)

    def investment_effect(self, hospital_action, dt):
        action = np.asarray(hospital_action, dtype=float)
        if action.shape != (self.num_hospital,) or not np.isfinite(action).all():
            raise ValueError(f"hospital_action must be a finite array with shape ({self.num_hospital},).")
        self.arr_hospital_action[dt] = action
        history_length = min(dt + 1, int(self.parameter_set.action_effect_K))
        history = self.arr_hospital_action[dt - history_length + 1:dt + 1][::-1]
        self.action_OHQ = self.parameter_set.Q_amax * (history * self.action_effect_weights[:history_length, None]).sum(axis=0)

    def update_total_OHQ(self):
        self.df_Hospital_info["OQ_objective_quality"] = np.clip(self.base_OHQ + self.vor_OHQ + self.action_OHQ,
                                                                   self.min_quality, self.max_quality)

    def expire_active_cancerpatients(self, dt):
        expired = np.flatnonzero((self.arr_active_cancer_end_dt >= 0) & (self.arr_active_cancer_end_dt <= dt))
        self.arr_population_CancerOrNot[expired] = False
        self.arr_population_is_new_cancerpatient[expired] = False
        self.arr_active_cancer_end_dt[expired] = -1
        self.df_Person_info.loc[expired, "is_cancerpatient"] = False
        for idx in expired:
            self.arr_Person[idx].is_cancerpatient = False
            self.arr_Person[idx].is_new_cancerpatient = False

    def designate_new_cancerpatient(self, dt):
        candidates = np.flatnonzero(~self.arr_population_CancerOrNot)
        if len(candidates) < self.parameter_set.n_p:
            raise ValueError(f"Not enough non-active cancer candidates at dt={dt}: need {self.parameter_set.n_p}, found {len(candidates)}.")
        self.idx_new_patient = np.random.choice(candidates, size=self.parameter_set.n_p, replace=False)
        self.arr_population_is_new_cancerpatient.fill(False)
        self.arr_population_CancerOrNot[self.idx_new_patient] = True
        self.arr_population_is_new_cancerpatient[self.idx_new_patient] = True
        self.arr_active_cancer_end_dt[self.idx_new_patient] = dt + int(self.parameter_set.cancer_duration)
        self.df_Person_info.loc[self.idx_new_patient, "is_cancerpatient"] = True
        choices = np.array([np.random.choice(self.num_hospital, p=self.arr_population_choice_prob[idx]) for idx in self.idx_new_patient])
        self.df_Person_info.loc[self.idx_new_patient, "idx_chosen_hospital"] = choices
        for idx, choice in zip(self.idx_new_patient, choices):
            person = self.arr_Person[idx]
            person.is_cancerpatient, person.is_new_cancerpatient = True, True
            person.choice_prob, person.idx_choice_hospital = self.arr_population_choice_prob[idx].copy(), int(choice)
        # 2026-11-11: realized cancer volume is an output only; VOR uses population mean choice probability.
        self.arr_hospital_volume[dt] = np.bincount(choices, minlength=self.num_hospital)

    def advance_one_timestep(self, dt, hospital_action=None):
        if hospital_action is None:
            hospital_action = np.zeros(self.num_hospital, dtype=float)
        self.expire_active_cancerpatients(dt)
        self.calculate_choice_prob_all()
        self.designate_new_cancerpatient(dt)
        self.VOR_effect(self.parameter_set.Q_vmax, dt)
        self.investment_effect(hospital_action, dt)
        self.update_total_OHQ()
        self.update_Archive(dt)
        # 2026-11-11: action at t updates OHQ after t's choice; its choice/VOR/outmigration effect begins at t+1.

    def run_simulation(self):
        for dt in tqdm(range(self.parameter_set.max_dt)):
            self.advance_one_timestep(dt)


class City:
    def __init__(self, idx, nm, nm_region, num_population, num_hospital):
        self.idx, self.nm, self.nm_region = idx, nm, nm_region
        self.num_population, self.num_hospital = num_population, num_hospital


class Hospital:
    def __init__(self, record):
        self.idx, self.nm_hospital = record["idx_hospital"], record["nm_hospital"]
        self.idx_city, self.nm_city, self.coord = record["idx_city"], record["nm_city"], record["coord"]


class Person:
    """Minimal person state retained for cancer lifecycle and realized choice records."""
    def __init__(self, idx_person, idx_city, nm_city, coord, num_hospital):
        self.idx_person, self.idx_city, self.nm_city, self.coord = idx_person, idx_city, nm_city, coord
        self.is_cancerpatient = self.is_new_cancerpatient = False
        self.choice_prob = np.zeros(num_hospital)
        self.idx_choice_hospital = -1
