"""Import-safe core for the hospital-choice ABM.

This module preserves the original model transition logic while exposing
episode setup and one-timestep advancement for an external environment.
"""

#!/usr/bin/env python
# coding: utf-8

# Copyright 2025. Jungwoo Kim. All rights reserved.<br>
# Complex System Design Laboratory<br>
# Dept. of Industrial & Systems Engineering, KAIST

# # Import Libraries

import numpy as np
import time  # 2026-09-04: run_simulation 구간별 실행시간 profiling을 위한 타이머 모듈
from tqdm import tqdm
import random
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import easydict
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix
from scipy.interpolate import griddata
from scipy.stats import gamma, invgamma
from shapely.geometry import Point, MultiPolygon, Polygon
import geopandas as gpd
import openpyxl
from easydict import EasyDict
from tqdm import tqdm
from sklearn.neighbors import KDTree
from descartes import PolygonPatch
import matplotlib.colors as mcolors
from pyproj import Proj, transform
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

proj_tm_mid = Proj(init='epsg:2097')
proj_wgs84 = Proj(init='epsg:4326')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
random.seed(4242)


# # Tool Functions



def random_points_in_multipolygon(shape, num_points):
    if isinstance(shape, Polygon):
        min_x, min_y, max_x, max_y = shape.bounds
    else:
        min_x, min_y, max_x, max_y = shape.geoms[0].bounds
        for i in range(1,len(shape.geoms)):
            polygon = shape.geoms[i]
            min_x_i, min_y_i, max_x_i, max_y_i = polygon.bounds
            if min_x_i<min_x:
                min_x = min_x_i
            if min_y_i<min_y:
                min_y = min_y_i
            if max_x_i>max_x:
                max_x = max_x_i
            if max_y_i>max_y:
                max_y = max_y_i
    # 좌표 설정
    points = []
    while len(points) < num_points:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if isinstance(shape, Polygon):
            if shape.contains(point):
                    points.append(point)
        else:
            for i in range(len(shape.geoms)):
                polygon = shape.geoms[i]
                if polygon.contains(point):
                    points.append(point)
            
    return points

# data들은 하나의 directory에 보관

def load_geographic_context(data_dir=".", geography_dir="."):
    global gdf, outside_polygon_mask
    geo_file = str(Path(geography_dir) / 'Korea_shapefiles/TL_SCCO_CTPRVN.shp')
    gdf = gpd.read_file(geo_file, encoding='euc-kr')
    outside_polygon_mask = np.load(Path(data_dir) / "outside_polygon_mask.npy")
    
    gdf.CTP_KOR_NM = [
        '강원',
        '경기',
        '경남',
        '경북',
        '광주',
        '대구',
        '대전',
        '부산',
        '서울',
        '세종',
        '울산',
        '인천',
        '전남',
        '전북',
        '제주',
        '충남',
        '충북'
    ]
    
    # Manual modifications
    
    coords1 = [(126.59156455632241, 37.593011823985485), (126.6512526639701, 37.63798791865732),
               (126.79369544519706, 37.58159385885768), (126.7662043574505, 37.554242375748366),
               (126.7544652746883, 37.41757293948264), (126.66344966004381, 37.35053506629968),
               (126.60996708541704, 37.38718717658724), (126.59156455632241, 37.593011823985485)]
    coords2 = [(126.43121685572747, 37.82986908194544), (126.35057985274231, 37.789565062650894),
               (126.40316813158975, 37.59428252797514), (126.51062866491168, 37.59662373953827),
               (126.5068975008081, 37.78234762310096), (126.43121685572747, 37.82986908194544)]
    polygon1 = Polygon(coords1)
    polygon2 = Polygon(coords2)
    gdf.loc[11,'geometry'] = MultiPolygon([polygon1, polygon2])
    coords1 = [(126.16122307201523, 36.84077447960923), (126.1346055789919, 36.740324711760486),
               (126.17654567314271, 36.71395826078249), (126.23492459115923, 36.71807007755431),
               (126.30206037378963, 36.62590094644257), (126.45646406403513, 36.595210697608685),
               (126.50292188756616, 36.43400816610959), (126.48058601093365, 36.38503110228303),
               (126.54732360572413, 36.2683451886003), (126.52610700099113, 36.167672899104176),
               (126.67649645277267, 36.00923289097959), (126.74156052854424, 36.0119448809583),
               (126.87081796412734, 36.06740464723523), (126.88282411119397, 36.13207402130215),
               (127.04003005195446, 36.13933847885477), (127.12318778724037, 36.06424768168607),
               (127.34009324309051, 36.128979280260644), (127.37649530932696, 36.02288908915842),
               (127.45674606514737, 35.983276216973415), (127.63829543249358, 36.06793987054469),
               (127.63791587136105, 36.06890166403231), (127.59802031438574, 36.21708890108369),
               (127.49257720191125, 36.23795395068193), (127.44871641082584, 36.19672289194393),
               (127.32394583489626, 36.203158372814976), (127.25877186484053, 36.2760505815688),
               (127.28212428234082, 36.414603964631645), (127.20138964906599, 36.44197527604719),
               (127.19378772255222, 36.564812513911036), (127.13437853561463, 36.7067919896922),
               (127.20793603234716, 36.71918904581354), (127.20792754283481, 36.71917583461923),
               (127.20792684751393, 36.71917607776152), (127.20787710929177, 36.71918850565105),
               (127.2078769797854, 36.71919046333673), (127.20787710929177, 36.71918850565105),
               (127.20792754283481, 36.71917583461923), (127.20789089293503, 36.719119287386114),
               (127.22822551880574, 36.708468998779054), (127.22822008391248, 36.70845514952263),
               (127.22822551880574, 36.708468998779054), (127.22823057936279, 36.70846556314337),
               (127.28126367067202, 36.69057059579415), (127.28125389722467, 36.69051337150437),
               (127.28126367067202, 36.69057059579415), (127.28127028658162, 36.69055905464245),
               (127.28529006643944, 36.690671454249056), (127.39951636598008, 36.7992190104284),
               (127.28782886465115, 36.893778879374906), (127.14366362142856, 36.97105116272797),
               (126.90970519398604, 36.90160135376443), (126.8169656460083, 36.8961446522021),
               (126.77958940321619, 36.96680085646616), (126.49675135100698, 37.052811981614845),
               (126.37649095830157, 36.9806341203674), (126.42112642747156, 36.92727201106454),
               (126.32973391103106, 36.81358358482278), (126.30162178982232, 36.82841928228409),
               (126.31328915259776, 36.90549865703278), (126.16122307201523, 36.84077447960923)]
    gdf.loc[15,'geometry'] = Polygon(coords1)

# input: excel file
def load_simulation_inputs(data_dir="."): 
    data_dir = Path(data_dir)
    df_city_info = pd.read_excel(data_dir / "ABM_city_info.xlsx", index_col=0)
    df_hospital_crd = pd.read_excel(data_dir / "ABM_hospital_crd.xlsx", index_col=0)
    return df_city_info, df_hospital_crd

# ABM simulation parameter 입력 함수화
def make_default_parameter_set(df_city_info, df_hospital_crd):
    return EasyDict({
        'max_dt': 100,
        'n_p': 28,
        # 2026-09-28: 신규 cancer patient가 active cancer 상태를 유지하는 timestep 수
        'cancer_duration': 10,
        'df_city_info': df_city_info,
        'df_hospital_crd': df_hospital_crd,
        'target_population_column': 'POPULATION_1',
        'dict_crd_region': {
            'SM': Point(126.9707, 37.5551),
            'YN': Point(129.0415, 35.1151),
            'HN': Point(126.8515, 35.1603),
            'CC': Point(127.4347, 36.3324),
            'GW': Point(128.8961, 37.7644),
        },
        'max_quality': 5.0,
        'min_quality': 0.0,
        'dict_parameter_initial_OHQ': {
            # 2026-09-21: 기존 region-choice calibration에 맞춰 YN 초기 PRQ를 2.56으로 유지한다.
            'SM': 2.90, 'YN': 2.56, 'HN': 2.31, 'CC': 2.07, 'GW': 2.31,
        },
        'dict_parameter_initial_PRQ': {
            'SM': 2.90, 'YN': 2.57, 'HN': 2.31, 'CC': 2.07, 'GW': 2.31,
        },

        # 2026-09-16: Volume-Outcome Relationship (VOR) parameters
        'vor_K' : 5,
        'gamma_alpha' : 2.0,
        'gamma_beta' : 2.0,
        # 2026-09-18: 기본 VOR 최대 효과와 half-saturation 기준(실제 적용 시 병원 수로 재계산)
        'Q_vmax' : 2.0,
        'S_half' : 1 / 47,
        # 2026-09-21: 병원별 고정 base OHQ를 calibrated 초기 PRQ 주변에서 생성하는 표준편차
        'sigma_base_OHQ': 0.02,

        # 2026-10-01: 병원 투자 action의 지연된 품질개선 효과 parameters
        # 현재 action을 포함한 최근 action_effect_K개 timestep을 inverse-gamma 가중합한다.
        'action_effect_K': 5,
        'action_inv_gamma_alpha': 3.0,
        'action_inv_gamma_beta': 8.0,
        'Q_amax': 2.0,

        'sigma_OHQ': 0.02,
        'sigma_PHQ': 0.02,
        'sigma_PRQ': 0.02,
        'beta_PQ': 10.62,
        'beta_d': 20.30,
        'delta': 1.85,
        'pi': 1.0,
        'mu_scale': 0.3,
        'mu_recog': 0.9,
        'sigma_recog': 0.05,
        'n_s': 10,
        'n_c': 50,
        'sigma_wom': 0.01,
        'mu_hc': 0.32, # test
        'lambda_c': 1.5,
        'mu_hs': 0.2,
        'lambda_s': 1.5,
        'mu_rc': 0.32, # test
        'mu_rs': 0.1,
        'mu_rn': 0.1,
        'wom_update_mode': 'synchronous_vectorized',
        'gamma_c': 0.3,
        'gamma_s': 0.01,
        'sigma_ip': 0.05,
        'prov_type': 0})
    
class Simulation:
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set
        self.df_city_info = parameter_set.df_city_info
        self.df_hospital_crd = parameter_set.df_hospital_crd

        self.num_population = sum(self.parameter_set.df_city_info[self.parameter_set.target_population_column])
        self.num_hospital = sum(self.parameter_set.df_city_info['NUM_HOSPITAL'])

        self.arr_population_PHQ = np.full((self.num_population,self.num_hospital), -1.0)
        self.arr_population_PRQ = np.full((self.num_population,len(self.parameter_set.dict_parameter_initial_PRQ)), -1.0)
        self.arr_population_choice_prob = np.zeros((self.num_population,self.num_hospital))  # [2026-09-07 calculate choice probability]: Simulation-level choice probability 공유 배열
        self.arr_population_CancerOrNot = np.full(self.num_population, False)
        # 2026-10-02: 신규 cancer patient의 1회성 recognition 대상을 Simulation-level Boolean 배열로 관리한다.
        self.arr_population_is_new_cancerpatient = np.full(self.num_population, False)
        # 2026-09-28: 사람별 active cancer 종료 timestep; -1은 현재 non-cancer 상태를 뜻한다.
        self.arr_active_cancer_end_dt = np.full(self.num_population, -1, dtype=int)

        # 2026-09-16: timestep별 병원 실제 선택량을 저장하여 VOR를 계산
        self.arr_hospital_volume = np.zeros((self.parameter_set.max_dt, self.num_hospital), dtype=int)
        # 2026-09-18: base_OHQ와 분리된 VOR 품질 성분은 매 timestep 재계산하며 누적하지 않음
        self.vor_OHQ = np.zeros(self.num_hospital)
        # VOR Gamma weights
        lags = np.arange(1, self.parameter_set.vor_K + 1)

        self.vor_weights = gamma.pdf(
            lags,
            a=self.parameter_set.gamma_alpha,
            scale=self.parameter_set.gamma_beta
        )

        # 2026-10-01: timestep별 병원 투자 action과 그에 따른 품질개선 효과
        self.arr_hospital_action = np.zeros(
            (self.parameter_set.max_dt, self.num_hospital), dtype=float
        )
        self.action_OHQ = np.zeros(self.num_hospital, dtype=float)

        action_effect_K = int(self.parameter_set.action_effect_K) # time lag
        action_alpha = float(self.parameter_set.action_inv_gamma_alpha) #inverse gamma function의 param
        action_beta = float(self.parameter_set.action_inv_gamma_beta) #inverse gamma function의 param

        # 현재 action을 lag 1로 간주 (k=1) -> 추후 수정 가능
        action_lags = np.arange(1, action_effect_K + 1, dtype=float)
        self.action_effect_weights = invgamma.pdf(action_lags, a=action_alpha, scale=action_beta)
        weight_sum = self.action_effect_weights.sum()
        self.action_effect_weights /= weight_sum # normalization

        self.max_quality = parameter_set.max_quality
        self.min_quality = parameter_set.min_quality

    def initialize(self):
        """
        [ABM 구조 초기화]
        도시ㆍ병원ㆍ환자 agent와 환자 위치, 이웃관계, WOM 연결구조를 생성한다.
        아직 OHQ/PHQ/PRQ의 episode 초기값을 설정하거나 시간을 전진시키지는 않는다.

        Gymnasium 환경에서는 매 reset마다 새 Simulation 객체를 만들어야하므로 이 메서드를
        먼저 호출한다. 환자 위치 생성 등에 기존 seed 생성기를 사용하므로 호출 순서는
        기존 simulation과 동일하게 유지
        """
        self.create_Cities()
        self.create_Hospitals()
        self.create_People()
        self.set_population_regions()  # [2026-09-07 WOM]: N_r 계산에 사용할 person별 region index를 초기화 시 한 번만 생성
        self.set_person_neighbors()
        self.set_population_WOM_arrays()  # [2026-09-07 WOM]: Simulation-level WOM 계산용 neighbor edge 배열 초기화

    def execute(self):
        self.prepare_episode()
        self.run_simulation()

    def prepare_episode(self):
        """
        [episode 초기 상태 준비]
        initialize()로 만든 agent 구조 위에 초기 OHQ, PHQ, PRQ와 archive의 t=0 상태를
        설정한다. 
        Gymnasium의 reset()시 초기 observation을 반환하도록 하기위함
        """
        self.set_OHQ()
        self.set_person_initial_perception()
        self.create_Archive()

    def create_Cities(self):
        self.arr_City_NM = self.df_city_info.SIDO_NM.values
        self.arr_City = np.array([])

        for idx_city in range(len(self.arr_City_NM)):
            nm_city = self.arr_City_NM[idx_city]
            df_city_parameter = self.df_city_info.loc[self.df_city_info.SIDO_NM == nm_city]
            parameter_set_city = EasyDict({
                'idx_city' : idx_city,
                'nm_city' : nm_city,

                'population_real' : df_city_parameter.POPULATION_REAL.values[0],
                'num_population' : df_city_parameter[self.parameter_set.target_population_column].values[0],
                'num_hospital' : df_city_parameter.NUM_HOSPITAL.values[0],
                'nm_region' : df_city_parameter.REGION.values[0],
                })
            self.arr_City = np.append(self.arr_City, City(parameter_set_city))
    
    def create_Hospitals(self):
        self.arr_Hospital = np.array([])
        self.df_Hospital_info = pd.DataFrame({'idx_hospital' : range(self.df_city_info.NUM_HOSPITAL.sum())})
        self.df_Hospital_info['nm_city'] = ''
        self.df_Hospital_info['idx_city'] = 0
        self.df_Hospital_info['coord'] = Point(0,0)
        self.df_Hospital_info['OQ_objective_quality'] = -1.0

        idx_hospital = 0
        self.idx_hospital_in_region = [[],[],[],[],[]]

        for idx_city in range(len(self.arr_City_NM)):
            nm_city = self.arr_City_NM[idx_city]
            df_hospital_crd_city = self.df_hospital_crd.loc[self.df_hospital_crd.SIDO_NM == nm_city]
            arr_idx_hospital_incity = np.array([], dtype=int)

            for idx_hospital_incity in range(self.arr_City[idx_city].num_hospital):
                nm_hospital = df_hospital_crd_city.iloc[idx_hospital_incity].INST_NM
                hospital_crd = Point(transform(proj_tm_mid, proj_wgs84, 
                                              df_hospital_crd_city.iloc[idx_hospital_incity].loc['CRD_X'], 
                                              df_hospital_crd_city.iloc[idx_hospital_incity].loc['CRD_Y']))
                parameter_set_hospital = EasyDict({
                    'idx_hospital' : idx_hospital,
                    'nm_hospital' : nm_hospital,
                    'idx_city' : idx_city,
                    'nm_city' : nm_city,
                    'nm_hospital' : nm_hospital,
                    'coord' : hospital_crd
                    })
                self.arr_Hospital = np.append(self.arr_Hospital, Hospital(parameter_set_hospital))
                arr_idx_hospital_incity = np.append(arr_idx_hospital_incity, idx_hospital)

                self.df_Hospital_info.loc[idx_hospital, 'nm_hospital'] = nm_hospital
                self.df_Hospital_info.loc[idx_hospital, 'nm_city'] = nm_city
                self.df_Hospital_info.loc[idx_hospital, 'idx_city'] = idx_city
                self.df_Hospital_info.loc[idx_hospital, 'coord'] = hospital_crd
                idx_hospital += 1

            self.arr_City[idx_city].arr_idx_hospital_incity = arr_idx_hospital_incity
        
        self.df_Hospital_info = pd.merge(self.df_Hospital_info, self.df_city_info[['SIDO_NM', 'REGION']], how='left', left_on = 'nm_city', right_on = 'SIDO_NM')
        region_mapping = {}
        for idx in range(len(self.parameter_set.dict_parameter_initial_PRQ)):
            region_mapping[list(self.parameter_set.dict_parameter_initial_PRQ.keys())[idx]] = idx
        self.df_Hospital_info['idx_REGION'] = self.df_Hospital_info['REGION'].map(region_mapping)

        for idx_REGION in range(len(region_mapping.keys())):
            self.idx_hospital_in_region[idx_REGION] = self.df_Hospital_info.loc[self.df_Hospital_info.idx_REGION == idx_REGION, 'idx_hospital'].values

        # [2026-09-07 calculate choice probability]: choice probability vectorization에 사용할 고정 hospital-region index
        self.arr_hospital_region = self.df_Hospital_info.idx_REGION.values.astype(int)
        self.idx_representative_hospital_by_region = np.array([
            idx_hospital[0] for idx_hospital in self.idx_hospital_in_region
        ])
        self.num_region = len(self.idx_hospital_in_region)
    
    def create_People(self):
        self.arr_Person = np.array([])

        self.df_Person_info = pd.DataFrame({'idx_person' : range(self.df_city_info[self.parameter_set.target_population_column].sum())})
        self.df_Person_info['nm_city'] = ''
        self.df_Person_info['idx_city'] = 0
        self.df_Person_info['coord'] = Point(0,0)
        self.df_Person_info['is_cancerpatient'] = False
        self.df_Person_info['idx_chosen_hospital'] = -1
        self.df_Person_info['idx_region_chosen_hospital'] = -1
        self.set_base_PHQ()
        idx_person = 0

        for idx_city in range(len(self.arr_City_NM)):
            nm_city = self.arr_City_NM[idx_city]
            df_city_parameter = self.df_city_info.loc[self.df_city_info.SIDO_NM == nm_city]

            arr_idx_person_incity = np.array([], dtype=int)

            geom_city = gdf.loc[gdf.CTP_KOR_NM == nm_city].iloc[0].geometry
            random_coords = random_points_in_multipolygon(gdf.loc[gdf.CTP_KOR_NM == nm_city].iloc[0].geometry, self.arr_City[idx_city].num_population)
            
            for idx_person_incity in range(self.arr_City[idx_city].num_population):
                parameter_set_person = EasyDict({                
                    'idx_person' : idx_person,
                    'is_cancerpatient' : False,
                    'is_new_cancerpatient' : False,
                    
                    'idx_city' :    idx_city,
                    'nm_city' :     nm_city,
                    'coord' :       random_coords[idx_person_incity],
                    'dict_parameter_initial_PRQ' : self.parameter_set.dict_parameter_initial_PRQ,

                    'sigma_OHQ' :   self.parameter_set.sigma_OHQ,
                    'sigma_OHQ' :   self.parameter_set.sigma_OHQ,
                    'sigma_PRQ' :   self.parameter_set.sigma_PRQ,

                    'beta_PQ' :     self.parameter_set.beta_PQ,
                    'beta_d' :      self.parameter_set.beta_d,
                    'delta' :       self.parameter_set.delta,
                    'pi' :          self.parameter_set.pi,
                    
                    'df_Hospital_info' : self.df_Hospital_info,
                    'df_Person_info' : self.df_Person_info,
                    'idx_hospital_in_region' : self.idx_hospital_in_region,
                    'arr_hospital_region' : self.arr_hospital_region,
                    'idx_representative_hospital_by_region' : self.idx_representative_hospital_by_region,

                    'arr_population_PHQ' : self.arr_population_PHQ,
                    'arr_population_PRQ' : self.arr_population_PRQ,
                    'arr_population_choice_prob' : self.arr_population_choice_prob,
                    'arr_population_CancerOrNot' : self.arr_population_CancerOrNot,

                    'base_PHQ' : self.base_PHQ
                    })
                

                self.arr_Person = np.append(self.arr_Person, Person(parameter_set_person, self.parameter_set))
                arr_idx_person_incity = np.append(arr_idx_person_incity, idx_person)
                self.df_Person_info.loc[idx_person, 'nm_city'] = nm_city
                self.df_Person_info.loc[idx_person, 'idx_city'] = idx_city
                self.df_Person_info.loc[idx_person, 'coord'] = random_coords[idx_person_incity]
                self.df_Person_info.loc[idx_person, 'is_cancerpatient'] = parameter_set_person.is_cancerpatient
                idx_person += 1
            
            # 각 도시의 인구 idx 넣어주기
            self.arr_City[idx_city].arr_idx_person_incity = arr_idx_person_incity

        self.df_Person_info = pd.merge(self.df_Person_info, 
                                       self.df_city_info[['SIDO_NM', 'REGION']],
                                       how='left', left_on = 'nm_city', right_on = 'SIDO_NM')

    # [2026-09-07 WOM]: N_r의 반복적인 DataFrame region 조회를 대체할 population-level region index
    def set_population_regions(self):
        region_mapping = {region: idx for idx, region in enumerate(self.parameter_set.dict_parameter_initial_PRQ.keys())}
        self.arr_population_region = self.df_Person_info.REGION.map(region_mapping).values
        self.idx_person_by_region = [
            np.where(self.arr_population_region == idx_region)[0]
            for idx_region in range(len(region_mapping))
        ]
        self.num_person_by_region = np.array([len(idx_person) for idx_person in self.idx_person_by_region])



    def create_Archive(self):
        self.achv_population_is_cancerpatient = np.full((self.parameter_set.max_dt, self.num_population), False)
        self.achv_population_is_cancerpatient[0] = self.df_Person_info.is_cancerpatient.values

        self.achv_population_choice_prob = np.full((self.parameter_set.max_dt, self.num_population, self.num_hospital), 0.0)
        # [2026-09-16] : VOR을 사용하기 위한 timestep별 병원 선택량을 저장하는 배열 추가
        self.achv_mean_hospital_choice_prob = np.zeros((self.parameter_set.max_dt, self.num_hospital))

        self.achv_population_PRQ = np.full((self.parameter_set.max_dt, self.num_population, len(self.parameter_set.dict_parameter_initial_PRQ)), 0.0)
        self.achv_population_PHQ = np.full((self.parameter_set.max_dt, self.num_population, self.num_hospital), 0.0)
        self.achv_population_True_Utility = np.full((self.parameter_set.max_dt, self.num_population, self.num_hospital), 0.0)
        self.achv_OHQ = np.full((self.parameter_set.max_dt, self.num_hospital), 0.0)
        # 2026-09-18: 총 OHQ를 base/VOR 성분으로 분해해 확인할 수 있도록 archive를 추가
        self.achv_base_OHQ = np.zeros((self.parameter_set.max_dt, self.num_hospital))
        self.achv_vor_OHQ = np.zeros((self.parameter_set.max_dt, self.num_hospital))
        self.achv_action_OHQ = np.zeros((self.parameter_set.max_dt, self.num_hospital))

        self.achv_population_choice_prob[0] = self.arr_population_choice_prob  # [2026-09-07 calculate choice probability]: Simulation-level choice probability archive 초기화
        self.achv_mean_hospital_choice_prob[0] = np.mean(self.arr_population_choice_prob, axis=0)  # [2026-09-16] : VOR을 사용하기 위한 timestep별 병원 선택량 초기화
        for idx_person in self.df_Person_info.idx_person.values:
            self.achv_population_PRQ[0,idx_person] = self.arr_Person[idx_person].PRQ
            self.achv_population_PHQ[0,idx_person] = self.arr_Person[idx_person].PHQ
        self.achv_OHQ[0] = self.df_Hospital_info.OQ_objective_quality.values
        self.achv_base_OHQ[0] = self.base_OHQ
        self.achv_vor_OHQ[0] = self.vor_OHQ
        self.achv_action_OHQ[0] = self.action_OHQ


    def update_Archive(self, dt):
        self.achv_population_is_cancerpatient[dt] = self.df_Person_info.is_cancerpatient.values
        self.achv_population_choice_prob[dt] = self.arr_population_choice_prob  # [2026-09-07 calculate choice probability]: vectorized choice probability를 archive에 저장
        self.achv_mean_hospital_choice_prob[dt] = np.mean(self.arr_population_choice_prob, axis=0)  # [2026-09-16] : VOR을 사용하기 위한 timestep별 병원 선택량 업데이트
        for idx_person in self.df_Person_info.idx_person.values:
            self.achv_population_PRQ[dt,idx_person] = self.arr_Person[idx_person].PRQ
            self.achv_population_PHQ[dt,idx_person] = self.arr_Person[idx_person].PHQ
        self.achv_OHQ[dt] = self.df_Hospital_info.OQ_objective_quality.values
        # 2026-09-18: 해당 timestep의 base/VOR OHQ 성분을 총 OHQ와 함께 저장
        self.achv_base_OHQ[dt] = self.base_OHQ
        self.achv_vor_OHQ[dt] = self.vor_OHQ
        self.achv_action_OHQ[dt] = self.action_OHQ

    # Initial Setting
    def set_OHQ(self):
        # 2026-09-21: base OHQ는 기존 calibrated PRQ(0~5)를 0~1로 정규화한 지역값 주변에서 한 번만 생성한다.
        # initial_PRQ_by_region = np.asarray(
        #     list(self.parameter_set.dict_parameter_initial_PRQ.values()), dtype=float
        # ) / 3.0
        # self.base_OHQ = np.clip(
        #     initial_PRQ_by_region[self.arr_hospital_region]
        #     + np.random.normal(0.0, self.parameter_set.sigma_base_OHQ, self.num_hospital),
        #     0.0,
        #     1.0,
        # )
        
        # 2026-09-28 base OHQ 1로 고정
        self.base_OHQ = np.ones(self.num_hospital) 
        self.vor_OHQ.fill(0.0)
        self.action_OHQ.fill(0.0)
        # 2026-09-21: dt=0 VOR는 PHQ/OHQ 없이 초기 PRQ와 지역 거리만으로 만든 가상 과거 선택확률을 사용한다.
        self.calculate_virtual_choice_prob()
        self.VOR_effect(self.parameter_set.Q_vmax, dt=0)
        self.update_total_OHQ()


    def set_person_initial_perception(self):
        # 2026-09-11: 초기 PHQ/PRQ와 거리효과를 population-level NumPy 연산으로 한 번에 생성
        # 2026-09-21: 초기 PHQ는 base가 아니라 dt=0 VOR까지 반영된 total OHQ 주변에서 생성한다.
        self.arr_population_PHQ[:] = np.clip(
            self.df_Hospital_info.OQ_objective_quality.values[np.newaxis, :] + np.random.normal(
                0, self.parameter_set.sigma_PHQ, (self.num_population, self.num_hospital)
            ),
            0.0, 3.0
        )
        initial_PRQ = np.asarray(list(self.parameter_set.dict_parameter_initial_PRQ.values()))
        self.arr_population_PRQ[:] = np.clip(
            initial_PRQ[np.newaxis, :] + np.random.normal(
                0, self.parameter_set.sigma_PRQ, (self.num_population, self.num_region)
            ),
            0.0, 3.0
        )

        # 2026-09-11: Shapely Point.distance와 동일한 평면 좌표 거리식을 broadcasting으로 계산
        person_coords = np.array([[coord.x, coord.y] for coord in self.df_Person_info.coord.values])
        hospital_coords = np.array([[coord.x, coord.y] for coord in self.df_Hospital_info.coord.values])
        region_coords = np.array([
            [self.parameter_set.dict_crd_region[region].x, self.parameter_set.dict_crd_region[region].y]
            for region in self.parameter_set.dict_parameter_initial_PRQ.keys()
        ])
        self.arr_distance_effect_hospital = np.exp(
            -self.parameter_set.delta * np.sqrt(
                np.sum((person_coords[:, np.newaxis, :] - hospital_coords[np.newaxis, :, :]) ** 2, axis=2)
            )
        )
        self.arr_distance_effect_region = np.exp(
            -self.parameter_set.delta * np.sqrt(
                np.sum((person_coords[:, np.newaxis, :] - region_coords[np.newaxis, :, :]) ** 2, axis=2)
            )
        )
        # 2026-09-11: region 거리효과(P, R)를 choice utility에서 사용할 hospital 축(P, H)으로만 매핑
        self.arr_distance_effect_region_hospital = self.arr_distance_effect_region[:, self.arr_hospital_region]

        # 2026-09-11: archive 호환에 필요한 상태만 shared population 배열의 row view로 연결
        for idx_person, person in enumerate(self.arr_Person):
            person.PHQ = self.arr_population_PHQ[idx_person]
            person.PRQ = self.arr_population_PRQ[idx_person]
            person.Q_h[0] = person.PHQ
            person.Q_r[0] = person.PRQ
            person.arr_distance_effect_hospital = self.arr_distance_effect_hospital[idx_person]
            person.arr_distance_effect_region = self.arr_distance_effect_region_hospital[idx_person]
            person.df_Person_info = self.df_Person_info

        self.calculate_choice_prob_all()

    # [2026-09-07 calculate choice probability]: 모든 person의 utility를 (P, H) NumPy 연산으로 계산
    def calculate_utility_all(self):
        arr_PRQ_hospital = self.arr_population_PRQ[:, self.arr_hospital_region]
        self.arr_utility_region = (
            arr_PRQ_hospital * self.parameter_set.beta_PQ
            + self.arr_distance_effect_region_hospital * self.parameter_set.beta_d
        )
        self.arr_utility_hospital = (
            self.arr_population_PHQ * self.parameter_set.beta_PQ
            + self.arr_distance_effect_hospital * self.parameter_set.beta_d
        )

    # [2026-09-07 calculate choice probability]: 모든 person의 2-stage/standard logit choice probability를 NumPy로 계산
    def calculate_choice_prob_all(self):
        self.calculate_utility_all()

        if self.parameter_set.pi != 0:
            exp_u_region = np.exp(self.arr_utility_region)
            regional_denominator = np.sum(
                exp_u_region[:, self.idx_representative_hospital_by_region], axis=1
            )
            self.arr_population_choice_prob_region = exp_u_region / regional_denominator[:, np.newaxis]

            exp_u_hospital = np.exp(self.arr_utility_hospital * self.parameter_set.pi)
            sum_exp_by_region = np.zeros((self.num_population, self.num_region))
            np.add.at(
                sum_exp_by_region,
                (np.arange(self.num_population)[:, np.newaxis], self.arr_hospital_region[np.newaxis, :]),
                exp_u_hospital
            )
            self.arr_population_choice_prob_hospital = (
                exp_u_hospital / sum_exp_by_region[:, self.arr_hospital_region]
            )
            self.arr_population_choice_prob[:] = (
                self.arr_population_choice_prob_region
                * self.arr_population_choice_prob_hospital
            )
        else:
            exp_u_hospital = np.exp(self.arr_utility_hospital)
            inf_mask = exp_u_hospital == np.inf
            if np.any(inf_mask):
                non_inf_mask = ~inf_mask
                if np.any(~np.any(non_inf_mask, axis=1)):
                    raise ValueError("max() arg is an empty sequence")
                first_non_inf = np.argmax(non_inf_mask, axis=1)
                replacement = exp_u_hospital[np.arange(self.num_population), first_non_inf]
                # [2026-09-07 calculate choice probability]: Python max와 같은 NaN 비교 순서를 hospital 축에서만 재현
                for idx_hospital in range(self.num_hospital):
                    candidate = exp_u_hospital[:, idx_hospital]
                    replacement = np.where(
                        non_inf_mask[:, idx_hospital] & (candidate > replacement),
                        candidate, replacement
                    )
                exp_u_hospital[inf_mask] = np.broadcast_to(
                    replacement[:, np.newaxis], exp_u_hospital.shape
                )[inf_mask]
            self.arr_population_choice_prob_hospital = (
                exp_u_hospital / np.sum(exp_u_hospital, axis=1, keepdims=True)
            )
            self.arr_population_choice_prob[:] = self.arr_population_choice_prob_hospital


    def set_person_neighbors(self):
        coords_people = np.array([[point.x, point.y] for point in self.df_Person_info.coord.values])
        kdtree = KDTree(coords_people)
        nearest_neighbors_indices = kdtree.query(coords_people, k=max(self.parameter_set.n_s, 
                                                                      self.parameter_set.n_c), 
                                                 return_distance=False)[:, 1:]

        for idx_person in range(len(self.arr_Person)):
            self.arr_Person[idx_person].list_idx_neighbor_susceptible = nearest_neighbors_indices[idx_person,:self.parameter_set.n_s]
            self.arr_Person[idx_person].list_dist_neighbor_susceptible = [self.df_Person_info.loc[idx_person].coord.distance(self.df_Person_info.loc[idx_neighbor].coord) for idx_neighbor in nearest_neighbors_indices[idx_person,:self.parameter_set.n_s]]
            self.arr_Person[idx_person].list_idx_neighbor_cancer = nearest_neighbors_indices[idx_person,:self.parameter_set.n_c]
            self.arr_Person[idx_person].list_dist_neighbor_cancer = [self.df_Person_info.loc[idx_person].coord.distance(self.df_Person_info.loc[idx_neighbor].coord) for idx_neighbor in nearest_neighbors_indices[idx_person,:self.parameter_set.n_c]]

    # [2026-09-07 WOM]: person/neighbor loop 없이 WOM contribution을 계산할 flattened edge 배열 생성
    def set_population_WOM_arrays(self):
        idx_person = np.arange(self.num_population)
        cancer_neighbor = np.array([person.list_idx_neighbor_cancer for person in self.arr_Person])
        cancer_distance = np.array([person.list_dist_neighbor_cancer for person in self.arr_Person])
        susceptible_neighbor = np.array([person.list_idx_neighbor_susceptible for person in self.arr_Person])
        susceptible_distance = np.array([person.list_dist_neighbor_susceptible for person in self.arr_Person])
        self.arr_wom_cancer_source = cancer_neighbor.reshape(-1)
        self.arr_wom_cancer_distance = cancer_distance.reshape(-1)
        self.arr_wom_cancer_target = np.repeat(idx_person, cancer_neighbor.shape[1])
        self.arr_wom_susceptible_source = susceptible_neighbor.reshape(-1)
        self.arr_wom_susceptible_distance = susceptible_distance.reshape(-1)
        self.arr_wom_susceptible_target = np.repeat(idx_person, susceptible_neighbor.shape[1])

    # [2026-09-07 WOM]: cancer neighbor의 hospital WOM contribution을 (P, H)로 synchronous 합산
    def calculate_C_h(self):
        C_h = np.zeros((self.num_population, self.num_hospital))
        source, target = self.arr_wom_cancer_source, self.arr_wom_cancer_target
        hospital = self.df_Person_info.idx_chosen_hospital.values[source]
        valid = self.arr_population_CancerOrNot[source] & ~self.arr_population_CancerOrNot[target] & (hospital >= 0)
        source, target, hospital = source[valid], target[valid], hospital[valid]
        diff = self.arr_population_PHQ[source, hospital] - self.arr_population_PHQ[target, hospital]
        contribution = self.parameter_set.mu_hc * diff * (diff**2 <= self.parameter_set.lambda_c**2) * np.exp(-self.parameter_set.delta * self.arr_wom_cancer_distance[valid])
        np.add.at(C_h, (target, hospital), contribution)

        return C_h
        

    # [2026-09-07 WOM]: susceptible neighbor의 hospital WOM contribution을 (P, H)로 synchronous 합산
    def calculate_S_h(self):
        S_h = np.zeros((self.num_population, self.num_hospital))
        source, target = self.arr_wom_susceptible_source, self.arr_wom_susceptible_target
        valid = ~self.arr_population_CancerOrNot[source] & ~self.arr_population_CancerOrNot[target]
        # valid = ~self.arr_population_CancerOrNot[target]
        source, target = source[valid], target[valid]
        diff = self.arr_population_PHQ[source] - self.arr_population_PHQ[target]
        contribution = self.parameter_set.mu_hs * diff * (diff**2 <= self.parameter_set.lambda_s**2) * np.exp(-self.parameter_set.delta * self.arr_wom_susceptible_distance[valid])[:, np.newaxis]
        np.add.at(S_h, target, contribution)
        # target별 유효 susceptible neighbor 수
        counts = np.zeros(self.num_population, dtype=int)
        np.add.at(counts, target, 1)

        # 합산값을 이웃 수로 나누어 평균 WOM effect로 변환
        has_neighbors = counts > 0
        S_h[has_neighbors] /= counts[has_neighbors, np.newaxis]
        return S_h

    # [2026-09-07 WOM]: cancer neighbor의 regional WOM contribution을 (P, R)로 synchronous 합산
    def calculate_C_r(self):
        C_r = np.zeros((self.num_population, self.num_region))
        source, target = self.arr_wom_cancer_source, self.arr_wom_cancer_target
        region = self.df_Person_info.idx_region_chosen_hospital.values[source]
        valid = self.arr_population_CancerOrNot[source] & ~self.arr_population_CancerOrNot[target] & (region >= 0)
        source, target, region = source[valid], target[valid], region[valid]
        diff = self.arr_population_PRQ[source, region] - self.arr_population_PRQ[target, region]
        contribution = self.parameter_set.mu_rc * diff * (diff**2 <= self.parameter_set.lambda_c**2) * np.exp(-self.parameter_set.delta * self.arr_wom_cancer_distance[valid])
        np.add.at(C_r, (target, region), contribution)

        return C_r

    # [2026-09-07 WOM]: susceptible neighbor의 regional WOM contribution을 (P, R)로 synchronous 합산
    def calculate_S_r(self):
        S_r = np.zeros((self.num_population, self.num_region))
        source, target = self.arr_wom_susceptible_source, self.arr_wom_susceptible_target
        valid = ~self.arr_population_CancerOrNot[source] & ~self.arr_population_CancerOrNot[target]
        # valid = ~self.arr_population_CancerOrNot[target]
        source, target = source[valid], target[valid]
        diff = self.arr_population_PRQ[source] - self.arr_population_PRQ[target]
        contribution = self.parameter_set.mu_rs * diff * (diff**2 <= self.parameter_set.lambda_s**2) * np.exp(-self.parameter_set.delta * self.arr_wom_susceptible_distance[valid])[:, np.newaxis]
        np.add.at(S_r, target, contribution)
        # target별 유효 susceptible neighbor 수
        counts = np.zeros(self.num_population, dtype=int)
        np.add.at(counts, target, 1)

        # 합산값을 이웃 수로 나누어 평균 WOM effect로 변환
        has_neighbors = counts > 0
        S_r[has_neighbors] /= counts[has_neighbors, np.newaxis]
        return S_r

    # [2026-09-07 WOM]: timestep 시작 PRQ 기준 region 평균으로 N_r을 synchronous 계산
    # 2026-09-30 현재 사용하지 않음(indirect effect 삭제)
    # def calculate_N_r_all(self):
    #     mean_PRQ_by_region = np.array([
    #         np.mean(self.arr_population_PRQ[idx_person], axis=0)
    #         for idx_person in self.idx_person_by_region
    #     ])
    #     return self.parameter_set.mu_rn * (mean_PRQ_by_region[self.arr_population_region] - self.arr_population_PRQ)

    def calculate_recognition_all(self, dt):
        """Calculate one-time cancer-patient recognition effects without Person-level math."""
        # 2026-10-02: 신규 cancer patient의 병원/지역 recognition을 (P,H), (P,R) 배열에 한 번에 scatter한다.
        E_h = np.zeros((self.num_population, self.num_hospital))
        E_r = np.zeros((self.num_population, self.num_region))
        idx_new_patient = np.flatnonzero(self.arr_population_is_new_cancerpatient)
        if len(idx_new_patient) == 0:
            return E_h, E_r

        chosen_hospital = self.df_Person_info.idx_chosen_hospital.to_numpy()[idx_new_patient]
        chosen_region = self.df_Person_info.idx_region_chosen_hospital.to_numpy()[idx_new_patient]
        if np.any(chosen_hospital < 0) or np.any(chosen_region < 0):
            raise ValueError("New cancer patient must choose a hospital before recognition.")

        objective_quality = self.df_Hospital_info.OQ_objective_quality.to_numpy()
        recognition_noise_h = np.random.normal(0, self.parameter_set.sigma_recog, len(idx_new_patient))
        recognition_noise_r = np.random.normal(0, self.parameter_set.sigma_recog, len(idx_new_patient))
        E_h[idx_new_patient, chosen_hospital] = (
            self.parameter_set.mu_recog
            * (objective_quality[chosen_hospital] - self.arr_population_PHQ[idx_new_patient, chosen_hospital])
            + recognition_noise_h
        )
        E_r[idx_new_patient, chosen_region] = (
            self.parameter_set.mu_recog
            * (objective_quality[chosen_hospital] - self.arr_population_PRQ[idx_new_patient, chosen_region])
            + recognition_noise_r
        )

        # 2026-10-02: recognition을 한 번 적용한 즉시 Simulation/Person의 신규 flag를 함께 해제한다.
        self.arr_population_is_new_cancerpatient[idx_new_patient] = False
        for idx_person in idx_new_patient:
            self.arr_Person[idx_person].is_new_cancerpatient = False
        return E_h, E_r

    # [2026-09-07 WOM]: C/S/N 효과를 모두 계산한 뒤 PHQ와 PRQ를 동시에 갱신
    def update_WOM_all(self, dt):
        C_h, S_h = self.calculate_C_h(), self.calculate_S_h()
        # C_r, S_r, N_r = self.calculate_C_r(), self.calculate_S_r(), self.calculate_N_r_all()
        C_r, S_r = self.calculate_C_r(), self.calculate_S_r()

        # 2026-10-02: Person.PHQ/PRQ_recognition 호출 대신 population-level recognition 효과를 사용한다.
        E_h, E_r = self.calculate_recognition_all(dt)

        new_PHQ = np.clip(
            self.arr_population_PHQ + C_h + S_h + E_h + np.random.normal(0, self.parameter_set.sigma_wom, self.arr_population_PHQ.shape),
            self.min_quality, self.max_quality
        )
        new_PRQ = np.clip(
            # self.arr_population_PRQ + C_r + S_r + N_r + E_r + np.random.normal(0, self.parameter_set.sigma_wom, self.arr_population_PRQ.shape),
            self.arr_population_PRQ + C_r + S_r + E_r + np.random.normal(0, self.parameter_set.sigma_wom, self.arr_population_PRQ.shape),
            self.min_quality, self.max_quality
        )
        self.arr_population_PHQ[:], self.arr_population_PRQ[:] = new_PHQ, new_PRQ

        for idx_person, person in enumerate(self.arr_Person):
            person.C_h[dt], person.S_h[dt], person.E_h[dt] = C_h[idx_person], S_h[idx_person], E_h[idx_person]
            # person.C_r[dt], person.S_r[dt], person.N_r[dt], person.E_r[dt] = C_r[idx_person], S_r[idx_person], N_r[idx_person], E_r[idx_person]
            person.C_r[dt], person.S_r[dt], person.E_r[dt] = C_r[idx_person], S_r[idx_person], E_r[idx_person]
            person.Q_h[dt], person.Q_r[dt] = new_PHQ[idx_person], new_PRQ[idx_person]
            person.PHQ, person.PRQ = new_PHQ[idx_person], new_PRQ[idx_person]


    def set_base_PHQ(self):
        self.base_PHQ = np.zeros(len(self.df_Hospital_info))
        for region in self.parameter_set.dict_parameter_initial_PRQ.keys():
            self.base_PHQ[self.df_Hospital_info.loc[self.df_Hospital_info.REGION == region].index] = self.parameter_set.dict_parameter_initial_PRQ[region] / 3.0

    def expire_active_cancerpatients(self, dt):
        """Return people whose active cancer duration has ended to the candidate pool."""
        # 2026-09-28: 종료 timestep(dt)에 도달한 사람은 cancer 상태를 해제하되, 과거 PHQ/PRQ와 선택 병원 정보는 보존한다.
        idx_expired = np.flatnonzero(
            (self.arr_active_cancer_end_dt >= 0)
            & (self.arr_active_cancer_end_dt <= dt)
        )
        if len(idx_expired) == 0:
            return

        self.arr_population_CancerOrNot[idx_expired] = False
        # 2026-10-02: 치료 종료자는 남아 있을 수 있는 신규 recognition flag도 함께 해제한다.
        self.arr_population_is_new_cancerpatient[idx_expired] = False
        self.arr_active_cancer_end_dt[idx_expired] = -1
        self.df_Person_info.loc[
            self.df_Person_info.idx_person.isin(idx_expired), 'is_cancerpatient'
        ] = False
        for idx_person in idx_expired:
            self.arr_Person[idx_person].is_cancerpatient = False
            self.arr_Person[idx_person].is_new_cancerpatient = False

    def designate_new_cancerpatient(self, dt):
        # 2026-09-28: 현재 active cancer 상태가 아닌 사람만 신규 cancer patient 후보가 된다.
        idx_candidate = np.flatnonzero(~self.arr_population_CancerOrNot)
        if len(idx_candidate) < self.parameter_set.n_p:
            raise ValueError(
                f"Not enough non-active cancer candidates at dt={dt}: "
                f"need {self.parameter_set.n_p}, found {len(idx_candidate)}."
            )
        self.idx_new_patient = np.random.choice(
            idx_candidate, size=self.parameter_set.n_p, replace=False
        )
        for idx_person in self.idx_new_patient:
            self.arr_Person[idx_person].is_cancerpatient = True
            self.arr_Person[idx_person].is_new_cancerpatient = True
            self.arr_Person[idx_person].calculate_choice_prob()
            self.arr_Person[idx_person].choose_hospital()
            self.arr_population_CancerOrNot[idx_person] = True
            # 2026-09-28: dt부터 dt + cancer_duration - 1까지 active이며, 종료 timestep부터 expire_active_cancerpatients가 해제한다.
            self.arr_active_cancer_end_dt[idx_person] = dt + self.parameter_set.cancer_duration
        # 2026-10-02: 이번 timestep 신규 지정자만 vectorized recognition 대상에 표시한다.
        self.arr_population_is_new_cancerpatient[self.idx_new_patient] = True
        
        self.df_Person_info.loc[self.df_Person_info.idx_person.isin(self.idx_new_patient), 'is_cancerpatient'] = True

    # 전체 simluation (using advance_one_timestep)
    def run_simulation(self):
        for dt in tqdm(range(self.parameter_set.max_dt)):
            self.advance_one_timestep(dt)

    # Scale Effect, 환자 선택에 따른 scale effect만 있는 version (naive version)
    # 2026-09-30 기준 사용 x
    # def scale_effect(self, dt):
    #     mean_hospital_choice_prob = np.mean(self.achv_population_choice_prob[dt-1], axis=0)
    #     surplus_choice_prob = mean_hospital_choice_prob - np.mean(mean_hospital_choice_prob)
    #     boundary_smoothing_factor = np.abs(np.array([
    #                                 np.min((self.df_Hospital_info.OQ_objective_quality.values[i] - self.min_quality, 
    #                                     self.max_quality - self.df_Hospital_info.OQ_objective_quality.values[i]))
    #                                     for i in range(len(self.df_Hospital_info))])) / \
    #                                     (self.max_quality - self.min_quality) * 2 # (0~1)

    #     new_OHQ = (1 + surplus_choice_prob * self.parameter_set.mu_scale * boundary_smoothing_factor) * \
    #                 self.df_Hospital_info.OQ_objective_quality.values
    #     new_OHQ[np.where(new_OHQ>self.max_quality)] = self.max_quality
    #     new_OHQ[np.where(new_OHQ<self.min_quality)] = self.min_quality
    #     self.df_Hospital_info.OQ_objective_quality = new_OHQ
    
    def calculate_virtual_choice_prob(self):
        """Create the fixed pseudo-history used only before enough observed choices exist."""
        # 2026-09-21: PHQ/OHQ 없이 calibrated 초기 PRQ와 사람-지역 거리로 region choice probability를 계산
        initial_PRQ = np.asarray(
            list(self.parameter_set.dict_parameter_initial_PRQ.values()), dtype=float
        )
        person_coords = np.array([[coord.x, coord.y] for coord in self.df_Person_info.coord.values])
        region_coords = np.array([
            [self.parameter_set.dict_crd_region[region].x, self.parameter_set.dict_crd_region[region].y]
            for region in self.parameter_set.dict_parameter_initial_PRQ.keys()
        ])
        distance_effect_region = np.exp(-self.parameter_set.delta * np.sqrt(
            np.sum((person_coords[:, np.newaxis, :] - region_coords[np.newaxis, :, :]) ** 2, axis=2)
        ))
        utility_region = initial_PRQ[np.newaxis, :] * self.parameter_set.beta_PQ + distance_effect_region * self.parameter_set.beta_d
        exp_utility_region = np.exp(utility_region)
        choice_prob_region = exp_utility_region / np.sum(exp_utility_region, axis=1, keepdims=True)

        # 2026-09-21: 지역 선택확률을 해당 지역 병원에 균등 배분해 hospital-level 가상 선택확률을 만듬
        hospital_count_by_region = np.bincount(self.arr_hospital_region, minlength=self.num_region)
        individual_virtual_choice_prob = choice_prob_region[:, self.arr_hospital_region] / hospital_count_by_region[self.arr_hospital_region]
        self.virtual_choice_prob = np.mean(individual_virtual_choice_prob, axis=0)
        if not np.isclose(self.virtual_choice_prob.sum(), 1.0):
            raise ValueError("virtual_choice_prob must sum to one")

        # 2026-09-21: 행은 [-vor_K, ..., -2, -1] 순서이며, VOR 계산 시 역순으로 최신 가상 과거부터 사용
        self.prehistory_choice_prob = np.repeat(
            self.virtual_choice_prob[np.newaxis, :], self.parameter_set.vor_K, axis=0
        )

    def VOR_effect(self, q_vmax, dt):
        # 2026-09-21: archive의 dt-1부터 실제 과거를 최신순으로 가져옴
        if dt > 0 and hasattr(self, 'achv_mean_hospital_choice_prob'):
            observed_history = self.achv_mean_hospital_choice_prob[:dt][::-1]
        else:
            observed_history = np.empty((0, self.num_hospital))
        observed_history = observed_history[:self.parameter_set.vor_K]

        # 2026-09-21: 부족한 과거만 -1, -2, ... 순서의 가상 선택확률로 보충해 항상 vor_K개 history를 구성
        missing = max(0, self.parameter_set.vor_K - len(observed_history))
        pseudo_history = self.prehistory_choice_prob[::-1][:missing]
        history = np.vstack([observed_history, pseudo_history])
        weights = self.vor_weights[:len(history)].copy()
        weights /= weights.sum()
        S_h = np.sum(history * weights[:, np.newaxis], axis=0)
        # 2026-09-21: VOR는 비누적 성분으로 매 timestep 재계산하며 Q_vmax=2, S_half=1/H를 적용
        Q_vmax = q_vmax
        S_half = 1.0 / self.num_hospital
        self.vor_OHQ = Q_vmax * S_h / (S_half + S_h)

    def investment_effect(self, hospital_action, dt):
        """Calculate the delayed OHQ improvement from current and past actions."""

        # binary action
        action = np.asarray(hospital_action, dtype=float)

        self.arr_hospital_action[dt] = action
        history_length = min(dt + 1, int(self.parameter_set.action_effect_K))
        action_history = self.arr_hospital_action[dt - history_length + 1:dt + 1][::-1]
        weights = self.action_effect_weights[:history_length]
        effective_action = np.sum(action_history * weights[:, np.newaxis], axis=0)
        self.action_OHQ = self.parameter_set.Q_amax * effective_action

    def update_total_OHQ(self):
        # 2026-10-01: 총 OHQ는 base, 비누적 VOR, 지연된 action 효과의 합으로 갱신
        self.df_Hospital_info["OQ_objective_quality"] = np.clip(
            self.base_OHQ + self.vor_OHQ + self.action_OHQ,
            self.min_quality,
            self.max_quality,)

    # Information Provision
    def Information_Provision_Policy(self, dt):
        idx_susceptible = np.array(list(set(self.df_Person_info['idx_person'].values) - set(self.idx_new_patient)))
        idx_susceptible_info_prov = np.random.choice(idx_susceptible, 
                                                     size=int(np.round(len(idx_susceptible) * 
                                                                       self.parameter_set.gamma_s)), 
                                                                       replace=False)
        idx_patient_info_prov = np.random.choice(self.idx_new_patient,
                                                 size=int(np.round(len(self.idx_new_patient) * 
                                                                   self.parameter_set.gamma_c)),
                                                 replace=False)

        # Regional-level IPP
        if self.parameter_set.prov_type == 1:
            temp_PRQ = self.df_Hospital_info.groupby('idx_REGION')['OQ_objective_quality'].mean().values
            for idx_person in np.concatenate((idx_susceptible_info_prov, idx_patient_info_prov)):
                self.arr_Person[idx_person].PRQ = np.random.normal(temp_PRQ, 
                                                    self.parameter_set.sigma_ip, 
                                                    len(temp_PRQ))
        
        # Hospital-level IPP
        if self.parameter_set.prov_type == 2:
            for idx_person in np.concatenate((idx_susceptible_info_prov, idx_patient_info_prov)):
                self.arr_Person[idx_person].PHQ = np.random.normal(self.achv_OHQ[dt-1], 
                                                    self.parameter_set.sigma_ip, 
                                                    len(self.df_Hospital_info))

    # gym과 호환을 위한 one step transition function
    def advance_one_timestep(self, dt, hospital_action=None):
        if hospital_action is None:
            hospital_action = np.zeros(self.num_hospital, dtype=float)

        # 2026-09-28: timestep 시작 시 종료자를 먼저 해제한 뒤, non-active 후보 중 신규 cancer patient를 지정한다.
        self.expire_active_cancerpatients(dt)
        self.designate_new_cancerpatient(dt)

        # Information provision
        if self.parameter_set.prov_type > 0:
            self.Information_Provision_Policy(dt)

            for idx_person in range(len(self.arr_Person)):
                self.arr_population_PHQ[idx_person] = self.arr_Person[idx_person].PHQ
                self.arr_population_PRQ[idx_person] = self.arr_Person[idx_person].PRQ

        
        self.update_WOM_all(dt)
        
        self.calculate_choice_prob_all()  # [2026-09-07 calculate choice probability]: person loop 없이 모든 choice probability를 한 번에 계산


        # 2026-10-01: 실제 병원 선택량 기반 VOR effect + 병원 action에 따른 effect
        # self.calculate_hospital_volume(dt)
        self.VOR_effect(self.parameter_set.Q_vmax, dt)
        self.investment_effect(hospital_action, dt)
        self.update_total_OHQ()
        # self.scale_effect(dt) # 추후 hospital action에 따른 scale effect 

        self.update_Archive(dt)



class City:
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set
        self.idx = parameter_set.idx_city
        self.nm_region = parameter_set.nm_region
        self.nm = parameter_set.nm_city
        
        self.num_population_real = parameter_set.population_real
        self.num_population = parameter_set.num_population
        self.num_cancer_susceptible = self.num_population

        self.num_hospital = parameter_set.num_hospital

        self.coord_centroid = Point(0,0)

        self.arr_idx_person_incity = np.array([])
        self.arr_idx_hospital_incity = np.array([])

class Hospital:
    def __init__(self, parameter_set):
        self.parameter_set = parameter_set
        self.idx_city = -1
        self.nm_city = ""
        self.nm_hospital = ""

        self.OQ_obj_quality = 0
        
        self.coord = parameter_set.coord

class Person:
    def __init__(self, parameter_set, parameter_set_simulation):
        self.parameter_set = parameter_set
        self.parameter_set_simulation = parameter_set_simulation
        self.df_Hospital_info = parameter_set.df_Hospital_info
        self.df_Person_info = parameter_set.df_Person_info
        self.idx_hospital_in_region = parameter_set.idx_hospital_in_region
        self.arr_hospital_region = parameter_set.arr_hospital_region
        self.idx_representative_hospital_by_region = parameter_set.idx_representative_hospital_by_region

        # Personal info
        self.idx_person = parameter_set.idx_person
        self.is_cancerpatient = parameter_set.is_cancerpatient
        self.is_new_cancerpatient = parameter_set.is_new_cancerpatient
        self.idx_city = parameter_set.idx_city
        self.nm_city = parameter_set.nm_city
        self.coord = parameter_set.coord
        self.pi = self.parameter_set_simulation.pi

        # PHQ, PRQ
        self.dict_parameter_initial_PRQ = parameter_set.dict_parameter_initial_PRQ
        self.PHQ = np.array([])
        self.PRQ = np.array([])
        self.sigma_OHQ = parameter_set.sigma_OHQ
        self.sigma_PRQ = parameter_set.sigma_PRQ
        
        self.arr_population_PHQ = parameter_set.arr_population_PHQ
        self.arr_population_PRQ = parameter_set.arr_population_PRQ
        self.arr_population_choice_prob = parameter_set.arr_population_choice_prob
        self.arr_population_CancerOrNot = parameter_set.arr_population_CancerOrNot

        self.max_quality = self.parameter_set_simulation.max_quality
        self.min_quality = self.parameter_set_simulation.min_quality

        # Distance effects
        self.arr_distance_to_hospital = np.array([])
        self.arr_distance_effect = np.array([])

        # Hospital Choice Coefficients
        self.beta_PQ = parameter_set.beta_PQ
        self.beta_d = parameter_set.beta_d
        self.delta = parameter_set.delta

        # Utility, Choice
        # self.Utility = np.zeros(len(self.df_Hospital_info))

        # [2026-09-07] recognition optimization
        self.arr_OQ = self.df_Hospital_info['OQ_objective_quality'].values

        self.arr_utility_region = np.zeros(len(self.df_Hospital_info))
        self.arr_utility_hospital = np.zeros(len(self.df_Hospital_info))

        self.choice_prob = np.zeros(len(self.df_Hospital_info))
        self.idx_choice_hospital = -1

        # WOM Target
        self.list_idx_neighbor = []
        
        # WOM : Hospital Quality
        self.Q_h = np.full((self.parameter_set_simulation.max_dt,
                            len(self.df_Hospital_info)), -1.0)
        self.C_h = np.full((self.parameter_set_simulation.max_dt,
                            len(self.df_Hospital_info)), -1.0)
        self.S_h = np.full((self.parameter_set_simulation.max_dt,
                            len(self.df_Hospital_info)), -1.0)
        self.E_h = np.full((self.parameter_set_simulation.max_dt,
                            len(self.df_Hospital_info)), -1.0)
            # noise
        self.sigma_wom = self.parameter_set_simulation.sigma_wom
            # C : Direct WOM from Cancer Patient
        self.mu_hc = self.parameter_set_simulation.mu_hc
        self.lambda_c = self.parameter_set_simulation.lambda_c
            # S : Direct WOM from Susceptible
        self.mu_hs = self.parameter_set_simulation.mu_hs
        self.lambda_s = self.parameter_set_simulation.lambda_s
            # E : Experience based Realization
        self.mu_recog = self.parameter_set_simulation.mu_recog
        self.sigma_recog = self.parameter_set_simulation.sigma_recog
        
        
        # WOM : Regional Quality
        self.Q_r = np.full((self.parameter_set_simulation.max_dt,
                            len(self.parameter_set_simulation.dict_parameter_initial_PRQ)), -1.0)
        self.C_r = np.full((self.parameter_set_simulation.max_dt,
                            len(self.parameter_set_simulation.dict_parameter_initial_PRQ)), -1.0)
        self.S_r = np.full((self.parameter_set_simulation.max_dt,
                            len(self.parameter_set_simulation.dict_parameter_initial_PRQ)), -1.0)
        self.N_r = np.full((self.parameter_set_simulation.max_dt,
                            len(self.parameter_set_simulation.dict_parameter_initial_PRQ)), -1.0)
        self.U_r = np.full((self.parameter_set_simulation.max_dt,
                            len(self.parameter_set_simulation.dict_parameter_initial_PRQ)), -1.0)
        self.E_r = np.full((self.parameter_set_simulation.max_dt,
                            len(self.parameter_set_simulation.dict_parameter_initial_PRQ)), -1.0)
        

            # C : Direct WOM from Cancer Patient
        self.mu_rc = self.parameter_set_simulation.mu_rc
            # S : Indirect WOM from Susceptible
        self.mu_rs = self.parameter_set_simulation.mu_rs
            # N : Indirect WOM from Region
        self.mu_rn = self.parameter_set_simulation.mu_rn

    # [2026-09-07 calculate choice probability]: Simulation-level choice probability 행을 Person API와 호환되도록 연결
    @property
    def choice_prob(self):
        return self.arr_population_choice_prob[self.idx_person]

    @choice_prob.setter
    def choice_prob(self, value):
        self.arr_population_choice_prob[self.idx_person] = value

    def calculate_utility(self):
        self.arr_utility_region = np.zeros(len(self.df_Hospital_info))
        self.arr_utility_hospital = np.zeros(len(self.df_Hospital_info))

        # PRQ
        arr_PRQ_hospital = self.PRQ[self.arr_hospital_region]  # [2026-09-07 calculate choice probability]: precomputed hospital-region index 사용
        self.arr_utility_region += arr_PRQ_hospital * self.beta_PQ
        # PRQ_distance
        self.arr_utility_region += self.arr_distance_effect_region * self.beta_d
        # print(self.arr_utility_region)

        # PHQ
        self.arr_utility_hospital += self.PHQ * self.beta_PQ
        self.arr_utility_hospital += self.arr_distance_effect_hospital * self.beta_d
        self.arr_utility_region[np.where(self.arr_utility_region == np.isnan)[0]] = 0.0

    # 전체 인구의 선택 확률은 simulation-level에서 계산
    # 신규 암환자의 병원 선택 확률 및 그에 따른 병원 선택은 person-level에서 계산
    def calculate_choice_prob(self):
        self.calculate_utility()

        if self.pi != 0:    # 2-stage choice model
            # Regional Choice Probability
            exp_u_region = np.exp(self.arr_utility_region)
            # regional utility exp sum으로 나눠주기
            self.choice_prob_region = exp_u_region / np.sum(exp_u_region[self.idx_representative_hospital_by_region])

            # Hospital Choice Probability
            exp_u_hospital = np.exp(self.arr_utility_hospital * self.pi)    # scale parameter (pi)
            sum_u_temp = np.bincount(
                self.arr_hospital_region, weights=exp_u_hospital,
                minlength=len(self.idx_hospital_in_region)
            )  # [2026-09-07 calculate choice probability]: one-hot matrix 없이 region별 exponential utility 합산
            self.choice_prob_hospital = exp_u_hospital / sum_u_temp[self.arr_hospital_region]

            self.choice_prob = self.choice_prob_region * self.choice_prob_hospital


            if np.any(np.isnan(exp_u_hospital)) or np.any(np.isinf(exp_u_hospital)):
                print("Error: Overflow encountered. Logging values for debugging:")
                print("idx_person:", self.idx_person)
                print("PHQ:", self.PHQ)
                print("arr_utility_hospital:", self.arr_utility_hospital)
                print("Computed exponent:", self.arr_utility_hospital / self.pi)
                print("Resulting exp values:", exp_u_hospital)

        else:   # for pi==0, we employed logit model between all the hospitals, without consideration to PRQ, just for explorative experiments
            exp_u_hospital = np.exp(self.arr_utility_hospital)
            exp_u_hospital[np.where(exp_u_hospital==np.inf)[0]] = max(exp_u_hospital[np.where(exp_u_hospital!=np.inf)[0]])
            self.choice_prob_hospital = exp_u_hospital / np.sum(exp_u_hospital)
            self.choice_prob = self.choice_prob_hospital
    
    def choose_hospital(self):
        self.idx_choice_hospital = np.random.choice(len(self.choice_prob), p=self.choice_prob)
        self.idx_region_choice_hospital = self.df_Hospital_info.iloc[self.idx_choice_hospital]['idx_REGION']
        self.df_Person_info.loc[self.df_Person_info.idx_person == self.idx_person, 'idx_chosen_hospital'] = self.idx_choice_hospital
        self.df_Person_info.loc[self.df_Person_info.idx_person == self.idx_person, 'idx_region_chosen_hospital'] = self.idx_region_choice_hospital
           
