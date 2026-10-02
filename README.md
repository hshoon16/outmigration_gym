# Outmigration ABM with Gymnasium Interface

지역 간 의료이용(outmigration)을 대상으로 객관적 병원 품질(OHQ)과 개인·지역 인식 품질(PHQ/PRQ)의 상호작용을 모의하는 agent-based model입니다. 병원 투자 action을 다루기 위한 Gymnasium 환경의 초기 구현도 포함합니다.

> Last updated: 2026-10-02

## Project layout

| File / folder | Purpose |
| --- | --- |
| `outmigration_core.py` | 현재 ABM의 기준 구현입니다. 데이터 로딩, 초기화, timestep 전이, WOM, VOR, 투자 효과를 포함합니다. |
| `hospital_ABM_gym.py` | `ABMenv_gym` Gymnasium wrapper입니다. 병원 전체의 joint action을 하나의 환경 action으로 받습니다. |
| `outmigration_runner.ipynb` | 데이터 로딩, simulation 실행, 결과 확인을 위한 notebook입니다. |
| `ABM_city_info.xlsx` | 도시별 인구, 병원 수, 지역 정보입니다. |
| `ABM_hospital_crd.xlsx` | 병원 좌표와 병원 정보입니다. |
| `Korea_shapefiles/` | 개인 위치 생성과 지리 연산에 사용하는 행정구역 shapefile입니다. |
| `outside_polygon_mask.npy` | 공간 계산에 사용하는 보조 mask입니다. |
| `outputs/` | 실행 결과를 저장하는 폴더입니다. |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux

```bash
source .venv/bin/activate
```


```text
ABM_city_info.xlsx
ABM_hospital_crd.xlsx
outside_polygon_mask.npy
Korea_shapefiles/TL_SCCO_CTPRVN.shp
```

Python에서 직접 실행할 수도 있습니다.

```python
import numpy as np
import random
import outmigration_core as core

core.load_geographic_context(data_dir=".", geography_dir=".")
df_city_info, df_hospital_crd = core.load_simulation_inputs(data_dir=".")

parameter_set = core.make_default_parameter_set(df_city_info, df_hospital_crd)
parameter_set.max_dt = 100

np.random.seed(42)
random.seed(4242)

sim = core.Simulation(parameter_set)
sim.initialize()       # 도시·병원·개인·neighbor 구조 생성
sim.prepare_episode()  # OHQ/PHQ/PRQ 및 archive 초기화
sim.run_simulation()
```

한 timestep만 진행하려면 다음을 사용합니다.

```python
sim.advance_one_timestep(dt=0)
```

## Model flow

각 timestep의 핵심 흐름은 다음과 같습니다.

```text
active cancer 종료 처리
  → 신규 cancer patient 지정 및 병원 선택
  → Information Provision Policy (선택 사항)
  → synchronous WOM 및 recognition
  → 병원 선택확률 재계산
  → VOR 및 투자 효과를 통한 OHQ 갱신
  → archive 저장
```

### Quality components

- **OHQ**: 병원의 객관적 품질입니다. 현재 total OHQ는 `base_OHQ + vor_OHQ + action_OHQ`를 quality 범위로 clip하여 계산합니다.
- **PHQ**: 개인별 병원 인식 품질입니다. shape은 `(population, hospitals)`입니다.
- **PRQ**: 개인별 지역 인식 품질입니다. shape은 `(population, regions)`입니다.
- **VOR**: 과거 병원 평균 선택확률 history를 gamma weight로 집계하는 volume-outcome relationship 효과입니다.
- **Investment effect**: 병원 action의 최근 history를 inverse-gamma lag weight로 집계해 `action_OHQ`를 계산합니다.

### Cancer-patient lifecycle

`is_cancerpatient`는 평생 상태가 아니라 현재 치료 중인 active cancer 상태입니다.

- `cancer_duration`의 현재 기본값은 10 timestep입니다.
- 신규 환자는 지정 timestep부터 `cancer_duration` 동안 active입니다.
- 종료 timestep 시작 시 non-cancer 상태로 전환되며, 이후 신규 환자 후보에 다시 포함됩니다.
- 치료 종료 후에도 PHQ/PRQ 및 과거 선택 병원 정보는 유지됩니다.

### WOM and recognition

WOM은 Simulation-level NumPy 연산으로 hospital/region 효과를 계산하고, PHQ와 PRQ를 synchronous하게 갱신합니다. 

## Gymnasium environment

`hospital_ABM_gym.py`의 `ABMenv_gym`은 전체 병원의 binary joint action을 받습니다.

```python
from hospital_ABM_gym import ABMenv_gym

env = ABMenv_gym(data_dir=".", geography_dir=".", max_dt=100)
observation, info = env.reset(seed=42)

joint_action = env.action_space.sample()  # shape: (num_hospitals,)
observation, reward, terminated, truncated, info = env.step(joint_action)
```

- Action space: `MultiBinary(num_hospitals)`
- Observation: 병원별 `ohq`, 직전 timestep 신규 환자 수 `demand`
- `info`: 병원별 신규 환자 수, 개별 보상, 누적 개별 보상을 포함합니다.


## Main parameters

`make_default_parameter_set()`에서 기본값을 확인·변경할 수 있습니다.

| Group | Main parameters |
| --- | --- |
| Simulation | `max_dt`, `n_p`, `cancer_duration` |
| Choice | `beta_PQ`, `beta_d`, `delta`, `pi` |
| Recognition | `mu_recog`, `sigma_recog` |
| WOM | `mu_hc`, `mu_hs`, `mu_rc`, `mu_rs`, `lambda_c`, `lambda_s`, `sigma_wom` |
| VOR | `vor_K`, `gamma_alpha`, `gamma_beta`, `Q_vmax` |
| Investment | `action_effect_K`, `action_inv_gamma_alpha`, `action_inv_gamma_beta`, `Q_amax` |

## Reproducibility

실행 전 NumPy와 Python `random` seed를 함께 지정하세요. 지리 기반 개인 위치 생성에도 Python `random`을 사용합니다.

```python
np.random.seed(42)
random.seed(4242)
```
