# AUDIT_LITEPP

## 1. Important Folders & Files
- `utils/cityflow_env.py`: Manages the CityFlow simulation, getting states (`get_state`, `get_lane_vehicles`, `get_lane_waiting_vehicle_count`), executing actions (`set_signal`), resetting environment.
- `utils/pipeline.py` & `framework/CoLLMlight.py` & `framework/FTSample.py`: Main classes organizing environments and invoking models. `FTSample` contains extensive evaluations like `calc_avg_travel_time`.
- `data/`: Contains datasets including `Hangzhou/4_4`, `Jinan/3_4`, `Synthetic/4_4`.
- `config/ppo_config.yaml` & `utils/config.py`: Stores configuration mappings, agent setup params, phase lists (`WT_ET`, `NT_ST`, `WL_EL`, `NL_SL`).
- `run_CoLLMlight.py` / `run_fts.py`: Examples of how environments and arguments are set up to run the simulation loop.

## 2. Environment Creation
`CityFlowEnv` is built via `cityflow.Engine(config_file, thread_num=...ration)`. The configuration connects to `roadnet.json` and `flow.json` in the respective `data/` subfolders (`Synthetic`, `Jinan`, `Hangzhou`).
The agent setup loads properties such as list of lanes, list of phases initialized in `utils/config.py`. 

## 3. Retrieving Observations
In `utils/cityflow_env.py`, `CityFlowEnv.get_state()` triggers `inter.get_state()`.
`eng.get_lane_vehicles()`, `eng.get_lane_waiting_vehicle_count()`, `eng.get_vehicle_speed()`, and `eng.get_vehicle_distance()` provide the needed fine-grained queue length, speed, and positional occupancy. 
Occupancy, waiting time, and queue length per lane are obtained (or easily computed) from this system state dict.

## 4. Setting Action & Signals
Signal phases are listed in `utils/config.py` under `"PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL']` and `"PHASE_MAP"`.
`inter.set_signal(action, action_pattern="set", ...)` correctly sets the traffic light phase on the respective intersection using `eng.set_tl_phase()` based on index (1, 2, 3, 4 map to 0-3 candidate actions).

## 5. Evaluating ATT / AWT
Average Travel Time (ATT) is calculated using `calc_avg_travel_time` inside `framework/FTSample.py`, computing diff between leave and enter times for vehicles. 
Average Queue Length (AQL) or Waiting Time (AWT) are evaluated through inner measurements mapped across timesteps in `framework/FTSample.py`.

## 6. Reusable Functions / Classes
- `CityFlowEnv` inside `utils/cityflow_env.py` (direct env integration).
- Phase Mappings and Configurations from `utils/config.py`.
- Metrics computations (ATT/AWT) from `framework/FTSample.py` and `utils/utils.py`. `calc_avg_travel_time()` is a primary target to reuse for output evaluation.
- OpenAI-compatible LLM wrappers if any exist in `utils/LLMs.py` (Else I will provide a new one).

## 7. Plan for Phase 1 & 2
**Phase 1 (Config):**
Write `config/collmlight_litepp.yaml` containing settings for train/eval datasets, limits, API baselines mapping the user instructions meticulously. Disable New York.

**Phase 2 (Sampling):**
Write `scripts/sample_litepp_cityflow.py` importing `CityFlowEnv`. Initialize it exactly like in `run_fts.py` pointing to Synth dataset.
Step through simulation to `num_samples` limits, capturing `lane_waiting_vehicle_count` and parsing into the designated `observation` JSON format. Will save out a `.jsonl` directly.

> **Dependencies Note:** Since New York 28x7 is disabled as requested, I will explicitly avoid hardcoding paths for it. Phase 4 will use CityFlow resetting / replays if clone states aren't natively supported.