import re

with open("scripts/sample_litepp_cityflow.py", "r") as f:
    code = f.read()

# Replace the simulation loop
old_loop = """    action = 0 # initially start with phase 0 (index 1)
    
    # Trace for replay
    action_trace = {}
    action_trace_list = []

    for i in range(args.simulation_time):
        if samples_collected >= args.num_samples:
            break

        # Generate dummy actions to step the environment uniformly.
        # Action is an index 0-3
        action_list = [action] * num_intersections
        action_trace_list.append(action)
        
        # Simulate 1 step
        env.step(action_list)
        
        # We only want to sample at decision steps (e.g. every 30s) or at some interval.
        # But we also need history. Let's say decision step is every 30 steps.
        if i % 30 == 0:
            if args.policy == "random":
                action = np.random.randint(0, 4)
            for inter in env.list_intersection:
                inter.set_signal(action, "set", yellow_time=5, path_to_log=work_dir)
            
            action_trace[i] = action

            for inter in env.list_intersection:"""

new_loop = """    action = [0] * num_intersections # initially start with phase 0 (index 1)
    
    # Trace for replay
    action_trace_list = []

    for i in range(args.simulation_time):
        if samples_collected >= args.num_samples:
            break

        # Generate dummy actions to step the environment uniformly.
        # Action is an index 0-3
        action_trace_list.append({
            "timestep": i,
            "actions": {env.list_intersection[j].inter_name: action[j] for j in range(num_intersections)}
        })
        
        # Simulate 1 step
        env.step(action)
        
        # We only want to sample at decision steps (e.g. every 30s) or at some interval.
        if i > 50 and i % 30 == 0:
            for j, inter in enumerate(env.list_intersection):"""

code = code.replace(old_loop, new_loop)

# Replace the inner parts
old_inner = """                # neighbor_observation (Approximation)
                neighbor_obs = {"upstream": {}, "downstream": {}}

                # Record sample
                sample = {
                    "dataset": args.dataset,
                    "intersection_id": inter.inter_name,
                    "timestep": env.eng.get_current_time(),
                    "current_phase": action_space[action % len(action_space)],
                    "current_observation": {
                        "local_lanes": local_obs
                    },
                    "neighbor_observation": neighbor_obs,
                    "history": intersection_histories[inter.inter_id][:history_window],
                    "candidate_actions": action_space,
                    "local_priority": ["WT_ET", "NT_ST", "WL_EL", "NL_SL"],
                    "replay": {
                        "seed": 42,
                        "policy": args.policy,
                        "actions_before_timestep": action_trace_list[:-1] # All actions taken before this timestep
                    }
                }

                # Push history
                hist_state = {
                    "timestep": env.eng.get_current_time(),
                    "action": action_space[action % len(action_space)],
                    "local_lanes": local_obs,
                    "neighbor_lanes": neighbor_obs
                }
                intersection_histories[inter.inter_id].append(hist_state)
                # Keep window size
                if len(intersection_histories[inter.inter_id]) > history_window:
                    intersection_histories[inter.inter_id].pop(0)
                
                # Only write out a sample if we actually have some traffic or after warm-up > 30 skip
                if i > 60: 
                    out_f.write(json.dumps(sample) + "\n")
                    samples_collected += 1"""

new_inner = """                # neighbor_observation
                neighbor_obs = {}
                node_dict = env.traffic_light_node_dict[inter.inter_name]
                # Gather queue sums for neighbors simply
                for nb_k in ["neighbor_ENWS", "neighbor_up_down_stream"]:
                    if nb_k in node_dict:
                        for nb_name in node_dict[nb_k]:
                            if nb_name and nb_name != "null":
                                # Find neighbor intersection
                                for adj_inter in env.list_intersection:
                                    if adj_inter.inter_name == nb_name:
                                        # just report sum queue to not clutter space during smoke test
                                        q_sum = 0
                                        for lane in adj_inter.list_entering_lanes:
                                            q_sum += adj_inter.dic_lane_waiting_vehicle_count_current_step.get(lane, 0)
                                        neighbor_obs[nb_name] = {"total_queue": q_sum}
                                        break

                # Resolve current phase: fallback to index math
                idx_phase = inter.current_phase_index - 1
                if 0 <= idx_phase < len(action_space):
                    act_str = action_space[idx_phase]
                else:
                    act_str = action_space[action[j] % len(action_space)]

                # Record sample
                sample = {
                    "dataset": args.dataset,
                    "intersection_id": inter.inter_name,
                    "timestep": env.eng.get_current_time(),
                    "current_phase": act_str,
                    "current_observation": {
                        "local_lanes": local_obs
                    },
                    "neighbor_observation": neighbor_obs,
                    "history": intersection_histories[inter.inter_id][:history_window],
                    "candidate_actions": action_space,
                    "local_priority": ["WT_ET", "NT_ST", "WL_EL", "NL_SL"],
                    "replay": {
                        "seed": 42,
                        "policy": args.policy,
                        "actions_before_timestep": action_trace_list.copy() # All actions taken before this timestep
                    }
                }

                # Push history
                hist_state = {
                    "timestep": env.eng.get_current_time(),
                    "action": act_str,
                    "local_lanes": local_obs,
                    "neighbor_lanes": neighbor_obs
                }
                intersection_histories[inter.inter_id].append(hist_state)
                # Keep window size
                if len(intersection_histories[inter.inter_id]) > history_window:
                    intersection_histories[inter.inter_id].pop(0)
                
                # Only write out a sample if history buffer is fully populated (e.g., 5 frames)
                if len(intersection_histories[inter.inter_id]) == history_window: 
                    out_f.write(json.dumps(sample) + "\\n")
                    samples_collected += 1

            # Decide next actions AFTER sampling state
            if args.policy == "random":
                action = [np.random.randint(0, 4) for _ in range(num_intersections)]
            
            for j, inter in enumerate(env.list_intersection):
                inter.set_signal(action[j], "set", yellow_time=5, path_to_log=work_dir)"""

code = code.replace(old_inner, new_inner)

with open("scripts/sample_litepp_cityflow.py", "w") as f:
    f.write(code)
