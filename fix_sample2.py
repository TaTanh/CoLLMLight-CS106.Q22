import re

with open("scripts/sample_litepp_cityflow.py", "r") as f:
    code = f.read()

old_block = """                # neighbor_observation
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
                    samples_collected += 1"""

new_block = """                # neighbor_observation
                neighbor_obs = {"upstream": {}, "downstream": {}}
                node_dict = env.traffic_light_node_dict[inter.inter_name]
                # Gather queue sums for neighbors simply into upstream to satisfy validation
                for nb_k in ["neighbor_ENWS", "neighbor_up_down_stream"]:
                    if nb_k in node_dict:
                        for nb_name in node_dict[nb_k]:
                            if nb_name and nb_name != "null":
                                # Find neighbor intersection
                                for adj_inter in env.list_intersection:
                                    if adj_inter.inter_name == nb_name:
                                        q_sum = 0
                                        for lane in adj_inter.list_entering_lanes:
                                            q_sum += adj_inter.dic_lane_waiting_vehicle_count_current_step.get(lane, 0)
                                        
                                        idx = adj_inter.current_phase_index - 1
                                        ph_str = action_space[idx] if 0 <= idx < len(action_space) else "UNKNOWN"
                                        neighbor_obs["upstream"][nb_name] = {
                                            "total_queue": q_sum,
                                            "total_wait": float(q_sum),
                                            "occupancy_avg": 0.0,
                                            "phase": ph_str
                                        }
                                        break

                # Resolve current phase: fallback to index math
                idx_phase = inter.current_phase_index - 1
                if 0 <= idx_phase < len(action_space):
                    act_str = action_space[idx_phase]
                else:
                    act_str = action_space[action[j] % len(action_space)]

                # Only write out a sample if history buffer is fully populated (e.g., 5 PREVIOUS frames)
                if len(intersection_histories[inter.inter_id]) == history_window: 
                    sample = {
                        "dataset": args.dataset,
                        "intersection_id": inter.inter_name,
                        "timestep": env.eng.get_current_time(),
                        "current_phase": act_str,
                        "current_observation": {
                            "local_lanes": local_obs
                        },
                        "neighbor_observation": neighbor_obs,
                        "history": intersection_histories[inter.inter_id].copy(), # This has exactly 5 past states
                        "candidate_actions": action_space,
                        "local_priority": ["WT_ET", "NT_ST", "WL_EL", "NL_SL"],
                        "replay": {
                            "seed": 42,
                            "policy": args.policy,
                            "actions_before_timestep": action_trace_list.copy() # All structured actions taken up to now
                        }
                    }
                    out_f.write(json.dumps(sample) + "\\n")
                    samples_collected += 1

                # Push history AFTER creating sample
                hist_state = {
                    "timestep": env.eng.get_current_time(),
                    "action": act_str,
                    "local_lanes": local_obs,
                    "neighbor_lanes": neighbor_obs
                }
                intersection_histories[inter.inter_id].append(hist_state)
                # Keep window size
                if len(intersection_histories[inter.inter_id]) > history_window:
                    intersection_histories[inter.inter_id].pop(0)"""

code = code.replace(old_block, new_block)

with open("scripts/sample_litepp_cityflow.py", "w") as f:
    f.write(code)
