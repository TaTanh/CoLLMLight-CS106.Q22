class ComplexityAnalyzer:
    def __init__(self, thresholds):
        """
        thresholds: dict mapped from config
        e.g., {'occupancy_threshold': 0.5, 'queue_threshold': 15, 'wait_threshold': 60.0}
        """
        self.occ_thresh = thresholds.get('occupancy_threshold', 0.5)
        self.q_thresh = thresholds.get('queue_threshold', 15)
        self.w_thresh = thresholds.get('wait_threshold', 60.0)

    def is_critical_lane(self, lane_state):
        """
        lane_state: dict containing 'queue', 'wait_time', 'occupancy'
        """
        q = lane_state.get('queue', 0)
        w = lane_state.get('wait_time', 0.0)
        occ = lane_state.get('occupancy', 0.0)

        if occ >= self.occ_thresh or q >= self.q_thresh or w >= self.w_thresh:
            return True
        return False

    def is_critical_neighbor(self, neighbor_state):
        """Check if a neighbor intersection is critical based on occupancy."""
        occ = neighbor_state.get('occupancy_avg', 0.0)
        return occ >= self.occ_thresh

    def compute_nc(self, observation):
        """
        Compute number of critical NEIGHBORING lanes (not local).
        nc = count of neighboring intersections with congestion.
        """
        nc = 0
        neighbor_obs = observation.get("neighbor_observation", {})

        # Count critical neighbors in both upstream and downstream
        for direction in ["upstream", "downstream"]:
            for inter_id, neighbor_state in neighbor_obs.get(direction, {}).items():
                if self.is_critical_neighbor(neighbor_state):
                    nc += 1

        return nc

    def classify_complexity(self, nc):
        if nc == 0:
            return "NO"
        elif nc == 1:
            return "Simple"
        else:
            return "Complex"

    def attach_complexity(self, sample):
        """
        Attaches the complexity fields to a given sample dict in-place.
        """
        nc = self.compute_nc(sample)
        label = self.classify_complexity(nc)
        sample["complexity"] = {
            "nc": nc,
            "label": label
        }
        return sample