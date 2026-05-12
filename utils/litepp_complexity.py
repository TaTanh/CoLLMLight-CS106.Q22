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

    def compute_nc(self, observation):
        """
        Compute congested lanes across local and neighboring intersections.
        """
        nc = 0
        local_lanes = observation.get("current_observation", {}).get("local_lanes", {})
        for lane, state in local_lanes.items():
            if self.is_critical_lane(state):
                nc += 1
                
        # If neighbor observation supports it, we evaluate it as well.
        # But for Lite++, local is typically the most direct measurement of congestion "score" 
        # unless specifically instructed to include neighbors to sum nc
        # Will just return local sum to adhere strictly to simple nc
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