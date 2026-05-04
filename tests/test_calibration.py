"""Tests for the core calibration module."""

import pytest
import numpy as np


class TestCalibratorBase:
    """Test adaptive threshold and geometric mean."""

    def test_geometric_mean(self):
        from equimem.calibration.base import CalibratorBase
        assert abs(CalibratorBase.geometric_mean(0.64, 0.36) - 0.48) < 1e-6
        assert CalibratorBase.geometric_mean(0.0, 1.0) == 0.0
        assert CalibratorBase.geometric_mean(1.0, 1.0) == 1.0

    def test_threshold_cold_start(self):
        from equimem.calibration.base import CalibratorBase

        class DummyCalibrator(CalibratorBase):
            def score(self, *args, **kwargs):
                pass

        cal = DummyCalibrator()
        assert cal.threshold == 0.5  # cold-start default

    def test_threshold_adapts(self):
        from equimem.calibration.base import CalibratorBase

        class DummyCalibrator(CalibratorBase):
            def score(self, *args, **kwargs):
                pass

        cal = DummyCalibrator()
        for v in [0.3, 0.5, 0.7, 0.4, 0.6]:
            cal._update_history(v)
        assert abs(cal.threshold - 0.5) < 1e-6  # median of [0.3,0.4,0.5,0.6,0.7]


class TestCredibility:
    """Test bidirectional credibility weight."""

    def test_decay_on_invalid(self):
        from equimem.calibration.credibility import CredibilityTracker
        ct = CredibilityTracker()
        ct._round_total_count = 10
        ct._round_invalid_count = 2  # delta = 0.2
        ct.update("agent_1", valid=False)
        assert ct.get_weight("agent_1") < 1.0

    def test_recovery_on_valid(self):
        from equimem.calibration.credibility import CredibilityTracker
        ct = CredibilityTracker()
        ct._weights["agent_1"] = 0.5
        ct._round_total_count = 10
        ct._round_invalid_count = 2
        ct.update("agent_1", valid=True)
        assert ct.get_weight("agent_1") > 0.5

    def test_steady_state(self):
        from equimem.calibration.credibility import CredibilityTracker
        ct = CredibilityTracker()
        # Simulate agent with 30% invalid rate
        for _ in range(200):
            ct._round_total_count = 100
            ct._round_invalid_count = 30
            if np.random.random() < 0.3:
                ct.update("agent_1", valid=False)
            else:
                ct.update("agent_1", valid=True)
        # Should converge to w* ≈ 1 - p = 0.7
        assert abs(ct.get_weight("agent_1") - 0.7) < 0.15


class TestRBO:
    """Test rank-biased overlap distance."""

    def test_identical_lists(self):
        from equimem.utils.rbo import rbo_distance
        assert rbo_distance([1, 2, 3], [1, 2, 3]) < 0.05

    def test_disjoint_lists(self):
        from equimem.utils.rbo import rbo_distance
        assert rbo_distance([1, 2, 3], [4, 5, 6]) > 0.9

    def test_empty_lists(self):
        from equimem.utils.rbo import rbo_distance
        assert rbo_distance([], []) == 0.0
        assert rbo_distance([1], []) == 1.0


class TestTrustDiscount:
    """Test trust-discounted retrieval."""

    def test_embedding_discount(self):
        from equimem.retrieval.trust_discount import trust_discount_embedding
        vec = np.array([1.0, 0.0, 0.0])
        discounted = trust_discount_embedding(vec, rho=0.25)
        assert np.allclose(discounted, [0.5, 0.0, 0.0])

    def test_path_discount(self):
        from equimem.retrieval.trust_discount import trust_discount_path
        assert abs(trust_discount_path([0.8, 0.9, 0.7]) - 0.504) < 1e-3
        assert trust_discount_path([0.8, 0.0, 0.7]) == 0.0
