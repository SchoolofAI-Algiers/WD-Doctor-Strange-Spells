import math

import pytest
from src.motion.velocity_tracker import VelocityTracker


def test_first_frame_has_zero_velocity_no_previous_to_compare():
    tracker = VelocityTracker()

    result = tracker.update(hand_id=1, position_px=(100.0, 100.0), dt=1 / 30)

    assert result.velocity_px_s == (0.0, 0.0)
    assert result.speed_px_s == 0.0
    assert result.acceleration_px_s2 == (0.0, 0.0)


def test_stationary_hand_has_zero_speed():
    """Property: a hand that doesn't move should report speed ~= 0."""
    tracker = VelocityTracker()
    tracker.update(1, (100.0, 100.0), dt=1 / 30)

    # feed the exact same position several frames in a row
    result = None
    for _ in range(5):
        result = tracker.update(1, (100.0, 100.0), dt=1 / 30)

    assert result.speed_px_s == pytest.approx(0.0, abs=1e-6)


def test_known_horizontal_movement_gives_expected_velocity():
    """Hand moves 30px to the right in exactly 1 second -> 30 px/s."""
    tracker = VelocityTracker(velocity_ema_alpha=1.0)  # alpha=1 disables smoothing
    tracker.update(1, (0.0, 0.0), dt=1.0)

    result = tracker.update(1, (30.0, 0.0), dt=1.0)

    assert result.velocity_px_s[0] == pytest.approx(30.0)
    assert result.velocity_px_s[1] == pytest.approx(0.0)
    assert result.speed_px_s == pytest.approx(30.0)
    assert result.direction_rad == pytest.approx(0.0)  # pointing along +x


def test_direction_points_straight_down_for_vertical_movement():
    tracker = VelocityTracker(velocity_ema_alpha=1.0)
    tracker.update(1, (0.0, 0.0), dt=1.0)

    result = tracker.update(1, (0.0, 50.0), dt=1.0)

    assert result.direction_rad == pytest.approx(math.pi / 2)


def test_constant_velocity_gives_zero_acceleration():
    """Property: moving at a constant speed -> acceleration ~= 0."""
    tracker = VelocityTracker(velocity_ema_alpha=1.0)
    tracker.update(1, (0.0, 0.0), dt=1.0)
    tracker.update(1, (10.0, 0.0), dt=1.0)  # 10 px/s

    result = tracker.update(1, (20.0, 0.0), dt=1.0)  # still 10 px/s

    assert result.acceleration_px_s2[0] == pytest.approx(0.0, abs=1e-6)


def test_speeding_up_gives_positive_acceleration():
    tracker = VelocityTracker(velocity_ema_alpha=1.0)
    tracker.update(1, (0.0, 0.0), dt=1.0)
    tracker.update(1, (10.0, 0.0), dt=1.0)  # v = 10 px/s

    result = tracker.update(1, (30.0, 0.0), dt=1.0)  # v = 20 px/s -> accel = +10

    assert result.acceleration_px_s2[0] == pytest.approx(10.0)


def test_smoothing_blends_new_and_previous_reading():
    """With alpha=0.5, a sudden jump should only move the result halfway."""
    tracker = VelocityTracker(velocity_ema_alpha=0.5)
    tracker.update(1, (0.0, 0.0), dt=1.0)
    tracker.update(1, (10.0, 0.0), dt=1.0)  # raw v=10, smoothed: 0.5*10+0.5*0=5

    result = tracker.update(1, (20.0, 0.0), dt=1.0)
    # raw v = 10 again, smoothed = 0.5*10 + 0.5*5 = 7.5, not 10
    assert result.velocity_px_s[0] == pytest.approx(7.5)


def test_two_hands_do_not_share_history():
    tracker = VelocityTracker(velocity_ema_alpha=1.0)
    tracker.update(hand_id=1, position_px=(0.0, 0.0), dt=1.0)
    tracker.update(hand_id=2, position_px=(0.0, 0.0), dt=1.0)

    result_1 = tracker.update(hand_id=1, position_px=(100.0, 0.0), dt=1.0)
    result_2 = tracker.update(hand_id=2, position_px=(5.0, 0.0), dt=1.0)

    assert result_1.velocity_px_s[0] == pytest.approx(100.0)
    assert result_2.velocity_px_s[0] == pytest.approx(5.0)


def test_clear_removes_a_hands_history():
    tracker = VelocityTracker()
    tracker.update(1, (50.0, 50.0), dt=1.0)

    tracker.clear(1)
    result = tracker.update(1, (999.0, 999.0), dt=1.0)

    # treated as brand new, so velocity is 0 again, not a huge jump
    assert result.velocity_px_s == (0.0, 0.0)
