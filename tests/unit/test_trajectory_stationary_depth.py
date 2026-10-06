#python -m pytest tests/unit/test_trajectory_stationary_depth.py -v

import importlib
import random


def _load_module():
    """Find trajectory_stationary_depth whichever way the project is set up.

    If none of these match your project, change the list (or replace this whole
    function with a normal `from ... import ...` line).
    """
    names = (
        "motion.trajectory_stationary_depth",       # src/ is the import root
        "src.motion.trajectory_stationary_depth",   # the repo root is the import root
        "trajectory_stationary_depth",              # test file sits next to the module
    )
    last_error = None
    for name in names:
        try:
            return importlib.import_module(name)
        except ImportError as error:
            last_error = error
    raise ImportError(f"could not import trajectory_stationary_depth, tried {names}") from last_error


_m = _load_module()
TrajectoryBuffer = _m.TrajectoryBuffer
StationaryDetector = _m.StationaryDetector
DepthTrend = _m.DepthTrend
DepthTrendDetector = _m.DepthTrendDetector
FingertipTrail = _m.FingertipTrail


# ---------------------------------------------------------------
# TrajectoryBuffer
# ---------------------------------------------------------------
def test_buffer_starts_empty():
    b = TrajectoryBuffer()
    assert len(b) == 0
    assert b.points() == []


def test_buffer_points_drop_the_timestamp_and_keep_oldest_first():
    b = TrajectoryBuffer()
    b.append(10, 20, 0.0)
    b.append(30, 40, 0.033)
    b.append(50, 60, 0.066)
    assert b.points() == [(10, 20), (30, 40), (50, 60)]
    assert len(b) == 3


def test_buffer_keeps_only_the_last_n_points():
    b = TrajectoryBuffer(maxlen=5)
    for i in range(8):
        b.append(i * 10, 0, i * 0.033)
    assert len(b) == 5
    assert b.points() == [(30, 0), (40, 0), (50, 0), (60, 0), (70, 0)]


def test_buffer_default_size_is_60():
    b = TrajectoryBuffer()
    for i in range(100):
        b.append(i, 0, i * 0.033)
    assert len(b) == 60
    assert b.points()[0] == (40, 0) and b.points()[-1] == (99, 0)


def test_buffer_clear_empties_it_and_it_can_be_used_again():
    b = TrajectoryBuffer()
    for i in range(10):
        b.append(i, i, i)
    b.clear()
    assert len(b) == 0 and b.points() == []
    b.append(1, 2, 0.0)
    assert b.points() == [(1, 2)]


def test_buffer_points_returns_a_copy():
    b = TrajectoryBuffer()
    b.append(1, 1, 0.0)
    pts = b.points()
    pts.append((99, 99))
    assert b.points() == [(1, 1)]          # changing the returned list does not touch the buffer


# ---------------------------------------------------------------
# StationaryDetector
# ---------------------------------------------------------------
def test_stationary_starts_as_moving():
    assert StationaryDetector().stationary is False


def test_stationary_needs_min_frames_slow_frames_in_a_row():
    d = StationaryDetector()                 # min_frames = 5
    for _ in range(4):
        assert d.update(5, 200) is False     # 0.025 hand-sizes/s: slow, but not long enough
    assert d.update(5, 200) is True          # 5th slow frame in a row


def test_stationary_speed_is_judged_relative_to_hand_size():
    # The same 15 px/s: slow for a big hand (0.075), not slow for a small hand (0.15).
    big, small = StationaryDetector(), StationaryDetector()
    for _ in range(10):
        big.update(15, 200)
        small.update(15, 100)
    assert big.stationary is True
    assert small.stationary is False


def test_stationary_a_frame_between_the_thresholds_resets_the_streak_while_moving():
    d = StationaryDetector()
    for _ in range(4):
        d.update(5, 200)
    d.update(30, 200)                        # 0.15: between enter (0.10) and exit (0.25)
    for _ in range(4):
        assert d.update(5, 200) is False     # the streak started again from 0
    assert d.update(5, 200) is True


def test_stationary_a_fast_frame_resets_the_streak():
    d = StationaryDetector()
    for _ in range(4):
        d.update(5, 200)
    d.update(100, 200)                       # 0.5: fast
    for _ in range(4):
        assert d.update(5, 200) is False
    assert d.update(5, 200) is True


def test_stationary_hysteresis_keeps_the_state_between_the_thresholds():
    d = StationaryDetector()
    for _ in range(5):
        d.update(5, 200)
    assert d.stationary
    for speed in (25, 30, 40, 49):           # 0.125 to 0.245 hand-sizes/s: in between
        assert d.update(speed, 200) is True


def test_stationary_leaves_the_state_only_above_the_exit_threshold():
    d = StationaryDetector()
    for _ in range(5):
        d.update(5, 200)
    assert d.update(60, 200) is False        # 0.30 > 0.25
    for _ in range(4):                       # and it needs 5 slow frames again
        assert d.update(5, 200) is False
    assert d.update(5, 200) is True


def test_stationary_thresholds_are_strict():
    d = StationaryDetector()
    for _ in range(10):
        d.update(20, 200)                    # exactly 0.10: not below enter_thresh
    assert d.stationary is False
    for _ in range(5):
        d.update(5, 200)
    assert d.update(50, 200) is True         # exactly 0.25: not above exit_thresh


def test_stationary_bad_hand_size_keeps_the_previous_answer():
    d = StationaryDetector()
    assert d.update(5, 0) is False
    assert d.update(5, -10) is False
    for _ in range(5):
        d.update(5, 200)
    assert d.update(999, 0) is True          # a broken measurement cannot flip the state
    assert d.update(999, -1) is True


def test_stationary_reset_goes_back_to_moving_and_forgets_the_streak():
    d = StationaryDetector()
    for _ in range(5):
        d.update(5, 200)
    d.reset()
    assert d.stationary is False
    for _ in range(4):
        assert d.update(5, 200) is False
    assert d.update(5, 200) is True


def test_stationary_custom_settings_are_used():
    d = StationaryDetector(enter_thresh=0.5, exit_thresh=1.0, min_frames=2)
    d.update(50, 200)                        # 0.25 < 0.5
    assert d.update(50, 200) is True         # only 2 frames needed
    assert d.update(150, 200) is True        # 0.75: still below exit_thresh = 1.0
    assert d.update(250, 200) is False       # 1.25 > 1.0


# ---------------------------------------------------------------
# DepthTrend
# ---------------------------------------------------------------
def test_depth_trend_has_three_distinct_values():
    assert {DepthTrend.NONE, DepthTrend.TOWARD, DepthTrend.AWAY} == set(DepthTrend)
    assert len(set(DepthTrend)) == 3


# ---------------------------------------------------------------
# DepthTrendDetector
# ---------------------------------------------------------------
def test_depth_says_none_until_there_is_enough_history():
    d, size = DepthTrendDetector(), 100.0    # window = 5, so 6 values are needed
    for _ in range(5):
        size *= 1.10                         # a very fast approach
        assert d.update(size) == DepthTrend.NONE
    size *= 1.10
    assert d.update(size) == DepthTrend.TOWARD


def test_depth_hand_getting_bigger_is_toward():
    d, size, r = DepthTrendDetector(), 100.0, None
    for _ in range(10):
        size *= 1.03
        r = d.update(size)
    assert r == DepthTrend.TOWARD


def test_depth_hand_getting_smaller_is_away():
    d, size, r = DepthTrendDetector(), 300.0, None
    for _ in range(10):
        size *= 0.97
        r = d.update(size)
    assert r == DepthTrend.AWAY


def test_depth_constant_size_is_none():
    d = DepthTrendDetector()
    for _ in range(50):
        assert d.update(200.0) == DepthTrend.NONE


def test_depth_small_noise_on_a_still_hand_is_ignored():
    random.seed(0)
    d = DepthTrendDetector()
    for _ in range(300):
        assert d.update(200 + random.uniform(-2, 2)) == DepthTrend.NONE


def test_depth_one_small_spike_is_smoothed_away():
    d = DepthTrendDetector()
    for _ in range(20):
        d.update(200.0)
    results = [d.update(208.0)]              # +4% for ONE frame
    results += [d.update(200.0) for _ in range(10)]
    assert DepthTrend.TOWARD not in results
    assert DepthTrend.AWAY not in results


def test_depth_very_slow_drift_is_ignored_but_a_steady_approach_is_not():
    slow, steady = DepthTrendDetector(), DepthTrendDetector()
    a = b = 200.0
    for _ in range(30):
        a *= 1.003
        b *= 1.01
        r_slow, r_steady = slow.update(a), steady.update(b)
    assert r_slow == DepthTrend.NONE         # 0.3% per frame
    assert r_steady == DepthTrend.TOWARD     # 1% per frame


def test_depth_more_smoothing_reacts_later():
    fast, slow = DepthTrendDetector(alpha=1.0), DepthTrendDetector(alpha=0.1)
    size = 100.0
    for _ in range(6):
        r_fast, r_slow = fast.update(size), slow.update(size)
        size *= 1.01
    assert r_fast == DepthTrend.TOWARD       # no smoothing: sees the full 5% growth
    assert r_slow == DepthTrend.NONE         # heavy smoothing: still catching up


def test_depth_bad_measurements_are_ignored_and_do_not_change_the_state():
    clean, dirty = DepthTrendDetector(), DepthTrendDetector()
    size = 100.0
    for _ in range(12):
        size *= 1.03
        expected = clean.update(size)
        assert dirty.update(0) == DepthTrend.NONE
        assert dirty.update(-5) == DepthTrend.NONE
        assert dirty.update(size) == expected


def test_depth_first_frame_uses_the_size_as_it_is():
    d = DepthTrendDetector(alpha=0.1)        # heavy smoothing must not drag the first value
    d.update(200.0)
    for _ in range(5):
        r = d.update(200.0)
    assert r == DepthTrend.NONE              # it would not be none if it had started from 0


def test_depth_reset_forgets_everything():
    d, size = DepthTrendDetector(), 100.0
    for _ in range(10):
        size *= 1.05
        d.update(size)
    d.reset()
    for _ in range(5):
        assert d.update(500.0) == DepthTrend.NONE    # history is empty again
    assert d.update(500.0) == DepthTrend.NONE        # and 500 is taken as the first size


def test_depth_spec_literal_rule_with_window_1_and_no_smoothing():
    d = DepthTrendDetector(alpha=1.0, window=1)      # the literal spec rule
    assert d.update(100) == DepthTrend.NONE
    assert d.update(103) == DepthTrend.TOWARD        # 1.03 > 1.02
    assert d.update(104) == DepthTrend.NONE          # 1.0097
    assert d.update(101) == DepthTrend.AWAY          # 0.971 < 0.98
    assert d.update(100) == DepthTrend.NONE          # 0.990


# ---------------------------------------------------------------
# FingertipTrail
# ---------------------------------------------------------------
def test_trail_is_empty_when_not_pointing():
    f = FingertipTrail()
    assert f.update((10, 10), 0.0, False) == []
    assert f.update((12, 12), 0.033, False) == []


def test_trail_records_the_tip_in_order_while_pointing():
    f = FingertipTrail()
    f.update((10, 20), 0.0, True)
    f.update((12, 22), 0.033, True)
    assert f.update((15, 25), 0.066, True) == [(10, 20), (12, 22), (15, 25)]


def test_trail_missing_tip_is_not_recorded():
    f = FingertipTrail()
    f.update((1, 1), 0.0, True)
    assert f.update(None, 0.033, True) == [(1, 1)]


def test_trail_keeps_the_drawing_through_a_pause_of_up_to_grace_frames():
    f = FingertipTrail()                     # grace_frames = 3
    f.update((1, 1), 0.0, True)
    for _ in range(3):
        f.update(None, 0.1, False)           # exactly 3 frames without pointing
    assert f.update((2, 2), 0.2, True) == [(1, 1), (2, 2)]


def test_trail_a_longer_pause_starts_a_new_drawing():
    f = FingertipTrail()
    f.update((1, 1), 0.0, True)
    f.update((2, 2), 0.033, True)
    for _ in range(4):
        f.update(None, 0.1, False)           # 4 frames: more than grace_frames
    assert f.update((9, 9), 0.2, True) == [(9, 9)]


def test_trail_last_drawing_stays_after_pointing_stops():
    f = FingertipTrail()
    f.update((1, 1), 0.0, True)
    f.update((2, 2), 0.033, True)
    for _ in range(20):
        assert f.update(None, 0.1, False) == [(1, 1), (2, 2)]


def test_trail_is_a_ring_buffer_of_60_points():
    f = FingertipTrail()
    for i in range(100):
        pts = f.update((i, 0), i * 0.033, True)
    assert len(pts) == 60
    assert pts[0] == (40, 0) and pts[-1] == (99, 0)


def test_trail_custom_size_and_grace_frames():
    f = FingertipTrail(maxlen=5, grace_frames=0)
    for i in range(8):
        pts = f.update((i, 0), i * 0.033, True)
    assert pts == [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0)]
    f.update(None, 1.0, False)               # one missed frame is already a new drawing
    assert f.update((50, 50), 1.1, True) == [(50, 50)]


def test_trail_reset_clears_the_drawing():
    f = FingertipTrail()
    f.update((1, 1), 0.0, True)
    f.reset()
    assert f.update(None, 0.1, False) == []
    assert f.update((5, 5), 0.2, True) == [(5, 5)]


if __name__ == "__main__":
    tests = [(n, fn) for n, fn in sorted(globals().items()) if n.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print("ok  ", name)
    print(f"\nAll {len(tests)} tests passed.")