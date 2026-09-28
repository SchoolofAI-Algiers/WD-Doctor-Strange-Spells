from src.geometry.geometry_calculatorV2 import distance, hand_size, palm_orientation
import numpy as np
import pytest

def test_distance_3_4_5():
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    result = distance(p1, p2)
    assert result == pytest.approx(5.0)
def test_distance_same_point_is_zero():
    p = np.array([2.0, 7.0])
    assert distance(p, p) == 0.0
def test_distance_is_symmetric():
    a = np.array([1.0, 2.0])
    b = np.array([4.0, 6.0])
    assert distance(a, b) == pytest.approx(distance(b, a))
def test_distance_negative_coordinates():
    p1 = np.array([-1, -1])
    p2 = np.array([2, 3])
    assert distance(p1, p2) == pytest.approx(5.0)
#-------------------------------------------------------------------------------------------------chihab is goat----------------------
def make_zero_landmarks():
    return np.zeros((21, 2))
def test_hand_size_wrist_middle_tip():
    lm = make_zero_landmarks()
    lm[0] = [0, 0]     # wrist
    lm[12] = [3, 4]    # middle fingertip
    assert hand_size(lm, "wrist_middle_tip") == pytest.approx(5.0)

def test_hand_size_wrist_index_tip():
    lm = make_zero_landmarks()
    lm[8] = [0, 10]    # index fingertip
    assert hand_size(lm, "wrist_index_tip") == pytest.approx(10.0)



def test_hand_size_avg_fingers():
    lm = make_zero_landmarks()
    lm[8] = [2, 0]
    lm[12] = [4, 0]
    lm[16] = [6, 0]
    lm[20] = [8, 0]
    assert hand_size(lm, "avg_fingers") == pytest.approx(5.0)   # (2+4+6+8)/4


def test_hand_size_default_is_wrist_middle_tip():
    lm = make_zero_landmarks()
    lm[12] = [3, 4]
    assert hand_size(lm) == pytest.approx(5.0)


def test_hand_size_wrist_not_at_origin():
    lm = make_zero_landmarks()
    lm[0] = [10, 10]
    lm[12] = [13, 14]
    assert hand_size(lm, "wrist_middle_tip") == pytest.approx(5.0)

@pytest.mark.parametrize("lm5, expected", [
    ([1, 0],  0.0),          # pointing right
    ([0, 1],  np.pi / 2),    # pointing down (image y goes down)
    ([-1, 0], np.pi),        # pointing left
    ([0, -1], -np.pi / 2),   # pointing up
])
def test_palm_orientation(lm5, expected):
    lm = np.zeros((21, 2))
    lm[9] = [0, 0]
    lm[5] = lm5
    assert palm_orientation(lm) == pytest.approx(expected)