import pytest
import numpy as np
from types import SimpleNamespace
from dataclasses import fields
from src.geometry.calculator import norm2px
from src.geometry.calculator import palm_center
from src.geometry.calculator import landmark_normalization
from src.geometry.calculator import GeometryCalculator, GeometryConfig, HandGeometry






@pytest.mark.parametrize(
    "L, w, h, expected",
    [
        (np.array([[0,1],[0.5,0.2]]), 200, 100, np.array([[0,100],[100,20]])),
        (np.array([[0.8,0.3],[0.1,0.5],[0,0]]), 100, 100, np.array([[80,30],[10,50],[0,0]])),
        (np.array([[1,0],[0,1],[1,1],[0,0]]), 100, 200, np.array([[100,0],[0,200],[100,200],[0,0]])),
        (np.array([[0.45,0.7],[0.45,0.7]]), 1, 1, np.array([[0.45,0.7],[0.45,0.7]])),
        (np.array([[0.5,0.5],[0.2,1],[0.75,0.9]]), 0, 0, np.array([[0,0],[0,0],[0,0]])),
        (np.array([[0.5,0.5,0.3],[0.1,0.2,-0.4]]), 200, 100, np.array([[100,50],[20,20]]))
   
    ],
         ids=["higher_width", "square_image", "higher_height", "unit_scale", "zero_dims", "xyz_input"]
)

def test_norm2px(L,w,h,expected):

   result = norm2px(L,w,h)
   np.testing.assert_allclose(result, expected)


def test_norm2px_xyz_output_shape():
    L = np.random.default_rng(0).random((21, 3))
    assert norm2px(L, 640, 480).shape == (21, 2)


def test_norm2px_z_has_no_effect():
    L = np.random.default_rng(0).random((21, 3))
    L_other_z = L.copy()
    L_other_z[:, 2] += 5
    np.testing.assert_allclose(norm2px(L, 640, 480), norm2px(L_other_z, 640, 480))








@pytest.mark.parametrize(
    "L, hand_size, expected",
    [
        (np.array([[35,120],[10,0.5],[0,0]]), 0, np.array([[0,0],[0,0],[0,0]])),
        (np.array([[0,0],[50,20],[80,45]]), 10, np.array([[0,0],[5,2],[8,4.5]])),
        (np.array([[60,15],[45,100],[150,10]]), 20, np.array([[0,0],[-0.75,4.25],[4.5,-0.25]])),
        (np.array([[35,120],[10,0.5],[0,0]]), 1e-9, np.array([[0,0],[0,0],[0,0]])),
        (np.array([[35,120],[10,0.5],[0,0]]), 1e-6, np.array([[0,0],[0,0],[0,0]])),
   
    ],
         ids = ["zero_hand_size", "simple", "wristHigher_or_moreLeft", "tiny_hand_size", "epsilon_boundary"]
)

def test_landmark_normalization(L,hand_size,expected):

   result = landmark_normalization(L,hand_size)
   np.testing.assert_allclose(result, expected)


def test_landmark_normalization_translation_invariant():
    L = np.array([[0,0], [50,20], [80,45]])
    L_translated = L + np.array([100, 50])
    hand_size = 10
    
    result_original = landmark_normalization(L, hand_size)
    result_translated = landmark_normalization(L_translated, hand_size)
    
    np.testing.assert_allclose(result_original, result_translated)








def make_landmarks():
    L = np.zeros((21, 2))
    L[0] = [20, 30]
    L[5] = [100, 40]
    L[9] = [45, 45]
    L[13] = [5, 20]
    L[17] = [10, 10]
    return L

@pytest.mark.parametrize(
    "strategy, expected",
    [
        ('five_point', [36, 29]),
        ('wrist_only', [20, 30]),
        ('mcp_only', [40, 28.75]),
        ('weighted_mcp', [37.5, 31.6666666667]),
    ],
    ids=["five_point", "wrist_only", "mcp_only", "weighted_mcp"]
)
def test_palm_center(strategy, expected):
    L = make_landmarks()
    result = palm_center(L, strategy)
    np.testing.assert_allclose(result, expected)

def test_palm_center_invalid_strategy():
    L = make_landmarks()
    with pytest.raises(ValueError):
        palm_center(L, 'not_a_real_strategy')


from src.geometry.calculator import distance, hand_size, palm_orientation
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
#-------------------------------------------------------------------------------------------------------------------------------------------------chihab is goat----------------------
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






IMG_W, IMG_H = 200, 100


def make_hand_xyz():
    L = np.zeros((21, 3))
    L[0] = [0.10, 0.50, 0.20]
    L[5] = [0.50, 0.60, 0.00]
    L[8] = [0.10, 0.80, -0.05]
    L[9] = [0.30, 0.40, 0.00]
    L[12] = [0.25, 0.90, 0.10]
    L[13] = [0.10, 0.20, 0.00]
    L[17] = [0.20, 0.10, 0.00]
    return SimpleNamespace(landmarks_norm=L)


def test_compute_default_config_with_xyz_landmarks():
    result = GeometryCalculator(IMG_W, IMG_H).compute(make_hand_xyz())

    assert isinstance(result, HandGeometry)
    assert result.palm_center_px == pytest.approx((48, 36))
    assert result.palm_center_norm == pytest.approx((0.24, 0.36))
    assert result.hand_size_px == pytest.approx(50)
    assert result.hand_size_norm == pytest.approx(0.25)
    assert result.palm_angle_rad == pytest.approx(np.arctan2(20, 40))
    assert result.landmarks_norm.shape == (21, 2)
    np.testing.assert_allclose(result.landmarks_norm[0], [0, 0])
    np.testing.assert_allclose(result.landmarks_norm[12], [0.6, 0.8])


def test_compute_returns_all_six_fields():
    result = GeometryCalculator(IMG_W, IMG_H).compute(make_hand_xyz())
    assert {f.name for f in fields(result)} == {
        "palm_center_px",
        "palm_center_norm",
        "hand_size_px",
        "hand_size_norm",
        "palm_angle_rad",
        "landmarks_norm",
    }


def test_compute_respects_custom_config():
    cfg = GeometryConfig(palm_center_strategy="wrist_only", hand_size_method="wrist_index_tip")
    result = GeometryCalculator(IMG_W, IMG_H, cfg).compute(make_hand_xyz())

    assert result.palm_center_px == pytest.approx((20, 50))
    assert result.palm_center_norm == pytest.approx((0.1, 0.5))
    assert result.hand_size_px == pytest.approx(30)
    assert result.hand_size_norm == pytest.approx(0.15)
    np.testing.assert_allclose(result.landmarks_norm[8], [0, 1])


def test_compute_xyz_and_xy_inputs_give_same_result():
    hand_xyz = make_hand_xyz()
    hand_xy = SimpleNamespace(landmarks_norm=hand_xyz.landmarks_norm[:, :2])
    calc = GeometryCalculator(IMG_W, IMG_H)
    a, b = calc.compute(hand_xyz), calc.compute(hand_xy)

    assert a.palm_center_px == pytest.approx(b.palm_center_px)
    assert a.hand_size_px == pytest.approx(b.hand_size_px)
    assert a.palm_angle_rad == pytest.approx(b.palm_angle_rad)
    np.testing.assert_allclose(a.landmarks_norm, b.landmarks_norm)


def test_compute_degenerate_hand_gives_zero_landmarks():
    hand = SimpleNamespace(landmarks_norm=np.full((21, 3), 0.5))
    result = GeometryCalculator(IMG_W, IMG_H).compute(hand)
    assert result.hand_size_px == 0.0
    np.testing.assert_allclose(result.landmarks_norm, np.zeros((21, 2)))


@pytest.mark.parametrize(
    "cfg",
    [
        GeometryConfig(palm_center_strategy="nope"),
        GeometryConfig(hand_size_method="nope"),
    ],
    ids=["bad_palm_strategy", "bad_hand_size_method"],
)
def test_compute_invalid_config_raises(cfg):
    with pytest.raises(ValueError):
        GeometryCalculator(IMG_W, IMG_H, cfg).compute(make_hand_xyz())