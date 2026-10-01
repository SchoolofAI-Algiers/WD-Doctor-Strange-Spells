import pytest
import numpy as np
from src.geometry.calculator import norm2px
from src.geometry.calculator import palm_center
from src.geometry.calculator import landmark_normalization






@pytest.mark.parametrize(
    "L, w, h, expected",
    [
        (np.array([[0,1],[0.5,0.2]]), 200, 100, np.array([[0,100],[100,20]])),
        (np.array([[0.8,0.3],[0.1,0.5],[0,0]]), 100, 100, np.array([[80,30],[10,50],[0,0]])),
        (np.array([[1,0],[0,1],[1,1],[0,0]]), 100, 200, np.array([[100,0],[0,200],[100,200],[0,0]])),
        (np.array([[0.45,0.7],[0.45,0.7]]), 1, 1, np.array([[0.45,0.7],[0.45,0.7]])),
        (np.array([[0.5,0.5],[0.2,1],[0.75,0.9]]), 0, 0, np.array([[0,0],[0,0],[0,0]]))
   
    ],
         ids=["higher_width", "square_image", "higher_height", "unit_scale", "zero_dims"]
)

def test_norm2px(L,w,h,expected):

   result = norm2px(L,w,h)
   np.testing.assert_allclose(result, expected)








@pytest.mark.parametrize(
    "L, hand_size, expected",
    [
        (np.array([[35,120],[10,0.5],[0,0]]), 0, np.array([[0,0],[0,0],[0,0]])),
        (np.array([[0,0],[50,20],[80,45]]), 10, np.array([[0,0],[5,2],[8,4.5]])),
        (np.array([[60,15],[45,100],[150,10]]), 20, np.array([[0,0],[-0.75,4.25],[4.5,-0.25]])),
   
    ],
         ids = ["zero_hand_size", "simple", "wristHigher_or_moreLeft"]
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