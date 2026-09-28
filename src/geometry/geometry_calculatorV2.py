import numpy as np
from dataclasses import dataclass



def distance(p1,p2)-> float:
   d = np.sqrt(np.sum((p2-p1)**2))   
   return d                             




def norm2px(landmarks_norm, image_width, image_height)-> np.ndarray:

   resolution = np.array([image_width,image_height])
   landmarks_px = landmarks_norm*resolution
   
   return landmarks_px       





def palmCenter(landmarks_px, palm_center_strategy='five_point')-> np.ndarray:
   if palm_center_strategy == 'five_point':
      palm_indices = [0, 5, 9, 13, 17]
      palm_center = np.mean(landmarks_px[palm_indices], axis=0)

   elif palm_center_strategy == 'wrist_only':
      palm_center = landmarks_px[0]

   elif palm_center_strategy == 'mcp_only':
      palm_indices = [5, 9, 13, 17]
      palm_center = np.mean(landmarks_px[palm_indices], axis=0)

   elif palm_center_strategy == 'weighted_mcp':
      palm_indices = [0, 5, 9, 13, 17]
      palm_center = np.average(landmarks_px[palm_indices], weights=[1,1,2,1,1], axis=0)
      

   return palm_center      




def hand_size(landmarks_px, hand_size_method = 'wrist_middle_tip')-> float:
   if hand_size_method == 'wrist_middle_tip':
      hand_size_px = distance(landmarks_px[0], landmarks_px[12])

   elif hand_size_method == 'wrist_index_tip':
      hand_size_px = distance(landmarks_px[0], landmarks_px[8])

   elif hand_size_method == 'avg_fingers':
      distances = [distance(landmarks_px[0],landmarks_px[12]), distance(landmarks_px[0],landmarks_px[8]), distance(landmarks_px[0],landmarks_px[16]), distance(landmarks_px[0],landmarks_px[20])]
      hand_size_px = np.mean(distances)

   return hand_size_px        




def palm_orientation(landmarks_px)-> float:
   v = landmarks_px[5] - landmarks_px[9]
   palm_angle_rad = np.arctan2(v[1], v[0])

   return palm_angle_rad




def landmark_normalization(landmarks_px, hand_size_px)-> np.ndarray:
   if hand_size_px >= 1 :
      centered = landmarks_px - landmarks_px[0]
      normalized = centered / hand_size_px

      return normalized
   else:
      return np.zeros_like(landmarks_px)



@dataclass
class GeometryConfig:
    palm_center_strategy: str = "five_point"
    hand_size_method: str = "wrist_middle_tip"




@dataclass(frozen=True)
class HandGeometry:
    palm_center_px: tuple[float, float]
    palm_center_norm: tuple[float, float]
    hand_size_px: float
    hand_size_norm: float
    palm_angle_rad: float
    landmarks_norm: np.ndarray

class GeometryCalculator:
    def __init__(self, image_width: int, image_height: int,config: GeometryConfig | None = None):
        self.image_width=image_width
        self.image_height=image_height
        self.config = config or GeometryConfig()
    
    def compute(self, hand: HandLandmarks) -> HandGeometry:
        landmarks_norm = hand.landmarks_norm
        landmarks_px = norm2px(landmarks_norm, self.image_width, self.image_height)
        palm_center_px = palmCenter(landmarks_px, self.config.palm_center_strategy)
        palm_center_norm = palm_center_px/[self.image_width, self.image_height]
        hand_size_px = hand_size(landmarks_px, self.config.hand_size_method)
        hand_size_norm = hand_size_px / max(self.image_width, self.image_height)
        palm_angle_rad = palm_orientation(landmarks_px)
        landmarks_centered= landmark_normalization(landmarks_px, hand_size_px)

        return HandGeometry(
        palm_center_px = tuple(palm_center_px),
        palm_center_norm = tuple(palm_center_norm),
        hand_size_px = hand_size_px,
        hand_size_norm = hand_size_norm,
        palm_angle_rad = palm_angle_rad,
        landmarks_norm = landmarks_centered,
)
