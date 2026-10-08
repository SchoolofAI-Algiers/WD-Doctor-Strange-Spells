from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from src.geometry.calculator import FloatArray, HandGeometry, distance

class Gesture(Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    POINTING = "pointing"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    TWO_HANDS_TOGETHER = "two_hands_together"

@dataclass
class GestureState:
    gesture: Gesture
    confidence: float
    frames_held: int
    hand_id: int







@dataclass(frozen=True)
class StabilityConfig:
    enter_frames:int = 3
    confirm_frames:int = 5
    exit_frames:int = 3
    ema_alpha:float = 0.3







@dataclass(frozen=True)
class GestureConfig:
   min_hand_size_norm: float= 0.05 #so we don't clsasify a non existing hand as a closed_fist
   thumb_threshold: float= 1.3
   finger_threshold:float= 1.5
   together_max_ratio: float =1.0
   stability: StabilityConfig= field(default_factory=StabilityConfig)




@dataclass(frozen=True)
class FingerSpec:
   name: str
   tip: int
   pip: int
   mcp: int


FINGERS=(
FingerSpec('thumb', tip=4, pip=3, mcp=2),
FingerSpec('index', tip=8, pip=6, mcp=5),
FingerSpec('middle', tip=12, pip=10, mcp=9),
FingerSpec('ring', tip=16,pip=14, mcp=13),
FingerSpec('pinky',tip=20, pip=18, mcp=17),
)




def is_finger_extended(landmarks_norm: FloatArray, finger:FingerSpec, threshold: float) -> bool:
   tip_to_mcp= distance(landmarks_norm[finger.tip], landmarks_norm[finger.mcp])
   pip_to_mcp= distance(landmarks_norm[finger.pip], landmarks_norm[finger.mcp])

   return tip_to_mcp > threshold*pip_to_mcp



def extended_fingers(landmarks_norm:FloatArray, config:GestureConfig)-> frozenset[str]:
   return frozenset(
      finger.name for finger in FINGERS if is_finger_extended(landmarks_norm, finger, config.thumb_threshold if finger.name=='thumb' else config.finger_threshold)
   )












def classify_hand(geometry: HandGeometry, config: GestureConfig | None= None) -> Gesture:
   config= config or GestureConfig()

   if geometry.hand_size_norm < config.min_hand_size_norm:
      return Gesture.NONE

   fingers= extended_fingers(geometry.landmarks_norm, config)

   if not fingers: 
      return Gesture.CLOSED_FIST
   if len(fingers)== len(FINGERS):
      return Gesture.OPEN_PALM
   if fingers == frozenset({'index'}):
      return Gesture.POINTING
   if fingers == frozenset({'index', 'middle'}):
      return Gesture.PEACE
   if fingers == frozenset({'thumb'}):
      return Gesture.THUMBS_UP

   return Gesture.NONE











#Two hands gesture
def hands_together(geometries: list[HandGeometry], config: GestureConfig | None=None)-> bool:
    config= config or GestureConfig()
    if len(geometries) != 2:
       return False
       
    a, b =geometries
    mean_size =(a.hand_size_px + b.hand_size_px)/2
    if mean_size<1e-6:
        return False
        
    d= distance(np.array(a.palm_center_px), np.array(b.palm_center_px))
    return d/ mean_size<config.together_max_ratio
    
    









#The shield of seraphim
def is_shield(geometries: list[HandGeometry], config: GestureConfig | None =None) -> bool:
   if len(geometries) !=2: #so that we have exactly 2 hands
      return False

   gestures = [classify_hand(geometry, config) for geometry in geometries]
   return gestures[0] == gestures[1] and gestures[0] in (
      Gesture.CLOSED_FIST,
      Gesture.OPEN_PALM,
   )










#State machine
class  PoseState(Enum):
    
    IDLE= 'idle'
    CANDIDATE= 'candidate'
    ACTIVE= 'active'

@dataclass(frozen=True)
class PoseStatus:
    state: PoseState
    confidence:float
    frames_held:int
    






class PoseStabilizer:
    def __init__(self, config: StabilityConfig | None= None) ->None:
        self.config= config or StabilityConfig()
        self._state= PoseState.IDLE
        self._seen_streak= 0
        self._lost_streak= 0
        self._frames_held= 0
        self._confidence= 0.0
        
        
    def update(self, pose_seen:bool)-> PoseStatus:
        cfg= self.config
        observation= 1.0 if pose_seen else 0.0
        self._confidence= cfg.ema_alpha* observation +  (1 - cfg.ema_alpha)* self._confidence
        
        if self._state is PoseState.ACTIVE:
            self._frames_held +=1
            if pose_seen:
                self._lost_streak= 0
            else:
                self._lost_streak +=1
                if self._lost_streak>= cfg.exit_frames:
                    self._go_idle()
        
        elif pose_seen:
            self._seen_streak += 1
            if self._seen_streak>= cfg.enter_frames + cfg.confirm_frames:
                self._state= PoseState.ACTIVE
                self._lost_streak=0
                self._frames_held= 1
            elif self._seen_streak>= cfg.enter_frames:
                self._state=PoseState.CANDIDATE
                
        else:
            self._go_idle()
            
        return PoseStatus(self._state, self._confidence, self._frames_held)
        
        
        
        
    
    def reset(self)-> None:
        self._go_idle()
        self._confidence= 0.0
        
    def _go_idle(self)-> None:
        self._state=PoseState.IDLE
        self._seen_streak= 0
        self._lost_streak= 0
        self._frames_held= 0
                
        







class GestureRecognizer:
    def __init__(self, config: GestureConfig | None=None)-> None:
        self.config= config or GestureConfig()
        self._stabilizers: dict[int, dict[Gesture, PoseStabilizer]] = {}
    
    def update(self, geometries: list[HandGeometry]) -> list[GestureState]:
        for lost_id in [h for h in self._stabilizers if h >= len(geometries)]:
            del self._stabilizers[lost_id]

        together = hands_together(geometries, self.config)

        states: list[GestureState] = []
        for hand_id, geometry in enumerate(geometries):
            raw = Gesture.TWO_HANDS_TOGETHER if together else classify_hand(geometry, self.config)
            states.append(self._update_hand(hand_id, raw))
        return states
    
    def reset(self)-> None:
        self._stabilizers.clear()
        

    def _update_hand(self, hand_id: int, raw: Gesture) -> GestureState:
        stabilizers = self._stabilizers.setdefault(hand_id, {})

        best: tuple[Gesture, PoseStatus] | None = None
        for gesture in Gesture:
            if gesture is Gesture.NONE:
                continue
            if gesture not in stabilizers:
                stabilizers[gesture] = PoseStabilizer(self.config.stability)
            status = stabilizers[gesture].update(raw is gesture)
            if status.state is PoseState.ACTIVE and (
                best is None or status.confidence > best[1].confidence
            ):
                best = (gesture, status)

        if best is None:
            return GestureState(Gesture.NONE, 0.0, 0, hand_id)
        gesture, status = best
        return GestureState(gesture, status.confidence, status.frames_held, hand_id)


class ShieldDetector:
    
    def __init__(self, config:GestureConfig | None=None)-> None:
        self.config= config or GestureConfig()
        self._stabilizer= PoseStabilizer(self.config.stability)
    
    def update(self, geometries: list[HandGeometry]) -> PoseStatus:
        return self._stabilizer.update(is_shield(geometries, self.config))
    
    def reset(self)->None:
        self._stabilizer.reset()