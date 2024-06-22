import numpy as np
import cv2
from collections import deque
import mediapipe as mp
from utils.utils_v2 import get_idx_to_coordinates, rescale_frame

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def recognize_hand_sign(thumb_tip, index_tip, middle_tip, pinky_tip, wrist_center):
  """
  Analyzes fingertip positions and angles to recognize hand signs.

  Args:
      thumb_tip: Tuple containing (x, y) coordinates of thumb tip.
      index_tip: Tuple containing (x, y) coordinates of index finger tip.
      middle_tip: Tuple containing (x, y) coordinates of middle finger tip.
      pinky_tip: Tuple containing (x, y) coordinates of pinky finger tip.
      wrist_center: Tuple containing (x, y) coordinates of wrist center (replace with actual calculation if needed).

  Returns:
      Integer representing the recognized hand sign (0-8) or -1 if not recognized.
  """

  # Calculate distances between fingertips
  thumb_index_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
  thumb_middle_dist = np.linalg.norm(np.array(thumb_tip) - np.array(middle_tip))

  # Calculate vectors for fingers relative to wrist
  thumb_vec = np.array(thumb_tip) - np.array(wrist_center)
  index_vec = np.array(index_tip) - np.array(wrist_center)
  middle_vec = np.array(middle_tip) - np.array(wrist_center)
  pinky_vec = np.array(pinky_tip) - np.array(wrist_center)

  # Calculate angles between fingers (acos handles radian values)
  thumb_index_angle = np.arccos(np.dot(thumb_vec, index_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec)))
  thumb_middle_angle = np.arccos(np.dot(thumb_vec, middle_vec) / (np.linalg.norm(thumb_vec) * np.linalg.norm(middle_vec)))

  # Decision logic based on finger positions and angles (adjust thresholds as needed)
  if all(tip is not None for tip in [thumb_tip, index_tip, middle_tip]) and thumb_index_dist < 50:
      # Check if fingers are close together (fist)
      if all(angle < 0.5 for angle in [thumb_index_angle, thumb_middle_angle]):
          # Front fist (fingers curled inwards)
          return 0
      else:
          # Back fist (fingers curled outwards)
          return 1
  elif all(tip is not None for tip in [index_tip, middle_tip, pinky_tip]) and thumb_tip[1] < wrist_center[1]:  # Check if thumb is below wrist
      # Fingers likely extended (open palm or peace sign)
      if thumb_index_angle > 1 and thumb_middle_angle > 1:
          # Thumbs Up (thumb far from other fingers)
          return 6
      elif abs(index_vec[0] - middle_vec[0]) < 30:  # Check if index and middle finger are close together
          # Peace sign (index and middle extended, others closed)
          if pinky_tip[1] < wrist_center[1]:  # Check if pinky is below wrist
              return 2  # Front peace sign
          else:
              return 3  # Back peace sign
      else:
          # Open palm (fingers more or less straight)
          return 4
  elif thumb_tip is not None and thumb_tip[1] > wrist_center[1] and all(tip is None for tip in [index_tip, middle_tip, pinky_tip]):
      # Thumb extended, other fingers closed (likely thumbs down)
      return 7
  elif thumb_tip is not None and index_tip is not None and thumb_tip[0] > index_tip[0] and all(tip is None for tip in [middle_tip, pinky_tip]):  # Check thumb relative to index
      # Ok sign (thumb and index touch, other fingers closed)
      return 8

  # Sign not recognized
  return -1
def main():
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
    cap = cv2.VideoCapture(0)
    pts = deque(maxlen=64)
    while cap.isOpened():
        idx_to_coordinates = {}
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec)
                idx_to_coordinates = get_idx_to_coordinates(image, results_hand)
        if 8 in idx_to_coordinates:
            pts.appendleft(idx_to_coordinates[8])  # Index Finger
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thick = int(np.sqrt(len(pts) / float(i + 1)) * 4.5)
            cv2.line(image, pts[i - 1], pts[i], (0, 255, 0), thick)
        # Assuming fingertip indices are 4 (thumb) and 8 (index)
        if 4 in idx_to_coordinates and 8 in idx_to_coordinates and 12 in idx_to_coordinates and 16 in idx_to_coordinates and 20 in idx_to_coordinates:
            thumb_tip = idx_to_coordinates[4]
            index_tip = idx_to_coordinates[8]
            middle_tip = idx_to_coordinates[12]
            pinky_tip = idx_to_coordinates[16]
            wrist_center = idx_to_coordinates[20]
            
            handSign = recognize_hand_sign(thumb_tip, index_tip, middle_tip, pinky_tip, wrist_center)
            if handSign == 0:
                cv2.putText(image, "Front fist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif handSign == 1:
                cv2.putText(image, "Back fist", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif handSign == 2:
                cv2.putText(image, "Front peace sign", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif handSign == 3:
                cv2.putText(image, "Back peace sign", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif handSign == 4:
                cv2.putText(image, "Open palm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif handSign == 6:
                cv2.putText(image, "Thumbs up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif handSign == 7:
                cv2.putText(image, "Thumbs down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif handSign == 8:
                cv2.putText(image, "Ok sign", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Not recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        cv2.imshow("Res", rescale_frame(image, percent=130))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    hands.close()
    cap.release()


if __name__ == '__main__':
    main()
