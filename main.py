import cv2
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Function to detect which fingers are up
def get_finger_states(hand_landmarks):
    finger_states = []

    # Tip landmarks of fingers: Thumb, Index, Middle, Ring, Pinky
    tips = [4, 8, 12, 16, 20]

    # Thumb: compare x (left vs right)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0] - 1].x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    # Other fingers: compare y (up vs down)
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states  # Example: [1, 1, 0, 0, 0]

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not opening")
else:
    print("✅ Webcam opened")

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Failed to read from webcam")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Detect finger states
            finger_states = get_finger_states(hand_landmark)

            # Rule-based letter detection
            letter = ""

            if finger_states == [0, 0, 0, 0, 0]:
                letter = "A"
            elif finger_states == [1, 1, 1, 1, 1]:
                letter = "B"
            elif finger_states == [0, 1, 0, 0, 0]:
                letter = "D"
            elif finger_states == [1, 1, 1, 1, 1]:
                letter = "C"  # same as B
            elif finger_states == [1, 1, 0, 0, 0]:
                letter = "L"
            elif finger_states == [0, 1, 1, 0, 0]:
                letter = "V"
            elif finger_states == [1, 0, 0, 0, 0]:
                letter = "E"
            elif finger_states == [1, 0, 1, 1, 1]:
                letter = "F"
            elif finger_states == [0, 1, 1, 0, 0]:
                letter = "U"
            elif finger_states == [1, 0, 0, 0, 1]:
                letter = "Y"

            # Only show letter if one is matched
            if letter != "":
                cv2.putText(frame, f"Letter: {letter}", (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 0), 3)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
