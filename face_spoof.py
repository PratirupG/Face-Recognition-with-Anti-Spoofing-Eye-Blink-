# Import the required libraries
import cv2
import time
import dlib
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
import imutils
from imutils import face_utils
from scipy import ndimage

# Facial Landmarks
face_landmarks = r'D:\blink-detection\shape_predictor_68_face_landmarks.dat'

class SPOOF_DETECTION:
    """
        SPOOF DETECTION
        -> Eye Blink Detection
    """

    def __init__(self):
        self.EYE_AR_THRESH = 0.10
        self.EYE_AR_CONSEC_FRAMES = 1
        self.COUNTER = 0
        self.TOTAL = 0
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_landmarks)


    def EYE_ASPECT_RATIO(self, eye):
        """
            Calculate Eye Aspect Ratio
            :param eye: Eye landmarks co-ordinates
            :return aspect ratio of eye
        """

        vertical_one = dist.euclidean(eye[1], eye[5])
        vertical_two = dist.euclidean(eye[2], eye[4])
        horizontal_dist = dist.euclidean(eye[0], eye[3])
        aspect_ratio = (vertical_one + vertical_two) / (2 * horizontal_dist)
        return aspect_ratio

    def spoof_detection(self,path):
        """
            This function finds whether a person is SPOOFING or not using eye blink and calculating the area of each eye
            and checking whether or not it's below some threshold, if it's below 50 then the person is SPOOFING.

        :param path: Location of the video
        :return True(NOT SPOOFING) or False(SPOOFING)
        """

        video_path = path
        (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS[ "left_eye" ]
        (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS[ "right_eye" ]
        vs = FileVideoStream(video_path).start()
        fileStream = True
        time.sleep(1.0)

        sub = 0
        i = 0
        pre_ear = 0
        bool_ = False

        while True:
            if fileStream and not vs.more():
                return bool_
            frame = vs.read()
            if frame is None:
                return bool_

            frame = imutils.resize(frame, width=450)
            frame = ndimage.rotate(frame, 180)
            frame = ndimage.rotate(frame, 90)
            #cv2.imshow('Frame',frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = self.detector(gray, 0)

            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                left_eye = shape[ lstart:lend ]
                right_eye = shape[ rstart:rend ]
                left_ear = self.EYE_ASPECT_RATIO(left_eye)
                right_ear = self.EYE_ASPECT_RATIO(right_eye)

                # Average aspect ratio of each eye
                ear = (right_ear+left_ear) / 2.0

                if i == 0:
                    pre_ear = ear
                else:
                    sub = abs(pre_ear - ear)

                # Convex Hull (Find the convex hull of each eye it's required for calculating area under each eye
                lefteyehull = cv2.convexHull(left_eye)
                righteyehull = cv2.convexHull(right_eye)

                # Average area of each eye
                avg_area = (cv2.contourArea(lefteyehull) + cv2.contourArea(righteyehull))/2

                cv2.drawContours(frame, [ lefteyehull ], -1, (255, 0, 0), 1)
                cv2.drawContours(frame, [ righteyehull ], -1, (255, 0, 0), 1)

                if i >= 1:
                    if (avg_area < 30):
                        return False
                    if sub > 0.09:
                        self.COUNTER += 1
                    else:
                        if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                            self.TOTAL += 1
                        self.COUNTER = 0
                    if self.COUNTER == 1 or self.COUNTER > 1:
                        cv2.putText(frame, 'REAL FACE', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        bool_ = True
                        return bool_

            cv2.imshow("FRAME ", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            i += 1

        cv2.destroyAllWindows()
        vs.stop()


