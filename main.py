import cv2
import face_recognition
from flask import Flask, Response, render_template, request

app = Flask(__name__)

reference_image = face_recognition.load_image_file("arhum.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                
                matches = face_recognition.compare_faces([reference_encoding], face_encoding)

                if matches[0]:
                    label = "arhum Recognized"
                    color = (0, 255, 0)  
                else:
                    label = "Unknown"
                    color = (0, 0, 255)  

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
