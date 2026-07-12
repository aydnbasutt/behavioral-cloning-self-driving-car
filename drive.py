import socketio
import eventlet
import numpy as np
import cv2
from flask import Flask
from io import BytesIO
from PIL import Image
import base64
from tensorflow.keras.models import load_model

sio = socketio.Server()
app = Flask(__name__)
model = load_model('model.h5')

MAX_SPEED = 15  # arabanın maksimum hız


def preprocess_image(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # (drive.py'da görüntü RGB geliyor)
    img = cv2.resize(img, (200, 66))
    return img


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        speed = float(data['speed'])

        # Base64 ile gelen görüntü
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)
        image = preprocess_image(image)
        image = np.array([image])  # modele (1, 66, 200, 3) şeklinde vermek için

        steering_angle = float(model.predict(image, verbose=0)[0][0])

        # hız kontrolü
        throttle = 1.0 - (speed / MAX_SPEED)

        print(f'steering: {steering_angle:.4f}, throttle: {throttle:.4f}, speed: {speed:.2f}')
        send_control(steering_angle, throttle)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("Simülatör bağlandı:", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        },
        skip_sid=True
    )


if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)