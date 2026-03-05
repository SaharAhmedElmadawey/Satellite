from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import cv2
import base64
from PIL import Image
import os

# --- 1. تعريف الدوال بنفس المسميات والبارامترات المستخدمة في التدريب ---


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(tf.cast(y_true, tf.float32), K.round(y_pred)))


def total_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# --- 2. تحميل الموديل من المسار الصحيح ---
# لاحظ إضافة 'model/' لأن ملفك داخل مجلد فرعي
MODEL_PATH = os.path.join('model', 'water_segmentation_vgg16_final.keras')

# التأكد من تحميل الموديل مع الدوال المخصصة (Custom Objects)
try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'total_loss': total_loss,
                'dice_coef': dice_coef,
                'binary_accuracy': binary_accuracy
            }
        )
        print("✅ Model loaded successfully from:", MODEL_PATH)
    else:
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500

    try:
        file = request.files['image']

        # 1. معالجة الصورة: تحويل لـ RGB وتغيير الحجم لـ 128x128
        img = Image.open(file).convert('RGB').resize((128, 128))

        # 2. تحويل الصورة لمصفوفة وعمل Normalization (/255)
        img_array = np.array(img, dtype=np.float32) / 255.0
        # إضافة dimension للـ batch
        img_array = np.expand_dims(img_array, axis=0)

        # 3. عمل التوقع (Inference)
        pred = model.predict(img_array)

        # 4. تحويل التوقع لـ Mask أسود وأبيض (Threshold 0.5)
        # القيم أكبر من 0.5 تصبح 255 (أبيض) والباقي 0 (أسود)
        pred_mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255

        # 5. تحويل الـ Mask لصورة Base64 لإرسالها للـ Frontend
        _, buffer = cv2.imencode('.png', pred_mask)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'mask': encoded_image})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # تشغيل السيرفر على Port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
