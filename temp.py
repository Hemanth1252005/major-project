import traceback
from tensorflow.keras.models import load_model, Model

try:
    load_model('models/model_19.h5')
except Exception as e:
    with open('error3.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
