#!/usr/bin/env python
# coding: utf-8

# # Task 3

# In[1]:


from fastapi import FastAPI, File, UploadFile
import numpy as np
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load trained model
model = load_model("image_classification_model.h5")

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    image = load_img(io.BytesIO(image_bytes), target_size=(64, 64))

    # Preprocess image
    image_array = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    result = model.predict(image)
    prediction = "car" if result[0][0] < 0.5 else "lorry"

    # Visualization (for debugging)
    plt.imshow(image_array)
    plt.title(f'Prediction: {prediction}')
    plt.axis('off')
    plt.savefig("prediction_result.png")  # Save instead of showing
    plt.close()

    return {"prediction": prediction}



# In[ ]:




