🚗 **Car Colour Detection & Counting System**

📘 **Project Overview**

This project automatically detects **cars and people** at a traffic signal using a YOLO deep learning model.
It identifies **car colours**, counts how many cars and people are present, and highlights them visually on the image:

* 🔴 **Red rectangles** for **blue cars**
* 🔵 **Blue rectangles** for **other cars**
* 🟢 **Green rectangles** for **people**

An elegant **Streamlit GUI** allows users to upload multiple images, preview results, and see object counts instantly.

⚙️ **Features**

* 🚘 Detects cars and people in uploaded traffic images
* 🎨 Classifies car colours (blue vs. non-blue) using HSV filtering
* 🔲 Draws coloured bounding boxes based on car colour
* 👥 Counts people present at the signal
* 🖼️ Displays annotated preview side by side with the original image
* 💻 Built with a clean Streamlit interface and YOLO for real-time inference

🧠 **Machine Learning Model**

**Model Type:** YOLO (You Only Look Once) — object detection model
**Framework:** `ultralytics` YOLOv8
**Purpose:** Detects objects such as cars and people in traffic scenes

**Detection Logic:**

```
if object == 'car':
    if color == 'blue':
        draw red rectangle
    else:
        draw blue rectangle
elif object == 'person':
    draw green rectangle
```

📊 **Performance Highlights**

* Works on static images and can be extended to live video feeds
* Achieves high accuracy in car and person detection
* Handles multiple objects per frame efficiently

🖥️ **Streamlit Interface**

* Upload one or more images simultaneously
* View both original and annotated versions side by side
* Displays real-time counts for:

  * 🚗 Total Cars
  * 🧍 Total People



🧰 **Technologies Used**

* Python
* OpenCV
* NumPy
* Pillow
* Streamlit
* Ultralytics YOLO

🚀 **How to Run**

1️⃣ Clone the repository:

```bash
git clone https://github.com/yourusername/Car-Color-Detection.git
cd Car-Color-Detection
```

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Add your YOLO model:

```
models/Yolomodel.pt
```

4️⃣ Run the Streamlit app:

```bash
streamlit run app.py
```

💬 **Summary**

This project combines **object detection and colour recognition** to identify car colours and count cars and people at traffic signals.
By integrating **YOLO** with **OpenCV** and **Streamlit**, it delivers a visually interactive and accurate traffic analysis system that mimics real-world smart surveillance.

🧩 *Smart detection made simple — see traffic intelligence in action!*
