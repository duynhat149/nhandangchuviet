import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
from keras.models import load_model
from gtts import gTTS
import pygame
import os
import tempfile

# Các định nghĩa chữ cái và tương ứng với các hình ảnh gesture
gesture_names = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F',
    16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L',
    22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
    34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd',
    40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j',
    46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p',
    52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v',
    58: 'w', 59: 'x', 60: 'y', 61: 'z'
}

# Load mô hình từ file .h5
model = load_model('models/mymodel.h5')

# Khởi tạo pygame
pygame.mixer.init()

# Hàm để chuyển đổi hình ảnh thành định dạng phù hợp để dự đoán
def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255
    return img_array

# Hàm dự đoán chữ cái từ ảnh
def predict_letter(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_letter = gesture_names[predicted_index]
    probability = predictions[0][predicted_index]
    return predicted_letter, probability

# Hàm để phân đoạn ảnh thành các ký tự
def segment_characters(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 2 and h >= 8:
            roi = image.crop((x, y, x + w, y + h))
            roi = roi.resize((224, 224))
            regions.append((roi, (x, y, w, h)))

    regions = sorted(regions, key=lambda region: region[1][0])
    return regions

# Hàm để chọn ảnh từ hệ thống
def select_image():
    global recognized_text  # Đặt biến recognized_text thành biến toàn cục
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        characters = segment_characters(image)
        recognized_text = ""
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)
        font = ImageFont.truetype("arial.ttf", 50)

        for roi, (x, y, w, h) in characters:
            predicted_letter, probability = predict_letter(roi)
            recognized_text += predicted_letter
            draw.rectangle([x, y, x + w, y + h], outline="red")
            draw.text((x, y - 50), f"{predicted_letter} ({probability:.2f})", fill="red", font=font)

        update_ui(image_with_boxes, recognized_text)

# Hàm để cập nhật giao diện người dùng với ảnh và kết quả dự đoán
def update_ui(image_with_boxes, predicted_text):
    # Xóa hình ảnh và kết quả cũ (nếu có)
    canvas.delete("all")
    label_predicted_text.config(text="")

    # Hiển thị ảnh trên giao diện
    img_resized = image_with_boxes.resize((300, 300))
    photo = ImageTk.PhotoImage(img_resized)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo

    # Hiển thị kết quả dự đoán
    label_predicted_text.config(text="Chuỗi ký tự nhận dạng được: " + predicted_text, font=("Arial", 14))

# Hàm chuyển văn bản thành giọng nói và phát âm thanh
def speak_text():
    tts = gTTS(text=recognized_text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        audio_file = temp_audio.name
    tts.save(audio_file)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    os.remove(audio_file)

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Nhận dạng chữ viết")

# Tạo khung thông tin sinh viên và môn học
frame_info = tk.Frame(root)
frame_info.pack(pady=10)

label_topic = tk.Label(frame_info, text="Đề tài: Nhận dạng chữ viết tay", font=("Arial", 12))
label_topic.pack()

label_group = tk.Label(frame_info, text="Nhóm sinh viên thực hiện:", font=("Arial", 12, "bold"))
label_group.pack()

label_student1 = tk.Label(frame_info, text="Nguyễn Duy Nhất - MSSV: 2154810082", font=("Arial", 12))
label_student1.pack()

label_student2 = tk.Label(frame_info, text="Cao Thị Minh Thư - MSSV: 2154810013", font=("Arial", 12))
label_student2.pack()

label_teacher_title = tk.Label(frame_info, text="Giảng viên hướng dẫn:", font=("Arial", 12, "bold"))
label_teacher_title.pack()

label_teacher_name = tk.Label(frame_info, text="TS. Trần Nguyên Bảo", font=("Arial", 12))
label_teacher_name.pack()
# Tạo các thành phần giao diện khác
button_select_image = tk.Button(root, text="Thêm ảnh", command=select_image, font=("Arial", 12))
button_select_image.pack(pady=10)

canvas = tk.Canvas(root, width=300, height=300, bg="white", borderwidth=2, relief="groove")
canvas.pack(pady=10)

label_predicted_text = tk.Label(root, text="", font=("Arial", 14))
label_predicted_text.pack(pady=10)

# Nút để đọc văn bản nhận dạng được
button_speak_text = tk.Button(root, text="Đọc văn bản", command=speak_text, font=("Arial", 12))
button_speak_text.pack(pady=10)

# Biến toàn cục để lưu trữ văn bản nhận dạng được
recognized_text = ""

# Chạy vòng lặp chính của giao diện
root.mainloop()
