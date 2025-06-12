import os
import pickle
import json
import tkinter as tk
from tkinter import messagebox
import face_recognition


def get_button(window, text, color, command, fg='white'):
    button = tk.Button(
                        window,
                        text=text,
                        activebackground="black",
                        activeforeground="white",
                        fg=fg,
                        bg=color,
                        command=command,
                        height=2,
                        width=20,
                        font=('Helvetica bold', 20)
                    )

    return button


def get_img_label(window):
    label = tk.Label(window)
    label.grid(row=0, column=0)
    return label


def get_text_label(window, text):
    label = tk.Label(window, text=text)
    label.config(font=("sans-serif", 21), justify="left")
    return label


def get_entry_text(window):
    inputtxt = tk.Text(window,
                       height=1,
                       width=15, font=("Arial", 20))
    return inputtxt


def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img, db_path):
    embeddings_unknown_list = face_recognition.face_encodings(img)
    if not embeddings_unknown_list:
        return 'no_persons_found', None

    embeddings_unknown = embeddings_unknown_list[0]

    if not os.path.exists(os.path.join(db_path, "users.json")):
        return 'unknown_person', None

    json_path = os.path.join(db_path, 'users.json')
    if os.stat(json_path).st_size == 0:
        users_data = {}
    else:
        with open(json_path, 'r') as f:
            users_data = json.load(f)

    files_in_db = sorted(os.listdir(db_path))  # ✅ renamed list of files

    for file_name in files_in_db:
        if not file_name.lower().endswith('.jpg'):
            continue

        name = file_name[:-4]
        img_path = os.path.join(db_path, file_name)
        db_image = face_recognition.load_image_file(img_path)
        db_encodings = face_recognition.face_encodings(db_image)

        if not db_encodings:
            continue

        match = face_recognition.compare_faces([db_encodings[0]], embeddings_unknown)[0]

        if match:
            emp_id = users_data.get(name, 'Unknown')
            return name, emp_id

    return 'unknown_person', None


