import os
import pickle
import json
import tkinter as tk
from tkinter import messagebox
import face_recognition

import numpy as np  # Make sure this is at the top if not already

def match_face(current_encoding, known_encodings, known_names, tolerance=0.42):
    """
    Compares current face encoding with known encodings using face distance.
    Returns matched name or 'Unknown'.
    """
    if not known_encodings:
        return "Unknown"

    face_distances = face_recognition.face_distance(known_encodings, current_encoding)
    best_match_index = np.argmin(face_distances)

    if face_distances[best_match_index] < tolerance:
        return known_names[best_match_index]
    else:
        return "Unknown"



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
    inputtxt = tk.Entry(window, font=("Arial", 20))
    return inputtxt



def msg_box(title, description):
    messagebox.showinfo(title, description)


def recognize(img, db_path, known_encodings=None, known_names=None):
    embeddings_unknown_list = face_recognition.face_encodings(img)

    # Check for face count
    if len(embeddings_unknown_list) == 0:
        return 'no_persons_found', None
    elif len(embeddings_unknown_list) > 1:
        return 'multiple_faces_detected', None

    embeddings_unknown = embeddings_unknown_list[0]

    json_path = os.path.join(db_path, 'users.json')
    if not os.path.exists(json_path) or os.stat(json_path).st_size == 0:
        users_data = {}
    else:
        with open(json_path, 'r') as f:
            users_data = json.load(f)

    # If no preloaded data, fallback to loading
    if known_encodings is None or known_names is None:
        known_encodings, known_names = load_known_faces(db_path)

    matched_name = match_face(embeddings_unknown, known_encodings, known_names, tolerance=0.42)

    if matched_name == "Unknown":
        return 'unknown_person', None
    else:
        emp_id = users_data.get(matched_name, 'Unknown')
        return matched_name, emp_id


def load_known_faces(db_path):
    """
    Loads and encodes all face images in the database folder once.
    Returns a list of known encodings and corresponding names.
    """
    known_encodings = []
    known_names = []

    for file_name in sorted(os.listdir(db_path)):
        if not file_name.lower().endswith('.jpg'):
            continue

        name = file_name[:-4]
        img_path = os.path.join(db_path, file_name)
        db_image = face_recognition.load_image_file(img_path)
        db_encodings = face_recognition.face_encodings(db_image)

        if db_encodings:
            known_encodings.append(db_encodings[0])
            known_names.append(name)

    return known_encodings, known_names


