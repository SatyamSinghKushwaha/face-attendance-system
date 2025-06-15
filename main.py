import os
import datetime
import json
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import threading
import time
import numpy as np
import pickle

from timing_counters import update_attendance, get_user_timer_data
import util


class App:
    def __init__(self):
        self.main_window = tk.Tk()

        # Dynamically center the main window
        screen_width = self.main_window.winfo_screenwidth()
        screen_height = self.main_window.winfo_screenheight()
        window_width = 1200
        window_height = 520

        self.x_pos = int((screen_width / 2) - (window_width / 2))
        self.y_pos = int((screen_height / 2) - (window_height / 2))

        self.main_window.geometry(f"{window_width}x{window_height}+{self.x_pos}+{self.y_pos}")

        self.main_window.title("Face Recognition Attendance System")

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(
            self.main_window, 'register new user', 'gray', self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.label_present_time = tk.Label(self.main_window, text="Present: 0s", font=("Helvetica", 12))
        self.label_present_time.place(x=750, y=30)

        self.label_absent_time = tk.Label(self.main_window, text="Absent: 0s", font=("Helvetica", 12))
        self.label_absent_time.place(x=750, y=60)

        self.label_total_missed = tk.Label(self.main_window, text="Total Missed: 0s", font=("Helvetica", 12))
        self.label_total_missed.place(x=750, y=90)

        self.add_webcam(self.webcam_label)

        self.db_dir = "face_db"
        os.makedirs(self.db_dir, exist_ok=True)
        self.known_encodings, self.known_names, self.multi_encodings_dict = util.load_known_faces(self.db_dir)


        self.users_file_path = os.path.join(self.db_dir, 'users.json')
        if not os.path.exists(self.users_file_path) or os.path.getsize(self.users_file_path) == 0:
            with open(self.users_file_path, 'w') as f:
                json.dump({}, f)

        self.log_path = './log.txt'
        self.current_user = None
        self.update_timers_job = None
        self.logged_in_emp_ids = set()

    def add_webcam(self, label):
        self.cap = cv2.VideoCapture(0)
        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            self.most_recent_capture_arr = frame
            img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)
        self._label.after(20, self.process_webcam)

    def login(self):
        def login_task():
            if self.current_user is not None:
                util.msg_box("Already Logged In", f"User '{self.current_user}' is already logged in.")
                return

            status, name_or_id = util.recognize(
                self.most_recent_capture_arr,
                self.db_dir,
                self.known_encodings,
                self.known_names
            )

            if status == 'no_persons_found':
                util.msg_box("Error", "No face detected. Please try again.")
            elif status == 'multiple_faces_detected':
                util.msg_box("Error", "Multiple faces detected. Ensure only one person is in front of the camera.")
            elif status == 'unknown_person':
                util.msg_box("Error", "Face not recognized. Please register first.")
            else:
                name = status
                emp_id = name_or_id
                util.msg_box('Welcome back!', f'Welcome, {name} (ID: {emp_id}).')
                with open(self.log_path, 'a') as f:
                    f.write(f'{name},{emp_id},{datetime.datetime.now()},in\n')
                self.current_user = name
                self.logged_in_emp_ids.add(emp_id)
                self.run_timer_updates()

        threading.Thread(target=login_task).start()

    def update_register_video_feed(self):
        if not self.running_register_feed:
            return

        ret, frame = self.cap.read()
        if ret:
            self.register_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=pil_img)
            self.capture_label.imgtk = imgtk
            self.capture_label.configure(image=imgtk)

        # Repeat every 50ms
        self.register_new_user_window.after(50, self.update_register_video_feed)

    def logout(self):
        def logout_task():
            if self.current_user is None:
                util.msg_box("Error", "No user is currently logged in.")
                return

            # Recognize who is in front of the camera
            status, name_or_id = util.recognize(
                self.most_recent_capture_arr,
                self.db_dir,
                self.known_encodings,
                self.known_names
            )

            if status == 'no_persons_found':
                util.msg_box("Error", "No face detected. Please try again.")
                return
            elif status == 'multiple_faces_detected':
                util.msg_box("Error", "Multiple faces detected. Ensure only one person is in front of the camera.")
                return
            elif status == 'unknown_person':
                util.msg_box("Error", "Face not recognized. Please try again.")
                return

            # ✅ Ensure only the logged-in person can logout
            if status != self.current_user:
                util.msg_box("Error", f"You are not the logged-in user ({self.current_user}). Logout denied.")
                return

            # Proceed with logout
            name = status
            emp_id = name_or_id
            util.msg_box("Hasta la vista!", f"Goodbye, {name} (ID: {emp_id}).")
            with open(self.log_path, 'a') as f:
                f.write(f'{name},{emp_id},{datetime.datetime.now()},out\n')

            if emp_id in self.logged_in_emp_ids:
                self.logged_in_emp_ids.remove(emp_id)

            if self.update_timers_job:
                self.main_window.after_cancel(self.update_timers_job)
                self.update_timers_job = None

            self.current_user = None
            self.label_present_time.config(text="Present: 0s")
            self.label_absent_time.config(text="Absent: 0s")
            self.label_total_missed.config(text="Total Missed: 0s")

            # Remove name and ID labels from UI
            if hasattr(self, 'label_name'):
                self.label_name.destroy()
                del self.label_name
            if hasattr(self, 'label_emp_id'):
                self.label_emp_id.destroy()
                del self.label_emp_id

        threading.Thread(target=logout_task).start()

    def run_timer_updates(self):
        def update():
            if self.current_user is None:
                return  # No user logged in

            def threaded_recognition():
                status, emp_id_detected = util.recognize(
                    self.most_recent_capture_arr,
                    self.db_dir,
                    use_multi_encodings=True  # New flag
                )

                is_present = (status == self.current_user)

                # Update attendance
                update_attendance(self.current_user, is_present)

                # Get updated timer data
                timers = get_user_timer_data(self.current_user)
                present = timers['presentCounter']
                absent = timers['absentCounter']
                missed = timers['absentTimeCounter']

                # Get emp_id from the user JSON
                try:
                    with open(self.users_file_path, 'r') as f:
                        users_data = json.load(f)
                        emp_id = users_data.get(self.current_user, "N/A")
                except:
                    emp_id = "N/A"

                # Update all labels (must be run on main thread)
                def update_ui():
                    self.label_present_time.config(text=f"Present: {present}s")
                    self.label_absent_time.config(text=f"Absent: {absent}s")
                    self.label_total_missed.config(text=f"Total Missed: {missed}s")

                    # Add name and ID if not already shown
                    if not hasattr(self, 'label_name'):
                        self.label_name = tk.Label(self.main_window, text=f"Name: {self.current_user}",
                                                   font=("Helvetica", 12))
                        self.label_name.place(x=750, y=120)
                    else:
                        self.label_name.config(text=f"Name: {self.current_user}")

                    if not hasattr(self, 'label_emp_id'):
                        self.label_emp_id = tk.Label(self.main_window, text=f"Emp ID: {emp_id}", font=("Helvetica", 12))
                        self.label_emp_id.place(x=750, y=150)
                    else:
                        self.label_emp_id.config(text=f"Emp ID: {emp_id}")

                    # Trigger alert every 30 seconds (30, 60, 90...) of total missed time
                    if missed > 0:
                        if not hasattr(self, 'last_alert_threshold'):
                            self.last_alert_threshold = 0

                        if missed > self.last_alert_threshold and missed % 30 == 0:
                            util.msg_box("Warning!", f"{self.current_user} has been absent for {missed} seconds!")
                            self.last_alert_threshold = missed

                self.main_window.after(0, update_ui)

                # Schedule the next update
                self.update_timers_job = self.main_window.after(5000, update)

            threading.Thread(target=threaded_recognition).start()

        # Reset alert flag for new session
        self.alert_shown = False
        update()

    def on_closing(self):
        if self.update_timers_job:
            self.main_window.after_cancel(self.update_timers_job)
        self.cap.release()
        self.main_window.destroy()

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)


        # Position slightly right of main window
        reg_x = self.x_pos + 40  # or any offset you like
        reg_y = self.y_pos + 20
        self.register_new_user_window.geometry(f"1200x520+{reg_x}+{reg_y}")

        self.register_new_user_window.title("Register New User")
        # Instruction Label
        instruction_text = (
            "📸 Move your face as instructed while we capture:\n"
            "1. Look straight (5 frames)\n"
            "2. Slowly turn left (5 frames)\n"
            "3. Slowly turn right (5 frames)\n"
            "4. Look slightly up/down (5 frames)\n"
            "5. Smile / neutral (10 frames)\n"
            "⚠️ Ensure good lighting and only one face is visible"
        )
        util.get_text_label(self.register_new_user_window, instruction_text).grid(row=2, column=0, columnspan=2,
                                                                                  pady=10)
        instruction_label = tk.Label(
            self.register_new_user_window,
            text=instruction_text,
            font=("Helvetica", 13),
            fg="blue",
            justify="left",
            wraplength=400,
            padx=10,
            pady=10
        )
        instruction_label.place(x=720, y=350)  # Adjust as needed

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)
        self.running_register_feed = True
        self.update_register_video_feed()

        x_label = 750
        x_entry = 900

        self.text_label_emp_id_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Emp ID:')
        self.text_label_emp_id_register_new_user.config(font=("sans-serif", 14))
        self.text_label_emp_id_register_new_user.place(x=x_label, y=40)

        self.entry_emp_id_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_emp_id_register_new_user.place(x=x_entry, y=40)

        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Username:')
        self.text_label_register_new_user.config(font=("sans-serif", 14))
        self.text_label_register_new_user.place(x=x_label, y=100)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=x_entry, y=100)

        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=850, y=180)

        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=850, y=300)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()
        self.running_register_feed = False
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get().strip()
        emp_id = self.entry_emp_id_register_new_user.get().strip()

        if not name or not emp_id:
            util.msg_box("Error", "Name and Emp ID cannot be empty!")
            return

        users_file = self.users_file_path
        users_data = {}
        if os.path.exists(users_file) and os.path.getsize(users_file) > 0:
            with open(users_file, 'r') as f:
                try:
                    users_data = json.load(f)
                except json.JSONDecodeError:
                    users_data = {}

        if name in users_data:
            util.msg_box("Error", f"Username '{name}' is already taken!")
            return

        if emp_id in users_data.values():
            util.msg_box("Error", f"Emp ID '{emp_id}' is already registered!")
            return

        # Create directory for this user
        user_dir = os.path.join(self.db_dir, name)
        os.makedirs(user_dir, exist_ok=True)

        self.capture_count = 0
        self.total_captures = 30
        self.capture_user_dir = user_dir

        # Label for progress
        self.label_capture_status = tk.Label(self.register_new_user_window, text="Capturing images...",
                                             font=("Helvetica", 12), fg="green")
        self.label_capture_status.place(x=850, y=250)

        # Start threaded capture
        threading.Thread(target=self.capture_images_for_registration, args=(name, emp_id)).start()

    def capture_images_for_registration(self, name, emp_id):
        saved = 0
        max_count = self.total_captures
        while saved < max_count:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Detect face
            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) != 1:
                self.register_new_user_window.after(0, lambda:
                self.label_capture_status.config(text="Ensure only one face is visible")
                                                    )
                time.sleep(0.5)
                continue

            img_path = os.path.join(self.capture_user_dir, f'{saved}.jpg')
            cv2.imwrite(img_path, frame)
            saved += 1

            self.register_new_user_window.after(0, lambda count=saved:
            self.label_capture_status.config(text=f"Capturing image {count}/{max_count}")
                                                )
            time.sleep(0.6)

        # Save user data
        users_file = self.users_file_path
        if os.path.exists(users_file):
            with open(users_file, 'r') as f:
                try:
                    users_data = json.load(f)
                except:
                    users_data = {}
        else:
            users_data = {}

        users_data[name] = emp_id
        with open(users_file, 'w') as f:
            json.dump(users_data, f, indent=4)

        # Compute average encoding from all valid face encodings
        encodings = []
        for i in range(self.total_captures):
            img_path = os.path.join(self.capture_user_dir, f'{i}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                continue
            face_encs = face_recognition.face_encodings(img)
            if face_encs:
                encodings.append(face_encs[0])

        if encodings:
            avg_encoding = np.mean(encodings, axis=0)
            enc_path = os.path.join(self.capture_user_dir, 'avg_encoding.pkl')
            with open(enc_path, 'wb') as f:
                pickle.dump(avg_encoding, f)

        # Save all encodings as multi_encodings.pkl
        multi_path = os.path.join(self.capture_user_dir, 'multi_encodings.pkl')
        with open(multi_path, 'wb') as f:
            pickle.dump(encodings, f)

        # Reload known encodings
        self.known_encodings, self.known_names, self.multi_encodings_dict = util.load_known_faces(self.db_dir)

        # Notify and close
        self.register_new_user_window.after(0, lambda: util.msg_box(
            'Success!', f'User {name} with ID {emp_id} registered successfully!'
        ))
        self.register_new_user_window.after(0, self.register_new_user_window.destroy)
        self.running_register_feed = False

    def start(self):
        self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.main_window.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
