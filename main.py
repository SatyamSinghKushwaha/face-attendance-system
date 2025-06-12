import os.path
import datetime
import pickle
import json
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
from timing_counters import update_attendance, get_user_timer_data

import util


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)


        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)


        # Labels to display time tracking
        self.label_present_time = tk.Label(self.main_window, text="Present: 0s", font=("Helvetica", 12))
        self.label_present_time.place(x=750, y=30)

        self.label_absent_time = tk.Label(self.main_window, text="Absent: 0s", font=("Helvetica", 12))
        self.label_absent_time.place(x=750, y=60)

        self.label_total_missed = tk.Label(self.main_window, text="Total Missed: 0s", font=("Helvetica", 12))
        self.label_total_missed.place(x=750, y=90)



        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.users_file_path = os.path.join(self.db_dir, 'users.json')
        if not os.path.exists(self.users_file_path) or os.path.getsize(self.users_file_path) == 0:
            with open(self.users_file_path, 'w') as f:
                json.dump({}, f)

        self.log_path = './log.txt'

        self.current_user = None  # Track who's logged in
        self.update_timers_job = None  # To cancel/restart the timer loop

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        self.most_recent_capture_arr = frame
        img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        if self.current_user is not None:
            util.msg_box("Already Logged In", f"User '{self.current_user}' is already logged in.")
            return

        name, emp_id = util.recognize(self.most_recent_capture_arr, self.db_dir)

        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Welcome back !', f'Welcome, {name} (ID: {emp_id}).')
            with open(self.log_path, 'a') as f:
                f.write('{},{},{},in\n'.format(name, emp_id, datetime.datetime.now()))

            self.current_user = name
            self.run_timer_updates()

    def on_closing(self):
        if self.update_timers_job:
            self.main_window.after_cancel(self.update_timers_job)
        self.cap.release()  # Safely release webcam
        self.main_window.destroy()

    def logout(self):
        name, emp_id = util.recognize(self.most_recent_capture_arr, self.db_dir)

        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Hasta la vista !', f'Goodbye, {name} (ID: {emp_id}).')
            with open(self.log_path, 'a') as f:
                f.write('{},{},{},out\n'.format(name, emp_id, datetime.datetime.now()))

            # ✅ Stop the timing loop if a user was logged in
            if self.update_timers_job:
                self.main_window.after_cancel(self.update_timers_job)
                self.update_timers_job = None

            self.current_user = None  # Clear the logged-in user

            # ✅ Reset UI Labels
            self.label_present_time.config(text="Present: 0s")
            self.label_absent_time.config(text="Absent: 0s")
            self.label_total_missed.config(text="Total Missed: 0s")

    def run_timer_updates(self):
        def update():
            if self.current_user is None:
                return  # No user logged in

            # Check if user is present via face recognition
            detected_name, _ = util.recognize(self.most_recent_capture_arr, self.db_dir)
            is_present = (detected_name == self.current_user)

            # Update counters
            update_attendance(self.current_user, is_present)

            # Get updated counters
            timers = get_user_timer_data(self.current_user)
            present = timers['presentCounter']
            absent = timers['absentCounter']
            missed = timers['absentTimeCounter']

            # Update labels
            self.label_present_time.config(text=f"Present: {present}s")
            self.label_absent_time.config(text=f"Absent: {absent}s")
            self.label_total_missed.config(text=f"Total Missed: {missed}s")

            # Repeat every 2 seconds
            self.update_timers_job = self.main_window.after(5000, update)

        update()

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")
        self.register_new_user_window.title("Register New User")

        # Live capture display
        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)
        self.add_img_to_label(self.capture_label)

        x_label = 750
        x_entry = 900

        # Employee ID Label and Entry (on top)
        self.text_label_emp_id_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Emp ID:')
        self.text_label_emp_id_register_new_user.config(font=("sans-serif", 14))
        self.text_label_emp_id_register_new_user.place(x=x_label, y=40)

        self.entry_emp_id_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_emp_id_register_new_user.place(x=x_entry, y=40)

        # Username Label and Entry (below Emp ID)
        self.text_label_register_new_user = util.get_text_label(
            self.register_new_user_window, 'Username:')
        self.text_label_register_new_user.config(font=("sans-serif", 14))
        self.text_label_register_new_user.place(x=x_label, y=100)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=x_entry, y=100)

        # Accept Button (vertically placed)
        self.accept_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=850, y=180)

        # Try Again Button (below Accept)
        self.try_again_button_register_new_user_window = util.get_button(
            self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=850, y=260)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c").strip()
        emp_id = self.entry_emp_id_register_new_user.get(1.0, "end-1c").strip()

        if not name or not emp_id:
            util.msg_box("Error", "Name and Emp ID cannot be empty!")
            return

        # Load or create JSON
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

        # Save face image
        img_path = os.path.join(self.db_dir, f'{name}.jpg')
        cv2.imwrite(img_path, self.register_new_user_capture)

        # Save user info to JSON
        users_data[name] = emp_id
        with open(users_file, 'w') as f:
            json.dump(users_data, f, indent=4)

        util.msg_box('Success!', f'User {name} with ID {emp_id} registered successfully!')
        self.register_new_user_window.destroy()


if __name__ == "__main__":
    app = App()
    app.start()
