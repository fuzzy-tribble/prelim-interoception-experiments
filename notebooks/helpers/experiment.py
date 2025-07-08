import os
import time
import pandas as pd
from datetime import datetime

from helpers.const import EXP_LOG_DIR

import ipywidgets as widgets
from IPython.display import display, clear_output

from IPython.display import display, HTML


class Experiment:
    def __init__(self, name):
        self.name = name # heartbeat, skin-discrimination, thermal-regulation
        self.experimental_run_template_path = None
        self.analysis_template_path = None
        self.log_path = self.create_log(name)

        print(f"Created new experiment: {self.name}")

    @property
    def log(self):
        # display log file
        with open(self.log_path, "r") as f:
            log_content = f.read()
        display(HTML(f"<pre>{log_content}</pre>"))

    def create_log(self):
        """
        Create timestamped write-only log file for the experiment.
        """
        log_path = os.path.join(EXP_LOG_DIR, f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_event(f"Experiment started: {self.name} at {timestamp}", log_path)
        print(f"Created write-only experimental log: {log_path}")
        return log_path
    
    def start_countdown(seconds, label="Countdown"):
        log_event(f"{label} started ({seconds} seconds)")
        for i in range(seconds, 0, -1):
            print(f"{label}: {i} seconds remaining", end="\r")
            time.sleep(1)
        print(" " * 40, end="\r")
        log_event(f"{label} ended")

def log_event(message, log_path):
    """
    Log an event with a timestamp to the log file.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def ask_confirm(question_text, step_name):
    output = widgets.Output()

    def on_yes_clicked(_):
        log_event(f"{step_name} | Response: Yes")
        with output:
            clear_output()
            print("✔️ Response recorded: Yes")

    def on_no_clicked(_):
        log_event(f"{step_name} | Response: No")
        with output:
            clear_output()
            print("❌ Response recorded: No")

    display(widgets.HTML(f"<b>{question_text}</b>"))
    yes_button = widgets.Button(description="Yes", button_style="success")
    no_button = widgets.Button(description="No", button_style="danger")
    yes_button.on_click(on_yes_clicked)
    no_button.on_click(on_no_clicked)
    display(widgets.HBox([yes_button, no_button]), output)


def record_note(prompt="Enter any notes you'd like to record:"):
    note_input = widgets.Textarea(placeholder="Write your notes here...")
    submit_button = widgets.Button(description="Submit Note", button_style="info")
    output = widgets.Output()

    def on_submit(_):
        log_event(f"Note: {note_input.value.strip()}")
        with output:
            clear_output()
            print("📝 Note recorded.")
        note_input.value = ""

    submit_button.on_click(on_submit)
    display(widgets.VBox([widgets.HTML(f"<b>{prompt}</b>"), note_input, submit_button, output]))