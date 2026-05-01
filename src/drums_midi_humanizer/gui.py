import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from .config.drums import DRUMMER_PROFILES
from .core.humanizer import DrumHumanizer, HumanizerConfig
from .visualization.visualizer import build_drum_figure

logger = logging.getLogger(__name__)

class HumanizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MIDI Drum Humanizer")
        self.root.geometry("900x800")
        self.root.minsize(800, 600)
        
        # Apply basic styling
        style = ttk.Style()
        style.configure("TLabel", padding=2)
        style.configure("TButton", padding=5)
        
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top-level variables
        self.input_file_var = tk.StringVar()
        self.output_file_var = tk.StringVar()
        self.style_var = tk.StringVar(value="balanced")
        self.library_var = tk.StringVar(value="gm")
        
        self.timing_var = tk.IntVar(value=10)
        self.velocity_var = tk.IntVar(value=15)
        self.ghost_var = tk.DoubleVar(value=0.1)
        self.accent_var = tk.DoubleVar(value=0.2)
        self.shuffle_var = tk.DoubleVar(value=0.0)
        self.flams_var = tk.DoubleVar(value=0.0)
        
        self.canvas_widget = None
        self.toolbar = None
        
        self._setup_file_selection()
        self._setup_parameters()
        self._setup_actions()
        self._setup_image_display()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        import matplotlib.pyplot as plt
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        import sys
        sys.exit(0)

    def _setup_file_selection(self):
        file_frame = ttk.LabelFrame(self.main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input
        ttk.Label(file_frame, text="Input MIDI:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.input_file_var, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse...", command=self._browse_input).grid(row=0, column=2)
        
        # Output
        ttk.Label(file_frame, text="Output MIDI:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_file_var, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self._browse_output).grid(row=1, column=2, pady=5)

    def _setup_parameters(self):
        params_frame = ttk.LabelFrame(self.main_frame, text="Humanization Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Style & Library
        ttk.Label(params_frame, text="Drummer Style:").grid(row=0, column=0, sticky=tk.W)
        style_cb = ttk.Combobox(params_frame, textvariable=self.style_var, values=list(DRUMMER_PROFILES.keys()), state="readonly")
        style_cb.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(params_frame, text="Drum Library:").grid(row=0, column=2, sticky=tk.W, padx=(15, 0))
        lib_cb = ttk.Combobox(params_frame, textvariable=self.library_var, values=["gm", "ad2", "sd3", "ez2", "ssd5", "mtpk2"], state="readonly")
        lib_cb.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Timing & Velocity Variation
        ttk.Label(params_frame, text="Timing Variation (ticks):").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(params_frame, from_=0, to=100, textvariable=self.timing_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Velocity Variation:").grid(row=1, column=2, sticky=tk.W, padx=(15, 0))
        ttk.Spinbox(params_frame, from_=0, to=127, textvariable=self.velocity_var, width=10).grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Probabilities
        ttk.Label(params_frame, text="Ghost Note Prob (0-1):").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.ghost_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Accent Prob (0-1):").grid(row=2, column=2, sticky=tk.W, padx=(15, 0))
        ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.accent_var, width=10).grid(row=2, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Shuffle Amount (0-0.5):").grid(row=3, column=0, sticky=tk.W)
        ttk.Spinbox(params_frame, from_=0.0, to=0.5, increment=0.05, textvariable=self.shuffle_var, width=10).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Flam Prob (0-1):").grid(row=3, column=2, sticky=tk.W, padx=(15, 0))
        ttk.Spinbox(params_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.flams_var, width=10).grid(row=3, column=3, padx=5, pady=5, sticky=tk.W)

    def _setup_actions(self):
        action_frame = ttk.Frame(self.main_frame)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.run_btn = ttk.Button(action_frame, text="▶ Run Humanization", command=self._run_humanization)
        self.run_btn.pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(action_frame, textvariable=self.status_var, font=("TkDefaultFont", 10, "italic")).pack(side=tk.LEFT, padx=15)

    def _setup_image_display(self):
        # We use a frame to host the matplotlib canvas and toolbar
        self.img_frame = ttk.LabelFrame(self.main_frame, text="Visualization", padding="5")
        self.img_frame.pack(fill=tk.BOTH, expand=True)

    def _browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input MIDI File",
            filetypes=[("MIDI files", "*.mid *.midi")]
        )
        if filename:
            self.input_file_var.set(filename)
            # Auto-fill output if empty
            if not self.output_file_var.get():
                p = Path(filename)
                self.output_file_var.set(str(p.parent / f"{p.stem}_humanized{p.suffix}"))

    def _browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output MIDI File",
            filetypes=[("MIDI files", "*.mid *.midi")],
            defaultextension=".mid"
        )
        if filename:
            self.output_file_var.set(filename)

    def _run_humanization(self):
        input_path = self.input_file_var.get().strip()
        output_path = self.output_file_var.get().strip()
        
        if not input_path:
            messagebox.showerror("Error", "Please select an Input MIDI file.")
            return
            
        if not Path(input_path).exists():
            messagebox.showerror("Error", "Input file does not exist.")
            return

        self.status_var.set("Processing... Please wait.")
        self.root.update()

        try:
            config = HumanizerConfig(
                timing_variation=self.timing_var.get(),
                velocity_variation=self.velocity_var.get(),
                ghost_note_prob=self.ghost_var.get(),
                accent_prob=self.accent_var.get(),
                shuffle_amount=self.shuffle_var.get(),
                flamming_prob=self.flams_var.get(),
                drummer_style=self.style_var.get(),
                drum_library=self.library_var.get(),
                visualize=False  # Do not generate static PNG explicitly
            )
            
            humanizer = DrumHumanizer(config)
            orig_msgs, human_msgs = humanizer.process_file(input_path, output_path)
            
            if orig_msgs and human_msgs:
                self._load_and_display_interactive_plot(orig_msgs, human_msgs, humanizer.ticks_per_beat)
                self.status_var.set(f"Done! Saved: {Path(output_path).name}")
            else:
                self.status_var.set("Done! But visualization failed (no notes).")
                
        except Exception as e:
            logger.exception("Error during humanization")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_var.set("Error occurred.")

    def _load_and_display_interactive_plot(self, orig_msgs, human_msgs, ticks_per_beat=480):
        if self.canvas_widget:
            self.canvas_widget.get_tk_widget().destroy()
            self.canvas_widget = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
            
        fig = build_drum_figure(orig_msgs, human_msgs, ticks_per_beat)
        
        # Create canvas
        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.img_frame)
        self.canvas_widget.draw()
        
        # Pack toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas_widget, self.img_frame)
        self.toolbar.update()
        
        # Pack canvas taking up the rest 
        self.canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    root = tk.Tk()
    HumanizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
