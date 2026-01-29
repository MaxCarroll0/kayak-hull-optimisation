import customtkinter as ctk
from dataclasses import dataclass, fields, asdict
import sys
import json

# Initialize appearance once
ctk.set_appearance_mode("Dark") 
ctk.set_default_color_theme("blue") 

# -----------------------------------------------------------------------------
# 1. Existing WeightSelector Class
# -----------------------------------------------------------------------------
class WeightSelector:
    """
    A modern GUI class to select optimization weights for result parameters.
    """
    def __init__(self, target_class, title="Optimization Configuration"):
        self.target_class = target_class
        self.title_text = title
        self.sliders = {}  
     
        self.final_weights = {}
        
        if hasattr(target_class, "__dataclass_fields__"):
            self.params = [f.name for f in fields(target_class)]
        elif hasattr(target_class, "__annotations__"):
            self.params = [k for k in target_class.__annotations__ if not k.startswith('_')]
        else:
            self.params = [k for k in dir(target_class) if not k.startswith('_') and not callable(getattr(target_class, k))]

    def _create_row(self, parent, param_name, row_idx):
        """Helper to create one label + slider + value display row."""
        
        label = ctk.CTkLabel(parent, text=param_name.replace("_", " ").title(), font=("Roboto", 14, "bold"))
        label.grid(row=row_idx, column=0, padx=20, pady=10, sticky="w")

        val_var = ctk.IntVar(value=0)
        self.sliders[param_name] = val_var
        

        val_label = ctk.CTkLabel(parent, text="0", width=30, font=("Roboto Mono", 14))
        val_label.grid(row=row_idx, column=2, padx=(10, 20), pady=10)

        def update_label(value):
            val_label.configure(text=f"{int(value)}")
            
        slider = ctk.CTkSlider(
            parent, 
            from_=-10, 
            to=10, 
            number_of_steps=10, 
            variable=val_var, 
            command=update_label,
            width=300
        )
        slider.grid(row=row_idx, column=1, padx=10, pady=10)

    def run(self):
        """
        Builds the window, pauses execution until user finishes, 
        and returns the dictionary.
        """
        app = ctk.CTk()
        app.title(self.title_text)
        app.geometry("600x600") # Increased height slightly for new control
        
        app.grid_columnconfigure(0, weight=1)
        app.grid_rowconfigure(1, weight=1)

        header_frame = ctk.CTkFrame(app)
        header_frame.pack(fill="x", padx=20, pady=20)
        
        header = ctk.CTkLabel(header_frame, text=f"Configure Priorities", font=("Roboto", 20, "bold"))
        header.pack(pady=5)
        sub = ctk.CTkLabel(header_frame, text="Set importance (0=Ignore, 10=Critical)", text_color="gray")
        sub.pack(pady=(0, 5))

        scroll_frame = ctk.CTkScrollableFrame(app, label_text="Parameters")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        for idx, param in enumerate(self.params):
            self._create_row(scroll_frame, param, idx)

        # --- New Time Selection Section ---
        time_frame = ctk.CTkFrame(app)
        time_frame.pack(fill="x", padx=20, pady=(0, 10))

        time_title = ctk.CTkLabel(time_frame, text="Time Limit (Minutes)", font=("Roboto", 14, "bold"))
        time_title.pack(side="left", padx=20, pady=15)

        self.time_var = ctk.IntVar(value=10) # Default value
        
        time_val_label = ctk.CTkLabel(time_frame, text=f"{self.time_var.get()}", width=30, font=("Roboto Mono", 14))
        time_val_label.pack(side="right", padx=(10, 20), pady=15)

        def update_time(value):
            time_val_label.configure(text=f"{int(value)}")

        time_slider = ctk.CTkSlider(
            time_frame,
            from_=1,
            to=60,
            number_of_steps=59,
            variable=self.time_var,
            command=update_time,
            width=200
        )
        time_slider.pack(side="right", padx=10, pady=15)
        # ----------------------------------

        def on_confirm():
            for param, var in self.sliders.items():
                self.final_weights[param] = var.get()
            
            # Add time to final dictionary
            self.final_weights["time"] = self.time_var.get()
            
            app.destroy() 

        btn = ctk.CTkButton(app, text="Confirm Configuration", command=on_confirm, height=50, font=("Roboto", 16, "bold"))
        btn.pack(fill="x", padx=20, pady=20)

        app.mainloop()
        
        return self.final_weights


# -----------------------------------------------------------------------------
# 2. New ResultVisualizer Class
# -----------------------------------------------------------------------------
class ResultVisualizer:
    """
    A GUI class to display optimization results, parameters, results breakdown, 
    and launch the 3D hull view.
    """
    def __init__(self, params_dataclass_instance, results_dict: dict, score: float, hull_class, title="Optimization Result"):
        """
        Args:
            params_dataclass_instance: The dataclass instance containing optimal parameters.
            results_dict (dict): The dictionary containing detailed score components.
            score (float): The final total score achieved.
            hull_class: The class used to instantiate the Hull (must take params in __init__ and have .show()).
            title (str): Window title.
        """
        self.params = params_dataclass_instance
        self.results_dict = results_dict
        self.score = score
        self.hull_class = hull_class
        self.title_text = title
        
        # Extract dictionary from dataclass for Parameters
        if hasattr(self.params, "__dataclass_fields__"):
            self.params_dict = asdict(self.params)
        else:
            self.params_dict = {k: v for k, v in self.params.__dict__.items() if not k.startswith('_')}

    def _create_row(self, parent, key, value, row_idx, value_color=None):
        """Helper to create one Key: Value display row."""
        
        # Format key text
        key_text = str(key).replace("_", " ").title()
        
        # Key Label
        label_key = ctk.CTkLabel(parent, text=key_text, font=("Roboto", 13, "bold"), anchor="w")
        label_key.grid(row=row_idx, column=0, padx=15, pady=6, sticky="w")
        
        # Value formatting
        if isinstance(value, float):
            val_text = f"{value:.4f}"
        elif isinstance(value, bool):
            val_text = "Yes" if value else "No"
        else:
            val_text = str(value)

        # Value Label
        if value_color is None:
            value_color = "#AABBAA" # Default grayish
            
        label_val = ctk.CTkLabel(parent, text=val_text, font=("Roboto Mono", 13), text_color=value_color, anchor="e")
        label_val.grid(row=row_idx, column=1, padx=15, pady=6, sticky="e")

    def run(self):
        """
        Displays the results window.
        """
        app = ctk.CTk()
        app.title(self.title_text)
        app.geometry("900x700") # Wider window for two columns
        
        # --- 1. Header Frame (Total Score) ---
        header_frame = ctk.CTkFrame(app, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        lbl_title = ctk.CTkLabel(header_frame, text="OPTIMAL DESIGN FOUND", font=("Roboto", 16), text_color="gray")
        lbl_title.pack()
        
        # BIG SCORE DISPLAY
        lbl_score = ctk.CTkLabel(header_frame, text=f"{self.score:.4f}", font=("Roboto", 48, "bold"), text_color="#4B8BBE") # Blue-ish
        lbl_score.pack(pady=(0, 5))
        
        lbl_units = ctk.CTkLabel(header_frame, text="Total Objective Score", font=("Roboto", 12))
        lbl_units.pack()

        # --- 2. Main Content Area (Two Columns) ---
        content_frame = ctk.CTkFrame(app, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        content_frame.grid_columnconfigure(0, weight=1) # Left col
        content_frame.grid_columnconfigure(1, weight=1) # Right col
        content_frame.grid_rowconfigure(0, weight=1)

        # --- LEFT COLUMN: Design Parameters ---
        left_frame = ctk.CTkScrollableFrame(content_frame, label_text="Input Parameters")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_frame.grid_columnconfigure(1, weight=1) # Push values to right

        for idx, (k, v) in enumerate(self.params_dict.items()):
            self._create_row(left_frame, k, v, idx)

        # --- RIGHT COLUMN: Score Breakdown ---
        right_frame = ctk.CTkScrollableFrame(content_frame, label_text="Score Breakdown")
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right_frame.grid_columnconfigure(1, weight=1) # Push values to right
        
        for idx, (k, v) in enumerate(self.results_dict.items()):
            # Use a slightly brighter color for results to distinguish them
            self._create_row(right_frame, k, v, idx, value_color="#E0E0E0")

        # --- 3. Action Button (Show 3D) ---
        def on_show_hull():
            print(f"Launching 3D View for hull...")
            try:
                hull_instance = self.hull_class(self.params)
                hull_instance.show()
            except Exception as e:
                print(f"Error launching Hull.show(): {e}")
                import tkinter.messagebox
                tkinter.messagebox.showerror("Error", f"Could not launch visualization:\n{e}")

        footer_frame = ctk.CTkFrame(app, fg_color="transparent")
        footer_frame.pack(fill="x", padx=20, pady=20)

        btn = ctk.CTkButton(
            footer_frame, 
            text="VIEW 3D HULL", 
            command=on_show_hull, 
            height=60, 
            font=("Roboto", 18, "bold"),
            fg_color="#2CC985", # Green accent
            hover_color="#229965"
        )
        btn.pack(fill="x")

        app.mainloop()
# -----------------------------------------------------------------------------
# 3. Main Execution Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    # --- MOCK DATA FOR DEMONSTRATION ---
    @dataclass
    class Params:
        """Parameters defining a hull (Mock)"""
        density: float = 970.0
        hull_thickness: float = 0.004
        length: float = 3.25
        beam: float = 0.65
        depth: float = 0.30
        cross_section_exponent: float = 2.1
        beam_position: float = 0.55
        rocker_bow: float = 0.15
        rocker_stern: float = 0.10
        rocker_position: float = 0.45
        rocker_exponent: float = 2.5
        cockpit_length: float = 0.85
        cockpit_width: float = 0.45
        cockpit_position: float = 0.52
        cockpit_opening: bool = False

    class MockHull:
        """Mock Hull class to simulate the external dependency"""
        def __init__(self, params):
            self.params = params
        
        def show(self):
            # Simulate opening a window
            print(">>> [MOCK] Opening 3D Window...")
            # Create a dummy pop-up just to prove it works
            pop = ctk.CTkToplevel()
            pop.title("3D View Simulation")
            pop.geometry("300x200")
            l = ctk.CTkLabel(pop, text="[3D Hull Visualization]\n(Simulated)", font=("Roboto", 20))
            l.pack(expand=True)
            pop.lift()

    # --- 1. Run Weight Selector (Existing) ---
    @dataclass
    class GP_Result:
        overall_stability: float
        initial_stability: float
        diminishing_stability: float
        tipping_point: float
        righting_energy: float
        overall_buoyancy: float
        initial_buoyancy: float
    # print("--- Step 1: Weight Selector ---")
    # gui_weights = WeightSelector(GP_Result)
    # weights = gui_weights.run()
    # print(f"Weights selected: {weights}")
    
    # --- 2. Run Result Visualizer (New) ---
    print("\n--- Step 2: Result Visualizer ---")
    
    # Create fake optimal data
    optimal_params = Params() # Uses defaults defined above
    final_score = 0.8742 # Example flow score
    liste = [1,2,3,4]

    asking_widget = WeightSelector(GP_Result).run()
    print(f"Output Dictionary keys: {asking_widget.keys()}")
    print(f"Output Dictionary: {asking_widget}")
    # Launch the visualizer
    visualizer = ResultVisualizer(optimal_params,  {"Speed":10,"Steerability": 2,"Druggability": 3}, final_score, MockHull)
    visualizer.run()
