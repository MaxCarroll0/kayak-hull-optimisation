"""
This is the function that takes in a function F: Hullparams, Inputparams -> Score as well as a domain across the input parameters and then outputs.

"""




import optuna
from hullopt.hull.params import Params
from hullopt.hull.hull import Hull
from hullopt.hull.constraints import Constraints
import hullopt


def hull_constraints(trial):
    """
    This is to avoid check_hull failing too much.
    """

    l = trial.params.get("length", 3.0)
    b = trial.params.get("beam", 0.6)
    d = trial.params.get("depth", 0.3)
    rb = trial.params.get("rocker_bow", 0.2)
    rs = trial.params.get("rocker_stern", 0.2)
    violations = []
    l_b_ratio = l / b
    violations.append(3.0 - l_b_ratio)  
    violations.append(l_b_ratio - 7.5)  

    # Ratio: Beam to Depth (1.5 to 3.0)
    b_d_ratio = b / d
    violations.append(1.5 - b_d_ratio)  
    violations.append(b_d_ratio - 3.0) 


    violations.append(rs - rb)     

    violations.append(abs(rb - rs) - 0.05)

    return violations

best_score = float('-inf')
best_dic = {}

def optimise(F, Constraint: Constraints, time=1) -> Params:
    """
    Optimizes hull parameters to maximize score from function F using Bayesian Optimization.
    
    Args:
        F: A function that takes a Params object and returns a float score.
        Constraint: An instance of Constraints defining the search space.
        
    Returns:
        The best Params object found within the time limit.
    """
    

    # 970 kg/m^3 is typical for High-Density Polyethylene (HDPE) used in kayaks
    FIXED_DENSITY = 900.0 
    best_score = float('-inf')
    best_dict = {}
   
    
    def objective(trial):

        p_len = trial.suggest_float("length", *Constraint.length_range)
        p_len_beam_ratio = trial.suggest_float("length_beam_ratio", *Constraint.length_to_beam_ratio_range)
        p_beam_depth_ratio = trial.suggest_float("beam_depth_ratio", *Constraint.beam_to_depth_ratio_range)
        p_thick = trial.suggest_float("hull_thickness", *Constraint.hull_thickness_range)
        
        p_cs_exp = trial.suggest_float("cross_section_exponent", *Constraint.cross_section_exponent_range)
        p_beam_pos = trial.suggest_float("beam_position", *Constraint.beam_position_range)
        
        p_r_bow = trial.suggest_float("rocker_bow", *Constraint.rocker_bow_range)
        p_r_stern = trial.suggest_float("rocker_stern", *Constraint.rocker_stern_range)
        p_r_pos = trial.suggest_float("rocker_position", *Constraint.rocker_position_range)
        p_r_exp = trial.suggest_float("rocker_exponent", *Constraint.rocker_exponent_range)
        
        p_c_len_ratio = trial.suggest_float("cockpit_length_ratio", *Constraint.cockpit_length_ratio_range)
        p_c_wid_ratio = trial.suggest_float("cockpit_width_ratio", *Constraint.cockpit_width_ratio_range)
        p_c_pos = trial.suggest_float("cockpit_position", *Constraint.cockpit_position_range)

        current_params = Params.from_ratio_parameterisation(
            density=FIXED_DENSITY,
            hull_thickness=p_thick,
            length=p_len,
            length_beam_ratio=p_len_beam_ratio,
            beam_depth_ratio=p_beam_depth_ratio,
            cross_section_exponent=p_cs_exp,
            beam_position=p_beam_pos,
            rocker_bow=p_r_bow,
            rocker_stern=p_r_stern,
            rocker_position=p_r_pos,
            rocker_exponent=p_r_exp,
            cockpit_length_ratio=p_c_len_ratio,
            cockpit_width_ratio=p_c_wid_ratio,
            cockpit_position=p_c_pos,
            cockpit_opening=False
        )
        print(f"Running Aggregator on: {current_params}")
        try:
            Constraint.check_hull(Hull(current_params))
        except ValueError:
            print("Trial pruned: Constraints not met!")
            raise optuna.TrialPruned()
        import traceback
        try:
            score, dic = F(Hull(current_params))
            if score > hullopt.optimise.best_score:
                hullopt.optimise.best_score = score
                hullopt.optimise.best_dict = dic

            return score
        except Exception as e:
            traceback.print_exc()
            return float('-inf')
        
    optuna.logging.set_verbosity(optuna.logging.WARNING) 

    sampler = optuna.samplers.TPESampler(
        constraints_func=hull_constraints,
        multivariate=True

    )
    
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    print("Starting Bayesian Optimization...")
    print(f"Time limit: {time} minutes")
    
    study.optimize(objective, timeout=time * 60)

    best_trial = study.best_trial
    
    print(f"Optimization finished. Best Score: {best_trial.value}")
    
    best_params = Params.from_ratio_parameterisation(
        density=FIXED_DENSITY,
        hull_thickness=best_trial.params["hull_thickness"],
        length_beam_ratio=best_trial.params["length_beam_ratio"],
        beam_depth_ratio=best_trial.params["beam_depth_ratio"],
        depth=best_trial.params["depth"],
        cross_section_exponent=best_trial.params["cross_section_exponent"],
        beam_position=best_trial.params["beam_position"],
        rocker_bow=best_trial.params["rocker_bow"],
        rocker_stern=best_trial.params["rocker_stern"],
        rocker_position=best_trial.params["rocker_position"],
        rocker_exponent=best_trial.params["rocker_exponent"],
        cockpit_length_ratio=best_trial.params["cockpit_length_ratio"],
        cockpit_width_ratio=best_trial.params["cockpit_width_ratio"],
        cockpit_position=best_trial.params["cockpit_position"],
        cockpit_opening=False
    )
    
    return best_params
