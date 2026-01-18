"""
This is the function that takes in a function F: Hullparams, Inputparams -> Score as well as a domain across the input parameters and then outputs.

"""




import optuna
from hullopt.hull.params import Params
from hullopt.hull.hull import Hull
from hullopt.hull.constraints import Constraints




def optimise(F, Constraint: Constraints) -> Params:
    """
    Optimizes hull parameters to maximize score from function F using Bayesian Optimization.
    
    Args:
        F: A function that takes a Params object and returns a float score.
        Constraint: An instance of Constraints defining the search space.
        
    Returns:
        The best Params object found within the time limit.
    """
    

    # 970 kg/m^3 is typical for High-Density Polyethylene (HDPE) used in kayaks
    FIXED_DENSITY = 970.0 
    best_score = float('-inf')
    best_dict = {}
   
    
    def objective(trial):

        p_len = trial.suggest_float("length", *Constraint.length_range)
        p_beam = trial.suggest_float("beam", *Constraint.beam_range)
        p_depth = trial.suggest_float("depth", *Constraint.depth_range)
        p_thick = trial.suggest_float("hull_thickness", *Constraint.hull_thickness_range)
        
        p_cs_exp = trial.suggest_float("cross_section_exponent", *Constraint.cross_section_exponent_range)
        p_beam_pos = trial.suggest_float("beam_position", *Constraint.beam_position_range)
        
        p_r_bow = trial.suggest_float("rocker_bow", *Constraint.rocker_bow_range)
        p_r_stern = trial.suggest_float("rocker_stern", *Constraint.rocker_stern_range)
        p_r_pos = trial.suggest_float("rocker_position", *Constraint.rocker_position_range)
        p_r_exp = trial.suggest_float("rocker_exponent", *Constraint.rocker_exponent_range)
        
        p_c_len = trial.suggest_float("cockpit_length", *Constraint.cockpit_length_range)
        p_c_wid = trial.suggest_float("cockpit_width", *Constraint.cockpit_width_range)
        p_c_pos = trial.suggest_float("cockpit_position", *Constraint.cockpit_position_range)

        current_params = Params(
            density=FIXED_DENSITY,
            hull_thickness=p_thick,
            length=p_len,
            beam=p_beam,
            depth=p_depth,
            cross_section_exponent=p_cs_exp,
            beam_position=p_beam_pos,
            rocker_bow=p_r_bow,
            rocker_stern=p_r_stern,
            rocker_position=p_r_pos,
            rocker_exponent=p_r_exp,
            cockpit_length=p_c_len,
            cockpit_width=p_c_wid,
            cockpit_position=p_c_pos,
            cockpit_opening=False
        )

        try:
            Constraint.check_hull(Hull(current_params))
        except ValueError:

            raise optuna.TrialPruned()

        try:
            score, dic = F(current_params)
            if score > best_score:
                best_score = score
                best_dict = dic
            return score
        except Exception as e:

            return float('-inf')
        
    optuna.logging.set_verbosity(optuna.logging.WARNING) 
    
    study = optuna.create_study(direction="maximize")
    
    print("Starting Bayesian Optimization...")
    print("Time limit: 20 minutes")
    
    study.optimize(objective, timeout=20 * 60)

    best_trial = study.best_trial
    
    print(f"Optimization finished. Best Score: {best_trial.value}")
    
    best_params = Params(
        density=FIXED_DENSITY,
        hull_thickness=best_trial.params["hull_thickness"],
        length=best_trial.params["length"],
        beam=best_trial.params["beam"],
        depth=best_trial.params["depth"],
        cross_section_exponent=best_trial.params["cross_section_exponent"],
        beam_position=best_trial.params["beam_position"],
        rocker_bow=best_trial.params["rocker_bow"],
        rocker_stern=best_trial.params["rocker_stern"],
        rocker_position=best_trial.params["rocker_position"],
        rocker_exponent=best_trial.params["rocker_exponent"],
        cockpit_length=best_trial.params["cockpit_length"],
        cockpit_width=best_trial.params["cockpit_width"],
        cockpit_position=best_trial.params["cockpit_position"],
        cockpit_opening=False
    )
    
    return best_params, best_dict, best_score