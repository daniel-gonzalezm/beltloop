"""beltloop: longitudinal dynamics of a belt-conveyor loop with a gravity take-up at an
arbitrary position (continuous model, closed-form characteristic equation, modal
start-up response)."""
from .conveyor import Conveyor, DimensionalStartup, GravityTakeUp
from .eigen import (ModalBasis, fixed_free_roots, modal_basis, natural_frequencies,
                    natural_frequencies_torque, poles, damped_drive_root, rayleigh_takeup_bound,
                    strand_modes, strand_participation, takeup_mass_approx, takeup_mass_modes,
                    takeup_mass_threshold)
from .forcing import PEAK_FACTOR, PiecewiseProfile, StartProfile
from .loop import Loop, Segment
from .metrics import (StartupMetrics, TakeupKinematics, TensionRequirement, takeup_kinematics, strand_extremes, strand_rebound, tension_requirement, crawl_onset, crawl_start, fast_start_limit, startup_metrics,
                      startup_with_onset, strand_curves, strand_travel_estimate)
from .response import (StartupResponse, integrate_modes, lagged_loads, residual_amplitude_sine,
                       startup_response)
from .transfer import AB, J, S, char_fun, char_fun_torque, takeup_transmission

__version__ = "0.1.0"
