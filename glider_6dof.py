import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D2R = np.pi / 180.0 # degrees to radians
FT20 = 20 * 0.3048 # 20 ft in meters

def mark_20ft(ax):
    ax.axvline(FT20, color="red", lw=1.5)
    ax.text(FT20, 0.97, " 20 ft", color="red", transform=ax.get_xaxis_transform(), va="top", ha="left")


# fixed simulation constants
AIR_DENSITY, GRAVITY = 1.225, 9.81      # [kg/m^3], [m/s^2]
SECTION_LIFT_SLOPE = 0.9 * 2 * np.pi    # lift slope per radian of an infinite wing [1/rad]
SPAN_EFFICIENCY = 0.90                  # span efficiency (wing and tail)
STALL_ANGLE = 12 * D2R                  # wing stall angle of a flat plate
TAIL_EFFECTIVENESS = 0.5                # tail works at roughly half strength (it sits in the wing's downwash and slower air)
LAUNCH_ALPHA = 5.0 * D2R                # angle of attack at release


def params():
    return dict(
        airframe_mass=0.01260822,           # airframe mass, no ballast [kg]
        airframe_cg_x=0.1758441,            # airframe CG aft of nose [m]
        airframe_pitch_inertia=8.898e-5,    # airframe pitch MOI (Iyy) about its CG [kg m^2]

        ballast_mass=0.0028,                # [kg]
        ballast_x=0.005,                    # aft of the nose tip [m]

        wing_span=0.388870,                 # [m]
        wing_root_chord=0.0762,             # [m]
        wing_tip_chord=0.04445,             # [m]
        wing_x=0.1349375,                   # wing aerodynamic center aft of nose [m]

        tail_span=0.127,                    # [m]
        tail_chord=0.0249174,               # [m]
        tail_x=0.3431413,                   # tail aerodynamic center aft of nose [m]
        tail_angle=-1.0 * D2R,              # tail angle relative to the wing [rad]

        CD0=0.045,                          # parasitic drag coefficient

        launch_speed=7.0,                   # [m/s]
        launch_angle=10.0 * D2R,            # flight-path angle at release, up is positive [rad]
        launch_height=1.5,                  # release height [m]
    )


def derive(design):
    design = dict(design)

    # combine the airframe and the ballast: total mass, CG, and pitch MOI
    ballast_mass, ballast_x = design["ballast_mass"], design["ballast_x"]
    design["mass"] = design["airframe_mass"] + ballast_mass
    design["cg_x"] = (design["airframe_mass"] * design["airframe_cg_x"] + ballast_mass * ballast_x) / design["mass"]
    design["pitch_inertia"] = (design["airframe_pitch_inertia"]
                               + design["airframe_mass"] * (design["airframe_cg_x"] - design["cg_x"]) ** 2
                               + ballast_mass * (ballast_x - design["cg_x"]) ** 2)

    # wing and tail areas and aspect ratios (aspect ratio = span^2 / area)
    design["wing_area"] = design["wing_span"] * (design["wing_root_chord"] + design["wing_tip_chord"]) / 2
    design["wing_AR"] = design["wing_span"] ** 2 / design["wing_area"]
    design["tail_area"] = design["tail_span"] * design["tail_chord"]
    design["tail_AR"] = design["tail_span"] ** 2 / design["tail_area"]

    # lift per radian of angle of attack, for the wing and for the tail
    def lift_slope(aspect_ratio):
        return SECTION_LIFT_SLOPE / (1 + SECTION_LIFT_SLOPE / (np.pi * SPAN_EFFICIENCY * aspect_ratio))

    design["wing_lift_slope"] = lift_slope(design["wing_AR"])
    design["tail_lift_slope"] = lift_slope(design["tail_AR"])

    # the most lift the wing can make before it stalls
    design["CL_max"] = design["wing_lift_slope"] * STALL_ANGLE

    return design


def aero(design, speed, alpha, pitch_rate):
    # dynamic pressure
    dynamic_pressure = 0.5 * AIR_DENSITY * speed ** 2

    # wing lift and drag
    # lift grows with angle of attack until stall
    # drag = parasitic + induced
    CL = np.clip(design["wing_lift_slope"] * alpha, -design["CL_max"], design["CL_max"])
    CD = design["CD0"] + CL ** 2 / (np.pi * SPAN_EFFICIENCY * design["wing_AR"])
    wing_lift = dynamic_pressure * design["wing_area"] * CL

    # tail lift, angle of attack (wing angle + tail angle + pitching)
    # TAIL_EFFECTIVENESS as strong because of slower air behind the wing
    tail_alpha = alpha + design["tail_angle"] + pitch_rate * (design["tail_x"] - design["cg_x"]) / speed
    tail_lift = TAIL_EFFECTIVENESS * dynamic_pressure * design["tail_area"] * design["tail_lift_slope"] * tail_alpha

    # pitching moment about the CG, lift behind the CG pushes the nose down, lift ahead of it pushes it up
    pitch_moment = (-wing_lift * (design["wing_x"] - design["cg_x"])
                    - tail_lift * (design["tail_x"] - design["cg_x"]))

    # total lift, total drag, and the pitching moment
    total_lift = wing_lift + tail_lift
    total_drag = dynamic_pressure * design["wing_area"] * CD
    return total_lift, total_drag, pitch_moment


def simulate(design):
    design = derive(design)

    def state_rates(t, state):
        # speed, flight-path angle, pitch angle, pitch rate, distance, height
        speed, path_angle, pitch_angle, pitch_rate, distance, height = state
        speed = max(speed, 0.5)

        # aerodynamic lift, drag, and moment (angle of attack = pitch angle - flight path angle)
        lift, drag, pitch_moment = aero(design, speed, pitch_angle - path_angle, pitch_rate)

        # weight of the glider
        weight = design["mass"] * GRAVITY

        # rates of change of the state
        return [(-drag - weight * np.sin(path_angle)) / design["mass"],            # speed gets slowed down by drag and weight
                (lift - weight * np.cos(path_angle)) / (design["mass"] * speed),   # flight path angle: lift beyond the weight turns it upward
                pitch_rate,                                                        # pitch angle changes at the pitch rate
                pitch_moment / design["pitch_inertia"],                            # pitch rate is moment / MOI
                speed * np.cos(path_angle),                                        # forward distance
                speed * np.sin(path_angle)]                                        # height

    # stop the simulation when the glider reaches the ground (height = 0)
    def hit_ground(t, state):
        return state[5]

    hit_ground.terminal, hit_ground.direction = True, -1

    # starting state: launch speed, launch angle, pitch angle, no pitch rate, distance 0, release height
    start = [design["launch_speed"], design["launch_angle"], design["launch_angle"] + LAUNCH_ALPHA,
             0.0, 0.0, design["launch_height"]]

    # solve the equations of motion until the glider lands and return its path and range
    sol = solve_ivp(state_rates, [0, 15], start, events=hit_ground, max_step=0.01, rtol=1e-8)
    return dict(distance=sol.y[4], height=sol.y[5], range=sol.y[4, -1])


def main():
    base_design = params()

    # 3 to 8 g in 0.5 g increments
    ballast_masses = np.arange(3.0, 8.01, 0.5) * 1e-3
    runs = [simulate({**base_design, "ballast_mass": m}) for m in ballast_masses]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(ballast_masses)))

    for color, m, run in zip(colors, ballast_masses, runs):
        ax.plot(run["distance"], run["height"], color=color, label=f"{m*1e3:.1f} g")

    mark_20ft(ax)

    ax.set(xlabel="forward distance [m]", ylabel="height [m]",
           title=f"side view by ballast mass, ballast_x = {base_design['ballast_x']*1e3:.0f} mm")
    ax.legend(title="ballast mass", fontsize=8)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig("glider_sideview_ballast.png", dpi=130)


if __name__ == "__main__":
    main()