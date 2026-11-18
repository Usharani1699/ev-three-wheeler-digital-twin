"""
EV three-wheeler digital twin (EG7023 Automotive Engineering, UEL, 2026)
Usha Rani Vamanagiri

Time-step simulation of a 3-wheel passenger EV on a synthetic urban cycle:
road load -> motor torque/speed -> motor + inverter losses -> pack current,
SOC, and lumped pack temperature, until the pack reaches 5% SOC.

Run:  python src/ev_digital_twin.py   (figures are written to figures/)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)




# ======================================================
# ANALYSIS ---VEHICLE PARAMETERS --- TECHNICAL
# ======================================================
#As per the table 1. the following specifications are considered
#Assumptions made as per the table are 3W/250km/60kmph/350kg

mass = 500  #350 kerb weight + 1 passenger+1 rider (75 kgs each)
g = 9.81
Cr = 0.015 #rolling co-efficient of auto
Cd = 0.45 #drag co-efficient of auto
A = 1.2 #front area in m^2
rho = 1.225 # density kg/m^3
#tyre
wheel_radius = 0.25
gear_ratio = 8 #transmission
drivetrain_eff = 0.9 #transmission

# ======================================================
# BATTERY PARAMETERS
# ======================================================

battery_capacity_kwh = 16
battery_capacity_wh = battery_capacity_kwh * 1000

R_int = 0.12
T = 298
T_amb = 298
mass_batt = 80 #lfp 32650/6Ah cell/80 grams per cell
Cp = 1000 #specific heat co-efficient
h = 5 #heat transfer co-efficient
A_th = 2 #surface area available per heat transfer

series_cells = 100
aux_power = 300

# ======================================================
# OCV LOOKUP
# ======================================================

soc_lookup = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
ocv_lookup = np.array([3.0,3.3,3.5,3.65,3.75,3.85,3.95,4.05,4.1,4.15,4.2])

def get_ocv(soc):
    return np.interp(np.clip(soc,0,1), soc_lookup, ocv_lookup)

# ======================================================
# MOTOR EFFICIENCY
# ======================================================

def motor_efficiency(torque, rpm):
    base_eff = 0.92
    torque_loss = 0.0006 * torque**2
    speed_loss = 0.00000025 * rpm**2
    return np.clip(base_eff - torque_loss - speed_loss, 0.75, 0.93)

# ======================================================
# INVERTER
# ======================================================

f_sw = 10000
E_on = 0.002
E_off = 0.002
R_on = 0.01

# ======================================================
# SIMULATION
# ======================================================

dt = 0.1
t = 0

soc_val = 1
dist = 0
prev_v = 0

soc = []
distance = []
battery_temp = []
power_log = []
speed_log = []

while soc_val > 0.05:

    v = 12*np.abs(np.sin(t/40)) + 5*np.abs(np.sin(t/8))
    v = np.clip(v, 0, 16.7)

    a = (v - prev_v)/dt
    prev_v = v

    F_roll = Cr * mass * g
    F_drag = 0.5 * rho * Cd * A * v**2
    F_acc = mass * a

    F_total = F_roll + F_drag + F_acc

    wheel_torque = F_total * wheel_radius
    motor_torque = wheel_torque / gear_ratio

    wheel_rpm = (v/(2*np.pi*wheel_radius))*60
    motor_rpm = wheel_rpm * gear_ratio

    eff = motor_efficiency(motor_torque, motor_rpm)

    regen_eff = 0.35 #reference article
    if F_total < 0 and soc_val < 0.9:
        P_motor = F_total * v * regen_eff
    else:
        P_motor = F_total * v

    if P_motor >= 0:
        P_elec = P_motor / (eff * drivetrain_eff + 1e-6)
    else:
        P_elec = P_motor * regen_eff * drivetrain_eff

    I_est = P_elec / (get_ocv(soc_val)*series_cells + 1e-6)
    P_inv = (I_est**2)*R_on + f_sw*(E_on+E_off)

    P_total = P_elec + P_inv + aux_power

    V_oc = get_ocv(soc_val) * series_cells
    I = P_total / (V_oc + 1e-6)
    I = np.clip(I, -150, 200)

    Q_gen = I**2 * R_int
    Q_cool = h*A_th*(T - T_amb)

    T += (Q_gen - Q_cool)/(mass_batt*Cp) * dt

    soc_val -= (P_total * dt / 3600) / battery_capacity_wh
    soc_val = np.clip(soc_val, 0, 1)

    dist += v * dt / 1000

    soc.append(soc_val)
    distance.append(dist)
    battery_temp.append(T)
    power_log.append(P_motor)
    speed_log.append(v)

    t += dt

# ======================================================
# ARRAYS
# ======================================================

time = np.arange(0, len(soc)*dt, dt)
soc = np.array(soc)
distance = np.array(distance)
battery_temp = np.array(battery_temp)
power_log = np.array(power_log)
speed_log = np.array(speed_log)

# ======================================================
# SAVE PLOTS
# ======================================================

def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.savefig(filename, dpi=300)
    plt.close()

save_plot(time, speed_log*3.6, "Time (s)", "Speed (km/h)", "Driving Cycle", os.path.join(OUT, "speed.png"))
save_plot(time, power_log/1000, "Time (s)", "Power (kW)", "Motor Power", os.path.join(OUT, "motor_power.png"))
save_plot(distance, soc*100, "Distance (km)", "SOC (%)", "SOC vs Distance", os.path.join(OUT, "soc.png"))
save_plot(time, battery_temp-273, "Time (s)", "Temp (°C)", "Battery Temperature", os.path.join(OUT, "temp.png"))

# ======================================================
# SUMMARY
# ======================================================

used_kwh = battery_capacity_kwh * (1 - soc[-1])
print(f"Range to {soc[-1]*100:.0f}% SOC: {distance[-1]:.1f} km")
print(f"Specific consumption: {used_kwh*1000/distance[-1]:.1f} Wh/km")
print(f"Peak pack temperature: {battery_temp.max()-273:.1f} C")
print("Figures saved to", os.path.abspath(OUT))
