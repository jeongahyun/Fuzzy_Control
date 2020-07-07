import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Antecedent & Consequent objects
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 81, 1), 'humidity')
power = ctrl.Consequent(np.arange(0, 101, 1), 'power')

temperature.automf(names=['A1', 'A2', 'A3', 'A4', 'A5'])
humidity.automf(names=['B1', 'B2', 'B3', 'B4', 'B5'])
power.automf(names=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])

# Generate fuzzy membership functions
temperature['A1'] = fuzz.trapmf(temperature.universe, [0, 0, 15, 16])
temperature['A2'] = fuzz.trapmf(temperature.universe, [15, 16, 18, 22])
temperature['A3'] = fuzz.trapmf(temperature.universe, [18, 22, 25, 26])
temperature['A4'] = fuzz.trapmf(temperature.universe, [25, 26, 30, 33])
temperature['A5'] = fuzz.trapmf(temperature.universe, [30, 33, 40, 40])

humidity['B1'] = fuzz.trapmf(humidity.universe, [0, 0, 15, 20])
humidity['B2'] = fuzz.trapmf(humidity.universe, [15, 20, 30, 35])
humidity['B3'] = fuzz.trapmf(humidity.universe, [30, 35, 40, 50])
humidity['B4'] = fuzz.trapmf(humidity.universe, [40, 50, 60, 70])
humidity['B5'] = fuzz.trapmf(humidity.universe, [60, 70, 80, 80])

power['C1'] = fuzz.trimf(power.universe, [0, 0, 15])
power['C2'] = fuzz.trimf(power.universe, [0, 15, 30])
power['C3'] = fuzz.trimf(power.universe, [15, 30, 50])
power['C4'] = fuzz.trimf(power.universe, [30, 50, 70])
power['C5'] = fuzz.trimf(power.universe, [50, 70, 85])
power['C6'] = fuzz.trimf(power.universe, [70, 85, 100])
power['C7'] = fuzz.trimf(power.universe, [85, 100, 100])

# Visualizing
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(temperature.universe, temperature['A1'].mf, 'b', linewidth=1.5, label="A1")
ax0.plot(temperature.universe, temperature['A2'].mf, 'g', linewidth=1.5, label="A2")
ax0.plot(temperature.universe, temperature['A3'].mf, 'r', linewidth=1.5, label="A3")
ax0.plot(temperature.universe, temperature['A4'].mf, 'y', linewidth=1.5, label="A4")
ax0.plot(temperature.universe, temperature['A5'].mf, 'blueviolet', linewidth=1.5, label="A5")
ax0.set_title('temperature')
ax0.legend()

ax1.plot(np.arange(0, 81, 1), humidity['B1'].mf, 'b', linewidth=1.5, label="B1")
ax1.plot(np.arange(0, 81, 1), humidity['B2'].mf, 'g', linewidth=1.5, label="B2")
ax1.plot(np.arange(0, 81, 1), humidity['B3'].mf, 'r', linewidth=1.5, label="B3")
ax1.plot(np.arange(0, 81, 1), humidity['B4'].mf, 'y', linewidth=1.5, label="B4")
ax1.plot(np.arange(0, 81, 1), humidity['B5'].mf, 'blueviolet', linewidth=1.5, label="B5")
ax1.set_title('humidity')
ax1.legend()

ax2.plot(np.arange(0, 101, 1), power['C1'].mf, 'b', linewidth=1.5, label="C1")
ax2.plot(np.arange(0, 101, 1), power['C2'].mf, 'g', linewidth=1.5, label="C2")
ax2.plot(np.arange(0, 101, 1), power['C3'].mf, 'r', linewidth=1.5, label="C3")
ax2.plot(np.arange(0, 101, 1), power['C4'].mf, 'y', linewidth=1.5, label="C1")
ax2.plot(np.arange(0, 101, 1), power['C5'].mf, 'blueviolet', linewidth=1.5, label="C2")
ax2.plot(np.arange(0, 101, 1), power['C6'].mf, 'orange', linewidth=1.5, label="C3")
ax2.plot(np.arange(0, 101, 1), power['C7'].mf, 'turquoise', linewidth=1.5, label="C3")
ax2.set_title('power')
ax2.legend()

for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
plt.tight_layout()
plt.savefig('./fig1.png')
plt.show()

temp = int(input('온도를 입력하세요:'))
hum = int(input('습도를 입력하세요:'))

# Mamdani
temp_level_A1 = fuzz.interp_membership(temperature.universe, temperature['A1'].mf, temp)
temp_level_A2 = fuzz.interp_membership(temperature.universe, temperature['A2'].mf, temp)
temp_level_A3 = fuzz.interp_membership(temperature.universe, temperature['A3'].mf, temp)
temp_level_A4 = fuzz.interp_membership(temperature.universe, temperature['A4'].mf, temp)
temp_level_A5 = fuzz.interp_membership(temperature.universe, temperature['A5'].mf, temp)

hum_level_B1 = fuzz.interp_membership(humidity.universe, humidity['B1'].mf, hum)
hum_level_B2 = fuzz.interp_membership(humidity.universe, humidity['B2'].mf, hum)
hum_level_B3 = fuzz.interp_membership(humidity.universe, humidity['B3'].mf, hum)
hum_level_B4 = fuzz.interp_membership(humidity.universe, humidity['B4'].mf, hum)
hum_level_B5 = fuzz.interp_membership(humidity.universe, humidity['B5'].mf, hum)

active_rule1 = np.fmin(temp_level_A1, hum_level_B1)
active_rule2 = np.fmin(temp_level_A1, hum_level_B2)
active_rule3 = np.fmin(temp_level_A1, hum_level_B3)
active_rule4 = np.fmin(temp_level_A1, hum_level_B4)
active_rule5 = np.fmin(temp_level_A1, hum_level_B5)
active_rule6 = np.fmin(temp_level_A2, hum_level_B1)
active_rule7 = np.fmin(temp_level_A2, hum_level_B2)
active_rule8 = np.fmin(temp_level_A2, hum_level_B3)
active_rule9 = np.fmin(temp_level_A2, hum_level_B4)
active_rule10 = np.fmin(temp_level_A2, hum_level_B5)
active_rule11 = np.fmin(temp_level_A3, hum_level_B1)
active_rule12 = np.fmin(temp_level_A3, hum_level_B2)
active_rule13 = np.fmin(temp_level_A3, hum_level_B3)
active_rule14 = np.fmin(temp_level_A3, hum_level_B4)
active_rule15 = np.fmin(temp_level_A3, hum_level_B5)
active_rule16 = np.fmin(temp_level_A4, hum_level_B1)
active_rule17 = np.fmin(temp_level_A4, hum_level_B2)
active_rule18 = np.fmin(temp_level_A4, hum_level_B3)
active_rule19 = np.fmin(temp_level_A4, hum_level_B4)
active_rule20 = np.fmin(temp_level_A4, hum_level_B5)
active_rule21 = np.fmin(temp_level_A5, hum_level_B1)
active_rule22 = np.fmin(temp_level_A5, hum_level_B2)
active_rule23 = np.fmin(temp_level_A5, hum_level_B3)
active_rule24 = np.fmin(temp_level_A5, hum_level_B4)
active_rule25 = np.fmin(temp_level_A5, hum_level_B5)

active_rule_c1 = np.fmax(active_rule1, np.fmax(active_rule2, np.fmax(active_rule3, np.fmax(active_rule6, active_rule7))))
power_activation_c1 = np.fmin(active_rule_c1, power['C1'].mf)

active_rule_c2 = np.fmax(active_rule4, active_rule8)
power_activation_c2 = np.fmin(active_rule_c2, power['C2'].mf)

active_rule_c3 = np.fmax(active_rule5, np.fmax(active_rule9, np.fmax(active_rule11, active_rule12)))
power_activation_c3 = np.fmin(active_rule_c3, power['C3'].mf)

active_rule_c4 = np.fmax(active_rule10, np.fmax(active_rule13, active_rule14))
power_activation_c4 = np.fmin(active_rule_c4, power['C4'].mf)

active_rule_c5 = np.fmax(active_rule15, np.fmax(active_rule16, active_rule17))
power_activation_c5 = np.fmin(active_rule_c5, power['C5'].mf)

active_rule_c6 = np.fmax(active_rule18, np.fmax(active_rule19, np.fmax(active_rule21, active_rule22)))
power_activation_c6 = np.fmin(active_rule_c6, power['C6'].mf)

active_rule_c7 = np.fmax(active_rule20, np.fmax(active_rule23, np.fmax(active_rule24, active_rule25)))
power_activation_c7 = np.fmin(active_rule_c7, power['C7'].mf)

power0 = np.zeros_like(power.universe)

# Visualizing
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(power.universe, power0, power_activation_c1, facecolor='b', alpha=0.7)
ax0.plot(power.universe, power['C1'].mf, 'b', linewidth=0.5, linestyle='--')
ax0.fill_between(power.universe, power0, power_activation_c2, facecolor='g', alpha=0.7)
ax0.plot(power.universe, power['C2'].mf, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(power.universe, power0, power_activation_c3, facecolor='r', alpha=0.7)
ax0.plot(power.universe, power['C3'].mf, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(power.universe, power0, power_activation_c4, facecolor='y', alpha=0.7)
ax0.plot(power.universe, power['C4'].mf, 'y', linewidth=0.5, linestyle='--')
ax0.fill_between(power.universe, power0, power_activation_c5, facecolor='blueviolet', alpha=0.7)
ax0.plot(power.universe, power['C5'].mf, 'blueviolet', linewidth=0.5, linestyle='--')
ax0.fill_between(power.universe, power0, power_activation_c6, facecolor='orange', alpha=0.7)
ax0.plot(power.universe, power['C6'].mf, 'orange', linewidth=0.5, linestyle='--')
ax0.fill_between(power.universe, power0, power_activation_c7, facecolor='turquoise', alpha=0.7)
ax0.plot(power.universe, power['C7'].mf, 'turquoise', linewidth=0.5, linestyle='--')
ax0.set_title("Output membership activity")

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
plt.tight_layout()
plt.savefig('./fig2.png')
plt.show()

# Defuzzification(centroid)
aggregated = np.fmax(power_activation_c1,
                     np.fmax(power_activation_c2,
                             np.fmax(power_activation_c3,
                                     np.fmax(power_activation_c4,
                                             np.fmax(power_activation_c5,
                                                     np.fmax(power_activation_c5,
                                                             np.fmax(power_activation_c6,
                                                                     power_activation_c7)))))))

output_power = fuzz.defuzz(power.universe, aggregated, 'centroid')
output_power_plot = fuzz.interp_membership(power.universe, aggregated, output_power)

# Visualizing
fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(power.universe, power['C1'].mf, 'b', linewidth=0.5, linestyle='--')
ax0.plot(power.universe, power['C2'].mf, 'g', linewidth=0.5, linestyle='--')
ax0.plot(power.universe, power['C3'].mf, 'r', linewidth=0.5, linestyle='--')
ax0.plot(power.universe, power['C4'].mf, 'y', linewidth=0.5, linestyle='--')
ax0.plot(power.universe, power['C5'].mf, 'blueviolet', linewidth=0.5, linestyle='--')
ax0.plot(power.universe, power['C6'].mf, 'orange', linewidth=0.5, linestyle='--')
ax0.plot(power.universe, power['C7'].mf, 'turquoise', linewidth=0.5, linestyle='--')
ax0.fill_between(power.universe, power0, aggregated, facecolor='cornflowerblue', alpha=0.7)
ax0.plot([output_power, output_power], [0, output_power_plot], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title("Aggregated membership and result (line)")

for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
plt.tight_layout()
plt.savefig('./fig3.png')
plt.show()
print('에어컨을 {}% 출력합니다.'.format(output_power))