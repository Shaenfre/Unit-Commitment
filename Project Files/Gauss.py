import pandas as pd
import numpy as np
import os
import cmath, math


class file_reading:
    def __init__(self, file_name, sheet_name):
        self.excel_sheet_df = pd.read_excel(file_name, sheet_name)

    def open_file(self):
        return self.excel_sheet_df

    # convert a particular column to a numpy array
    def to_numpy(self, col_name):
        return self.excel_sheet_df[col_name].to_numpy()

    # convert all columns into numpy array and store them in a dictionary with column name as the key
    def to_numpy_all(self):
        datasheet_dict = {}
        for col in self.excel_sheet_df.columns:
            datasheet_dict[col] = self.excel_sheet_df[col].to_numpy()
        return datasheet_dict


os.chdir('/Users/shubham/Desktop/Codes/finalYrproj/Project Files')

# Reading bus data
bus_fileReading = file_reading('IEEE3.xlsx', 'Bus Data')
bus_datasheet = bus_fileReading.open_file()

# converting element by element to numpy array
Pg = bus_fileReading.to_numpy('Pg')
Qg = bus_fileReading.to_numpy('Qg')
Pd = bus_fileReading.to_numpy('Pd')
Qd = bus_fileReading.to_numpy('Qd')
type = bus_fileReading.to_numpy('Type')
print("type")
print(type)
no_of_buses = len(Pg)
slack_bus_position = np.where(type == 1)

# print(Qg)

feeder_file_reading = file_reading('IEEE3.xlsx', 'Feeder Data')
feeder_datasheet = feeder_file_reading.open_file()
# print(list(feeder_datasheet.columns))
# subtracting 1 since in Ymatrix will have 0 based indexing for python
from_bus_col_feeder = feeder_file_reading.to_numpy('From Bus') - 1
to_bus_col_feeder = feeder_file_reading.to_numpy('To Bus') - 1
resistance_bw_frm_and_2_bus_feeder = feeder_file_reading.to_numpy('Resistance')
x_bw_frm_and_2_bus_feeder = feeder_file_reading.to_numpy('Reactance')
chrgng_admtnc_bw_frm_and_2_bus = feeder_file_reading.to_numpy('Charging  susceptance')
chrgng_admtnc_bw_frm_and_2_bus = chrgng_admtnc_bw_frm_and_2_bus / 2

# Reading general info datasheet for base mva value
general_info_reading = file_reading('IEEE3.xlsx', 'General Info')
base_mva = general_info_reading.to_numpy('base mva')

# Reading PV bus data
PV_bus_reading = file_reading('IEEE3.xlsx', 'PV Bus Data')
specified_voltage_vector = PV_bus_reading.to_numpy('Specified Voltage')
# 0 based indexing
pv_bus_code_vector = PV_bus_reading.to_numpy('Bus Code') - 1
print("Pvbus")
print(pv_bus_code_vector)
P_min_vector = PV_bus_reading.to_numpy('Pmin')/base_mva
Q_min_vector = PV_bus_reading.to_numpy('Qmin')/base_mva
P_max_vector = PV_bus_reading.to_numpy('Pmax')/base_mva
Q_max_vector = PV_bus_reading.to_numpy('Qmax')/base_mva
pv_bus_sp_volt = PV_bus_reading.to_numpy('Specified Voltage')

#Reading slack bus data
slack_bus_reading = file_reading('IEEE3.xlsx', 'Slack Bus Data')
# 0 based indexing
slack_bus_code = slack_bus_reading.to_numpy('Bus Code') - 1
slack_bus_spcfd_volt = slack_bus_reading.to_numpy('Specified Voltage')


class convert_to_polar:
    def __init__(self, R_vector, X_vector):
        self.R = R_vector
        self.X = X_vector

    def polar_form(self):
        Y_vector = np.empty(len(self.R), dtype=complex)
        for i in range(len(self.R)):
            Y_vector[i] = 1 / complex(self.R[i], self.X[i])
        return Y_vector


# getting y_vector and charging admmitance vector for feeder
Y_vector_obj = convert_to_polar(resistance_bw_frm_and_2_bus_feeder, x_bw_frm_and_2_bus_feeder)
Y_vector_feeder = Y_vector_obj.polar_form()
# inverting charging admmitance since we are sending impedance to covert_to_polar function
chrgng_admtnc_feedr_obj = convert_to_polar(np.zeros(len(from_bus_col_feeder)),
                                           1 / chrgng_admtnc_bw_frm_and_2_bus)
chrgng_admtnc_feedr = chrgng_admtnc_feedr_obj.polar_form()


def form_ybus_with_feeder_data(no_of_bus, from_bus_vector, to_bus_vector, Y_vector, charging_admt_vector):
    Y_bus_matrix = np.zeros((no_of_bus, no_of_bus), dtype=complex)
    for i in range(len(to_bus_vector)):
        Y_bus_matrix[from_bus_vector[i]][from_bus_vector[i]] += Y_vector[i] + charging_admt_vector[i]
        Y_bus_matrix[to_bus_vector[i]][to_bus_vector[i]] += Y_vector[i] + charging_admt_vector[i]
        Y_bus_matrix[from_bus_vector[i]][to_bus_vector[i]] -= Y_vector[i]
        Y_bus_matrix[to_bus_vector[i]][from_bus_vector[i]] -= Y_vector[i]
    return Y_bus_matrix


Y_bus_matrix = form_ybus_with_feeder_data(no_of_buses, from_bus_col_feeder, to_bus_col_feeder
                                          , Y_vector_feeder, chrgng_admtnc_feedr)
# print(Y_bus_matrix)

delta_vec = np.zeros((no_of_buses))
voltage_vec = np.zeros((no_of_buses))

for i in range(len(voltage_vec)):
    if i in slack_bus_code:
        voltage_vec[i] = slack_bus_spcfd_volt
    elif i in pv_bus_code_vector:
        index_of_pv_bus = np.where(pv_bus_code_vector == i)
        voltage_vec[i] = pv_bus_sp_volt[index_of_pv_bus]
    else:
        voltage_vec[i] = 1


def form_P_specfd_vector(Pg, Pd, type, no_of_bus, base_mva):
    P_specfd_vec = np.zeros(no_of_bus)
    for i in range(no_of_bus):
        P_specfd_vec[i] = Pg[i] - Pd[i]
    # finding index of slack bus
    slack_bus_index = np.where(type == 1)
    P_specfd_vec = np.delete(P_specfd_vec, slack_bus_index)
    P_specfd_vec_mod = P_specfd_vec / base_mva
    return P_specfd_vec_mod

P_specified = form_P_specfd_vector(Pg, Pd, type, no_of_buses, base_mva)
print(P_specified)

# counting the number of PQ buses
unique, count_of_buses = np.unique(type, return_counts=True)
count_dict = dict(zip(unique, count_of_buses))
no_of_PQ_buses = count_dict[3]

PQ_bus_index_list = np.where(type == 3)[0]
print("Pqbusindx")
print(PQ_bus_index_list)

def form_Q_spcfd_vec(Qg, Qd, no_of_PQ_bus, PQ_bus_index_list, base_mva):
    Q_specfd_vec = np.zeros(no_of_PQ_bus)
    for i in range(no_of_PQ_bus):
        PQ_bus_index = PQ_bus_index_list[i]
        Q_specfd_vec[i] = Qg[PQ_bus_index] - Qd[PQ_bus_index]
    Q_specfd_vec_mod = Q_specfd_vec / base_mva
    return Q_specfd_vec_mod

Q_specfd_vec = form_Q_spcfd_vec(Qg, Qd, no_of_PQ_buses,PQ_bus_index_list, base_mva)
print(Q_specfd_vec)

def form_complex_voltage(Voltage_mag, Phase):
    real = Voltage_mag * math.cos(Phase)
    imag = Voltage_mag * math.sin(Phase)
    z = complex(real, imag)
    return z

V = np.ndarray(no_of_buses, dtype = complex)

for i in range(len(voltage_vec)):
    if i in slack_bus_code:
        V[i] = form_complex_voltage(slack_bus_spcfd_volt, 0)
    elif i in pv_bus_code_vector:
        index_of_pv_bus = np.where(pv_bus_code_vector == i)
        V[i] = form_complex_voltage(pv_bus_sp_volt[index_of_pv_bus], 0)
    else:
        V[i] = form_complex_voltage(1, 0)

print("V")
print(V)

V_old = np.ndarray(no_of_buses, dtype = complex)
V_new = np.ndarray(no_of_buses, dtype = complex)

#Doing the iterations
itr_max = 1
for i in range(itr_max):
    for j in range(no_of_buses):
        if (type[i] == 3):
            V_old[i] = V[i]
            A = (P_specified[i] - j * Q_specfd_vec[i]) / np.conj(V[i])

            B = complex(0, 0)
            



