import pandas as pd
import numpy as np
import os
import cmath


class file_reading:
    def __init__(self, file_loc):
        self.excel_sheet_df = pd.read_csv(file_loc)

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

# Reading feeder data
feeder_file_reading = file_reading('/Users/shubham/Desktop/Test feeder Data-Table 1-1.csv')
feeder_df = feeder_file_reading.open_file()
print(feeder_df)
# subtracting 1 since in Ymatrix will have 0 based indexing for python
from_bus_col_feeder = feeder_file_reading.to_numpy('From Bus') - 1
to_bus_col_feeder = feeder_file_reading.to_numpy('To Bus') - 1
resistance_bw_frm_and_2_bus_feeder = feeder_file_reading.to_numpy('Resistance')
x_bw_frm_and_2_bus_feeder = feeder_file_reading.to_numpy('Reactance')
chrgng_admtnc_bw_frm_and_2_bus = feeder_file_reading.to_numpy('Charging Admittance')
chrgng_admtnc_bw_frm_and_2_bus = chrgng_admtnc_bw_frm_and_2_bus / 2

class convert_to_polar:
    def __init__(self, R_vector, X_vector):
        self.R = R_vector
        self.X = X_vector

    def polar_form(self):
        Y_vector = np.empty(len(self.R), dtype=complex)
        for i in range(len(self.R)):
            Y_vector[i] = 1 / complex(self.R[i], self.X[i])
        return Y_vector

# getting_y_vector and charging admmitance vector for feeder
Y_vector_obj = convert_to_polar(resistance_bw_frm_and_2_bus_feeder, x_bw_frm_and_2_bus_feeder)
Y_vector_feeder = Y_vector_obj.polar_form()
# # inverting charging admmitance since we are sending impedance to covert_to_polar function
# chrgng_admtnc_feedr_obj = convert_to_polar(np.zeros(len(from_bus_col_feeder)),
#                                            1 / chrgng_admtnc_bw_frm_and_2_bus)
# chrgng_admtnc_feedr = chrgng_admtnc_feedr_obj.polar_form()

def form_ybus_with_feeder_data(no_of_bus, from_bus_vector, to_bus_vector, Y_vector):
    Y_bus_matrix = np.zeros((no_of_bus, no_of_bus), dtype=complex)
    for i in range(len(to_bus_vector)):
        Y_bus_matrix[from_bus_vector[i]][from_bus_vector[i]] += Y_vector[i]
        Y_bus_matrix[to_bus_vector[i]][to_bus_vector[i]] += Y_vector[i]
        Y_bus_matrix[from_bus_vector[i]][to_bus_vector[i]] -= Y_vector[i]
        Y_bus_matrix[to_bus_vector[i]][from_bus_vector[i]] -= Y_vector[i]
    return Y_bus_matrix


Y_bus_matrix = form_ybus_with_feeder_data(3, from_bus_col_feeder, to_bus_col_feeder
                                          , Y_vector_feeder)

print(Y_bus_matrix)

def form_b1_matrix(Y_bus_matrix, slack_bus_position):
    b1 = np.delete(Y_bus_matrix, slack_bus_position, 0)
    b1_modified = np.delete(b1, slack_bus_position, 1)
    b1_modified = cmath.sqrt(-1) * b1_modified.imag
    return b1_modified


b1 = form_b1_matrix(Y_bus_matrix, 0)
print(b1)

def form_b2_matrix(Y_bus_matrix, bus_type_vector, no_of_PQ_buses):
    b2_matrix = np.zeros((no_of_PQ_buses, no_of_PQ_buses), dtype=complex)
    PQ_buses_position = []
    for i in range(len(bus_type_vector)):
        if (bus_type_vector[i] == 3):
            # remember wrt 0 based indexing
            PQ_buses_position.append(i)
    for row in range(len(b2_matrix)):
        for col in range(len(b2_matrix)):
            b2_matrix[row][col] = Y_bus_matrix[PQ_buses_position[row]][PQ_buses_position[col]]
    return b2_matrix


b2_matrix = form_b2_matrix(Y_bus_matrix, type, 1)


# # Reading transformer data
# transf_file_reading = file_reading('/Users/shubham/Desktop/IEEE9 2/Sheet 1-Table 1.csv')
# from_bus_col_transf = transf_file_reading.to_numpy('From Bus') - 1
# to_bus_col_transf = transf_file_reading.to_numpy('To Bus') - 1
# transf_reactance = transf_file_reading.to_numpy('Reactance')
# transf_resistance = transf_file_reading.to_numpy('Resistance')
# off_nomnl_tap_ratio = transf_file_reading.to_numpy('Off-nominal Tap Ratio')
#
# Y_vector_obj_transf = convert_to_polar(transf_resistance, transf_reactance)
# Y_vector_transf = Y_vector_obj_transf.polar_form()
#
# def modify_ybus_with_transf_reactance(from_bus_vec, to_bus_vec,
#                                       Y_vector, off_nomnl_tap_vec):
#     Y_bus_matrix = np.zeros((2, 2), dtype=complex)
#     for i in range(len(from_bus_vec)):
#         Y_bus_matrix[from_bus_vec[i]][from_bus_vec[i]] += Y_vector[i] / (off_nomnl_tap_vec[i] ** 2)
#         Y_bus_matrix[from_bus_vec[i]][to_bus_vec[i]] -= Y_vector[i] / off_nomnl_tap_ratio[i]
#         Y_bus_matrix[to_bus_vec[i]][from_bus_vec[i]] -= Y_vector[i] / off_nomnl_tap_ratio[i]
#         Y_bus_matrix[to_bus_vec[i]][to_bus_vec[i]] += Y_vector[i]
#     return Y_bus_matrix
#
# Y_bus_matrix_1 = modify_ybus_with_transf_reactance(from_bus_col_transf
#                                                  , to_bus_col_transf, Y_vector_transf
#                                                  , off_nomnl_tap_ratio)
# print(Y_bus_matrix_1)
