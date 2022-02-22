import pandas as pd
import numpy as np
import os
import cmath, math
from tabulate import tabulate


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


os.chdir('Project Files')

# Reading general info datasheet for base mva value
general_info_reading = file_reading('IEEE9.xlsx', 'General Info')
base_mva = general_info_reading.to_numpy('base mva')
no_bus = general_info_reading.to_numpy('no of bus')[0]
# print("no of bus " + str(no_bus))
no_fdr = general_info_reading.to_numpy('no of feeder')[0]
no_tr = general_info_reading.to_numpy('no of transformer')[0]
no_pv_bus = general_info_reading.to_numpy('no of pv bus')[0]

# Reading bus data
bus_fileReading = file_reading('IEEE9.xlsx', 'Bus Data')
bus_datasheet = bus_fileReading.open_file()

# converting element by element to numpy array
Pg = bus_fileReading.to_numpy('Pg')
Qg = bus_fileReading.to_numpy('Qg')
Pd = bus_fileReading.to_numpy('Pd')
Qd = bus_fileReading.to_numpy('Qd')
type = bus_fileReading.to_numpy('Type')
slack_bus_position = np.where(type == 1)
# print(slack_bus_position)

# Reading feeder data
feeder_file_reading = file_reading('IEEE9.xlsx', 'Feeder Data')
# subtracting 1 since in Ymatrix will have 0 based indexing for python
from_bus_col_feeder = feeder_file_reading.to_numpy('From Bus') - 1
to_bus_col_feeder = feeder_file_reading.to_numpy('To Bus') - 1
resistance_bw_frm_and_2_bus_feeder = feeder_file_reading.to_numpy('Resistance')
x_bw_frm_and_2_bus_feeder = feeder_file_reading.to_numpy('Reactance')
chrgng_admtnc_bw_frm_and_2_bus = feeder_file_reading.to_numpy('Charging Admittance')
# chrgng_admtnc_bw_frm_and_2_bus = chrgng_admtnc_bw_frm_and_2_bus / 2

# Reading transformer data
transf_file_reading = file_reading('IEEE9.xlsx', 'Transformer Data')
from_bus_col_transf = transf_file_reading.to_numpy('From Bus') - 1
to_bus_col_transf = transf_file_reading.to_numpy('To Bus') - 1
transf_reactance = transf_file_reading.to_numpy('Reactance')
transf_resistance = transf_file_reading.to_numpy('Resistance')
off_nomnl_tap_ratio = transf_file_reading.to_numpy('Off-nominal Tap Ratio')


class convert_to_polar:
    def __init__(self, R_vector, X_vector):
        self.R = R_vector
        self.X = X_vector

    def polar_form(self):
        Y_vector = np.zeros(len(self.R), dtype=complex)
        for i in range(len(self.R)):
            Y_vector[i] = 1 / complex(self.R[i], self.X[i])
        return Y_vector


# getting y_vector and charging admmitance vector for feeder
Y_vector_obj = convert_to_polar(resistance_bw_frm_and_2_bus_feeder, x_bw_frm_and_2_bus_feeder)
Y_vector_feeder = Y_vector_obj.polar_form()
# inverting charging admmitance since we are sending impedance to covert_to_polar function
chrgng_admtnc_feedr_obj = convert_to_polar(np.zeros(no_fdr),
                                           -2 / chrgng_admtnc_bw_frm_and_2_bus)
chrgng_admtnc_feedr = chrgng_admtnc_feedr_obj.polar_form()

# getting y_vector for transformer
Y_vector_obj_transf = convert_to_polar(transf_resistance, transf_reactance)
Y_vector_transf = Y_vector_obj_transf.polar_form()


def form_ybus_with_feeder_data():
    Y_bus_matrix = np.zeros((no_bus, no_bus), dtype=complex)
    for i in range(no_fdr):
        n = from_bus_col_feeder[i]
        m = to_bus_col_feeder[i]

        Y_bus_matrix[n][n] += Y_vector_feeder[i] + chrgng_admtnc_feedr[i]
        Y_bus_matrix[m][m] += Y_vector_feeder[i] + chrgng_admtnc_feedr[i]
        Y_bus_matrix[n][m] -= Y_vector_feeder[i]
        Y_bus_matrix[m][n] -= Y_vector_feeder[i]
    return Y_bus_matrix


Y_bus_matrix = form_ybus_with_feeder_data()


# print(Y_bus_matrix)

def modify_ybus_with_transf_reactance():
    Y_bus_mat = Y_bus_matrix
    for i in range(no_tr):
        n = from_bus_col_transf[i]
        m = to_bus_col_transf[i]

        Y_bus_mat[n, n] += off_nomnl_tap_ratio[i] * np.conj(off_nomnl_tap_ratio[i]) * Y_vector_transf[i]
        Y_bus_mat[n, m] -= np.conj(off_nomnl_tap_ratio[i]) * Y_vector_transf[i]
        Y_bus_mat[m, n] -= off_nomnl_tap_ratio[i] * Y_vector_transf[i]
        Y_bus_mat[m, m] += Y_vector_transf[i]
    return Y_bus_mat


if no_tr > 0:
    Y_bus_matrix = modify_ybus_with_transf_reactance()


# print(Y_bus_matrix)

def form_b1_matrix():
    b1 = np.delete(Y_bus_matrix, slack_bus_position, 0)
    b1_modified = np.delete(b1, slack_bus_position, 1)
    b1_modified = b1_modified.imag
    return b1_modified


b1 = form_b1_matrix()
b1_inv = np.linalg.inv(b1)

# print(b1)
# counting the number of PQ buses
unique, count_of_buses = np.unique(type, return_counts=True)
count_dict = dict(zip(unique, count_of_buses))
no_of_PQ_buses = count_dict[3]


def form_b2_matrix():
    b2_matrix = np.zeros((no_of_PQ_buses, no_of_PQ_buses), dtype=complex)
    PQ_buses_position = []
    for i in range(len(type)):
        if (type[i] == 3):
            # remember wrt 0 based indexing
            PQ_buses_position.append(i)
    for row in range(len(b2_matrix)):
        for col in range(len(b2_matrix)):
            b2_matrix[row][col] = Y_bus_matrix[PQ_buses_position[row]][PQ_buses_position[col]]
    b2_matrix = b2_matrix.imag
    return b2_matrix


b2_matrix = form_b2_matrix()
b2_inv = np.linalg.inv(b2_matrix)
# print(b2_matrix)

P_injected = Pg - Pd
# normalizing P_injected vector
P_injected_normlzd = P_injected / base_mva
Q_injected = Qg - Qd
O_injected_normlzd = Q_injected / base_mva

# Reading PV bus data
PV_bus_reading = file_reading('IEEE9.xlsx', 'PV Bus Data')
specified_voltage_vector = PV_bus_reading.to_numpy('Specified Voltage')
# 0 based indexing
pv_bus_code_vector = PV_bus_reading.to_numpy('Bus Code') - 1
P_min_vector = PV_bus_reading.to_numpy('Pmin') / base_mva
Q_min_vector = PV_bus_reading.to_numpy('Qmin') / base_mva
P_max_vector = PV_bus_reading.to_numpy('Pmax') / base_mva
Q_max_vector = PV_bus_reading.to_numpy('Qmax') / base_mva
pv_bus_sp_volt = PV_bus_reading.to_numpy('Specified Voltage')

# Reading slack bus data
slack_bus_reading = file_reading('IEEE9.xlsx', 'Slack Bus Data')
# 0 based indexing
slack_bus_code = slack_bus_reading.to_numpy('Bus Code') - 1
slack_bus_spcfd_volt = slack_bus_reading.to_numpy('Specified Voltage')


# initializing delta vector
# initializing voltage vector
def initialze_voltage():
    delta_vec = np.zeros((no_bus))
    voltage_vec = np.empty((no_bus))
    for i in range(no_bus):
        if i in slack_bus_code:
            voltage_vec[i] = slack_bus_spcfd_volt
        elif i in pv_bus_code_vector:
            index_of_pv_bus = np.where(pv_bus_code_vector == i)
            voltage_vec[i] = pv_bus_sp_volt[index_of_pv_bus]
        else:
            voltage_vec[i] = 1
    return voltage_vec, delta_vec


voltage_vec, delta_vec = initialze_voltage()
# print(voltage_vec)

G_matrix = Y_bus_matrix.real
# print(G_matrix)
B_matrix = Y_bus_matrix.imag


# print(B_matrix)

def calc_active_p_vector():
    no_of_bus = len(G_matrix)
    active_power_vec = np.zeros((no_of_bus), dtype=float)
    for i in range(no_of_bus):
        for k in range(no_of_bus):
            if i != slack_bus_code:
                cos_term = G_matrix[i][k] * math.cos(delta_vec[i] - delta_vec[k])
                sin_term = B_matrix[i][k] * math.sin(delta_vec[i] - delta_vec[k])
                summation_term = voltage_vec[k] * (cos_term + sin_term)
                active_power_vec[i] += voltage_vec[i] * summation_term
    active_power_vec = np.delete(active_power_vec, slack_bus_code)
    return active_power_vec


def form_P_specfd_vector():
    P_specfd_vec = np.zeros(no_bus)
    for i in range(no_bus):
        P_specfd_vec[i] = Pg[i] - Pd[i]
    # finding index of slack bus
    slack_bus_index = np.where(type == 1)
    P_specfd_vec = np.delete(P_specfd_vec, slack_bus_index)
    P_specfd_vec_mod = P_specfd_vec / base_mva
    return P_specfd_vec_mod


P_specified = form_P_specfd_vector()


def form_delta_P_by_V_vector(delta_P_vec):
    slack_bus_index = np.where(type == 1)
    new_voltage_vect = np.delete(voltage_vec, slack_bus_index)
    del_P_by_V_vec = delta_P_vec / new_voltage_vect
    return del_P_by_V_vec


PQ_bus_index_list = np.where(type == 3)[0]
# print(PQ_bus_index_list)

delta_V_vector = np.zeros(no_of_PQ_buses)


# print(delta_V_vector)

def form_Q_spcfd_vec():
    Q_specfd_vec = np.zeros(no_of_PQ_buses)
    for i in range(no_of_PQ_buses):
        PQ_bus_index = PQ_bus_index_list[i]
        Q_specfd_vec[i] = Qg[PQ_bus_index] - Qd[PQ_bus_index]
    Q_specfd_vec_mod = Q_specfd_vec / base_mva
    return Q_specfd_vec_mod


Q_specfd_vec = form_Q_spcfd_vec()


# print(Q_specfd_vec)

def calc_reactive_Q_vector():
    reactive_power_vec = np.zeros(no_bus, dtype=object)
    for i in range(no_bus):
        for k in range(no_bus):
            if i in PQ_bus_index_list:
                cos_term = G_matrix[i][k] * math.sin(delta_vec[i] - delta_vec[k])
                sin_term = B_matrix[i][k] * math.cos(delta_vec[i] - delta_vec[k])
                summation_term = voltage_vec[k] * (cos_term - sin_term)
                reactive_power_vec[i] += voltage_vec[i] * summation_term
    reactive_power_vec = reactive_power_vec[reactive_power_vec != 0]
    return reactive_power_vec


# print(reactive_Q_vec)

def form_delta_Q_by_V_vector(voltage_vec, delta_Q_vec):
    voltage_vec = voltage_vec[PQ_bus_index_list]
    del_Q_by_V_vec = delta_Q_vec / voltage_vec
    return del_Q_by_V_vec


def perform_Nrlf():
    no_of_iter = 40
    epsilon_p = 10 ** (-5)
    epsilon_q = 10 ** (-5)
    cnvrgd_at = -1
    for i in range(no_of_iter):

        P_calc = calc_active_p_vector()
        delta_P_vec = P_specified - P_calc
        del_P_by_V_vec = form_delta_P_by_V_vector(delta_P_vec)
        del_delta_vector = - np.dot(b1_inv, del_P_by_V_vec)
        max_del_P = max(delta_P_vec)

        if max_del_P > epsilon_p:

            # modifying the delta vector
            delta_vec_col_no = 0
            del_delta_vec_col_no = 0

            while (delta_vec_col_no < len(delta_vec)):

                if delta_vec_col_no != slack_bus_code:
                    delta_vec[delta_vec_col_no] += del_delta_vector[del_delta_vec_col_no]
                    delta_vec_col_no += 1
                    del_delta_vec_col_no += 1
                else:
                    delta_vec_col_no += 1

        # iteration for modifiying the V vector
        Q_calc = calc_reactive_Q_vector()
        delta_Q_vec = Q_specfd_vec - Q_calc
        del_Q_by_V_vec = form_delta_Q_by_V_vector(voltage_vec, delta_Q_vec)
        del_V_vec = - np.dot(b2_inv, del_Q_by_V_vec)
        max_del_Q = abs(max(np.amax(delta_Q_vec), np.amin(delta_Q_vec), key=abs))

        if max_del_Q > epsilon_q:

            voltage_vec_col_no = 0
            del_V_col_no = 0

            while (voltage_vec_col_no < len(voltage_vec)):
                if voltage_vec_col_no in PQ_bus_index_list:
                    voltage_vec[voltage_vec_col_no] += del_V_vec[del_V_col_no]
                    voltage_vec_col_no += 1
                    del_V_col_no += 1
                else:
                    voltage_vec_col_no += 1

        # checking if the load flow converged
        if max_del_P < epsilon_p and max_del_Q < epsilon_q:
            cnvrgd_at = i
            break
    return cnvrgd_at


def form_cmplx_vltg():
    cmplx_vltg = np.zeros(no_bus, dtype=complex)
    for i in range(no_bus):
        v_real = voltage_vec[i] * math.cos(delta_vec[i])
        v_imag = voltage_vec[i] * math.sin(delta_vec[i])
        cmplx_vltg[i] = complex(v_real, v_imag)
    return cmplx_vltg


def calc_i_from_feder():
    fdr_I = np.zeros((no_fdr, 2), dtype=complex)
    for k in range(no_fdr):
        i = from_bus_col_feeder[k]
        j = to_bus_col_feeder[k]
        fdr_I[k, 0] = (cmplx_vltg_vec[i] - cmplx_vltg_vec[j]) * (-Y_bus_matrix[i, j]) + \
                      cmplx_vltg_vec[i] * chrgng_admtnc_feedr[k]
        fdr_I[k, 1] = (cmplx_vltg_vec[j] - cmplx_vltg_vec[i]) * (-Y_bus_matrix[j, i]) + \
                      cmplx_vltg_vec[i] * chrgng_admtnc_feedr[k]

    return fdr_I


def calc_pwr_from_fdr():
    fd_S = np.zeros((no_fdr, 2), dtype=complex)
    fd_S_max_case1 = np.zeros(no_fdr, dtype=complex)
    for k in range(no_fdr):
        i = from_bus_col_feeder[k]
        j = to_bus_col_feeder[k]
        fd_S[k, 0] = voltage_vec[i] * np.conj(fdr_I[k, 0])
        fd_S[k, 1] = voltage_vec[i] * np.conj(fdr_I[k, 1])
        fd_S_max_case1[k] = max(fd_S[k, 0], fd_S[k, 1]) * 1.5
    return fd_S, fd_S_max_case1


def calc_fdr_loss():
    fd_Sloss = np.zeros(no_fdr, dtype=complex)
    for k in range(no_fdr):
        fd_Sloss[k] = fd_S[k, 0] + fd_S[k, 1]
    return fd_Sloss


def calc_i_frm_trans():
    tr_I = np.zeros((no_tr, 2), dtype=complex)
    for k in range(no_tr):
        i = from_bus_col_transf[k]
        j = to_bus_col_transf[k]

        tr_I[k, 0] = voltage_vec[i] * off_nomnl_tap_ratio[k] * np.conj(off_nomnl_tap_ratio[k]) * Y_vector_transf[k] - \
                     voltage_vec[j] * np.conj(off_nomnl_tap_ratio[k]) * Y_vector_transf[k]
        tr_I[k, 1] = (-voltage_vec[i] * off_nomnl_tap_ratio[k] + voltage_vec[k]) * Y_vector_transf[k]

    return tr_I


def calc_pwr_frm_trans():
    tr_S = np.zeros((no_tr, 2), dtype=complex)
    for k in range(no_tr):
        i = from_bus_col_transf[k]
        j = to_bus_col_transf[k]

        tr_S[k, 0] = voltage_vec[i] * np.conj(tr_I[k, 0])
        tr_S[k, 1] = voltage_vec[j] * np.conj(tr_I[k, 1])

    return tr_S


def calc_transf_loss():
    tr_Sloss = np.zeros(no_tr, dtype=complex)
    for k in range(no_tr):
        tr_Sloss[k] = tr_S[k, 0] + tr_S[k, 1]
    return tr_Sloss


def print_res():
    bus_data = list(zip(voltage_vec, delta_vec))
    bus_df = pd.DataFrame(bus_data, columns=['Vm', 'delta degree'])
    print(bus_df)


cnvrgd_at = perform_Nrlf()

if (cnvrgd_at == -1):
    print("Load flow did not converged")
else:
    cmplx_vltg_vec = form_cmplx_vltg()
    # print(cmplx_vltg_vec)
    fdr_I = calc_i_from_feder()
    # print(fdr_I)
    fd_S, fd_S_max_case1 = calc_pwr_from_fdr()
    # print(fd_S)
    # feeder losses
    fd_Sloss = calc_fdr_loss()
    # print(fd_Sloss)

    if no_tr > 0:
        tr_I = calc_i_frm_trans()
        tr_S = calc_pwr_frm_trans()
        tr_Sloss = calc_transf_loss()
        # print(tr_Sloss)

    # finding Ig and Sd
    Ii = np.dot(Y_bus_matrix, voltage_vec)
    # print(Ii)
    Si = np.dot(voltage_vec, Ii)
    print(Si)

    print_res()

    # remaining to be done

    # contingency analysis
    for line_no in range(no_fdr):
        n = from_bus_col_feeder[line_no] - 1
        m = to_bus_col_feeder[line_no] - 1

        Y_bus_matrix[n][n] -= Y_vector_feeder[line_no] + chrgng_admtnc_feedr[line_no]
        Y_bus_matrix[m][m] -= Y_vector_feeder[line_no] + chrgng_admtnc_feedr[line_no]
        Y_bus_matrix[n][m] += Y_vector_feeder[line_no]
        Y_bus_matrix[m][n] += Y_vector_feeder[line_no]

        G_matrix = Y_bus_matrix.real
        # print(G_matrix)
        B_matrix = Y_bus_matrix.imag

        b1_matrix = form_b1_matrix()
        b1_inv = np.linalg.inv(b1_matrix)
        b2_matrix = form_b2_matrix()
        b2_inv = np.linalg.inv(b2_matrix)

        voltage_vec, delta_vec = initialze_voltage()

        perform_Nrlf()

        fdr_I = calc_i_from_feder()
        fd_S, fd_S_max_case2 = calc_pwr_from_fdr()

        if no_tr > 0:
            tr_I = calc_i_frm_trans()
            tr_S = calc_pwr_frm_trans()
            tr_Sloss = calc_transf_loss()
            # print(tr_Sloss)







