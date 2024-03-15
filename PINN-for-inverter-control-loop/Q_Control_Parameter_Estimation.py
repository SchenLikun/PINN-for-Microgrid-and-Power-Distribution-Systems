import pandas as pd
import deepxde as dde
import numpy as np
import re
import datetime
import os
import time
import matplotlib.pyplot as plt


class PID():
    def __init__(self, path=None, date=None, index=None, **kwargs):

        self.st = 0.
        self.et = 1.
        self.num_domain = 0
        self.num_boundary = 0
        self.num_test = None

        '''
        输入的维度：时间t + input transform
        '''
        self.net = dde.nn.FNN([1] + [100] * 3 + [5], "swish", "Glorot uniform")

        self.method = "adam"
        self.lr = 0.0001
        self.period = 200
        self.iterations = 20000

        '''
        基本路径设置
        '''
        self.path = path
        self.date = date
        self.index = index
        '''
        保存数据，同时防止数据冲突
        '''
        if date is None:
            now = datetime.datetime.now()
            self.date = now.strftime("%m%d")
        if index is None:
            self.fnamevar_path = "PI_variables_ModelSave_" + self.date + ".dat"
        else:
            self.fnamevar_path = "PI_variables_ModelSave_" + self.date + "_" + str(index) + ".dat"

        self.set_variable()
        self.__dict__.update(kwargs)
        self.iterations = int(self.iterations)
        self.check_dat_file()

        '''
        读取数据
        '''
        input_data = pd.read_csv(path)
        step_time = input_data.Time[1] - input_data.Time[0]
        input_data.Time -= step_time * int(input_data.shape[0] * self.st)
        input_data = input_data[int(input_data.shape[0] * self.st): int(input_data.shape[0] * self.et)]

        self.input_data = input_data

        self.rename_pid_csv()

        PQkp = dde.Variable(self.init_PQkp)
        PQki = dde.Variable(self.init_PQki)
        Ikp = dde.Variable(self.init_Ikp)
        Iki = dde.Variable(self.init_Iki)

        self.variable_list = [PQkp, PQki, Ikp, Iki]

        '''
        这里修改微分方程及其表达形式
        对于一个PI环节而言标准表达式为(Input) * Kp + (Input) * Ki * 1/s = (Output)
        i.e., Kp * s * Input + Ki * Input = s * Output
        '''

        def PI_ode(x, y):

            DeltaQ, Iqref, Iq, DeltaIq, DeltaUq = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4], y[:, 4:5]

            dI_dt1 = dde.grad.jacobian(y, x, i=0)
            dO_dt1 = dde.grad.jacobian(y, x, i=1)

            dI_dt2 = dde.grad.jacobian(y, x, i=3)
            dO_dt2 = dde.grad.jacobian(y, x, i=4)

            '''
                    这里修改微分方程及其表达形式
            '''
            return [
                -(PQkp * dI_dt1 + DeltaQ * PQki) - dO_dt1,
                (Ikp * dI_dt2 + DeltaIq * Iki) - dO_dt2,
                Iqref - Iq - DeltaIq
            ]

        self.pi_ode = PI_ode
        self.real_value_list = [self.real_PQkp, self.real_PQki, self.real_Ikp, self.real_Iki]

    '''
    确认dat文件不冲突
    '''

    def check_dat_file(self):
        if os.path.exists(self.fnamevar_path):
            new_index = input(
                "The dat file already exists, please enter a new index to modify, or press Enter to continue")
            if new_index != '':
                self.index = new_index
                self.fnamevar_path = "PI_variables_ModelSave_" + self.date + "_" + str(new_index) + ".dat"

    def rename_pid_csv(self):
        dict_col_D = {}

        dict_col_Q = {
            'Time': 'Time',
            'Qerr8': 'DeltaQ',
            'Isqrefongrid8': 'Iqref',
            'IsqA8': 'Iq',
            'Isqerr8': 'DeltaIq',
            'uqA8': 'DeltaUq'
        }
        self.input_data.rename(columns=lambda x: dict_col_Q[x.split('|')[-1]], inplace=True)
        self.input_data = self.input_data[list(dict_col_Q.values())]
        self.input_data.set_index('Time', inplace=True)

    def set_variable(self):
        self.init_PQkp = 1.
        self.init_PQki = 1.
        self.init_Ikp = 1.
        self.init_Iki = 1.

        self.real_PQkp = 0.025
        self.real_PQki = 5.
        self.real_Ikp = 0.025
        self.real_Iki = 0.5

    def check_init_data(self):
        data = self.input_data
        time_step = data.index[1] - data.index[0]

        def data_plot(I, O, kp, ki):
            I_t = (data[I].to_numpy()[2:] - data[I].to_numpy()[:-2]) / (2 * time_step)
            O_t = (data[O].to_numpy()[2:] - data[O].to_numpy()[:-2]) / (2 * time_step)
            print(pd.DataFrame([I_t * kp + data[I].to_numpy()[1:-1] * ki, O_t]).T.plot())

        data_plot(data.DeltaQ, data.Iqref, self.real_PQkp, self.real_PQki)
        data_plot(data.DeltaIq, data.DeltaUq, self.real_PQkp, self.real_PQki)

    def build(self, transform=None, **kwargs):
        if transform == ('input' or 'all'):
            self.net = dde.nn.FNN([6] + [100] * 3 + [5], "tanh", "Glorot uniform")
        self.__dict__.update(kwargs)
        input_data = self.input_data
        x = input_data.index.to_numpy()
        geom = dde.geometry.TimeDomain(0, x[-1])
        y = input_data.to_numpy()
        observe_t = x.reshape(-1, 1)
        n = y.shape[0]
        y_true = y

        def boundary(_, on_initial):
            return on_initial

        '''
        这里先目前假设的只能观测到工程实际值, 如果运行失败在考虑增加其他学术理论观测值，比如Iqref， DeltaIq的初始值 / 全部值
        2024-01-15 五个观测成功
        2024-02-02 四个观测成功
        2024-03-05 明确三个观测点无法成功，只能重新构建网络，单一网络无法完全推导所有参数
        '''
        observe_y0 = dde.icbc.PointSetBC(observe_t, y[:, 0:1], component=0)
        observe_y1 = dde.icbc.PointSetBC(observe_t, y[:, 1:2], component=1)
        observe_y2 = dde.icbc.PointSetBC(observe_t, y[:, 2:3], component=2)
        observe_y3 = dde.icbc.PointSetBC(observe_t, y[:, 3:4], component=3)
        observe_y4 = dde.icbc.PointSetBC(observe_t, y[:, 4:5], component=4)
        ic0 = dde.icbc.IC(geom, lambda X: y[0, 0], boundary, component=0)
        ic1 = dde.icbc.IC(geom, lambda X: y[0, 1], boundary, component=1)
        ic2 = dde.icbc.IC(geom, lambda X: y[0, 2], boundary, component=2)
        ic3 = dde.icbc.IC(geom, lambda X: y[0, 3], boundary, component=3)
        ic4 = dde.icbc.IC(geom, lambda X: y[0, 4], boundary, component=4)

        '''
            这里修改观测值，选择给神经网络喂哪部分的数据
        '''
        data = dde.data.PDE(
            geom,
            self.pi_ode,
            [ic0, observe_y0, ic1, ic2, observe_y2, ic3, ic4, observe_y4],
            anchors=observe_t,
        )

        if transform is not None:
            from deepxde.backend import torch
            tran_in = False
            tran_out = False
            if transform == 'input':
                tran_in = True
            if transform == 'output':
                tran_out = True
            if transform == 'all':
                tran_in = True
                tran_out = True

            '''
            这里是输入转换: Input transform
            '''

            def feature_transform(t):
                t = 0.01 * t
                return torch.concat(
                    (
                        t,
                        torch.sin(t),
                        torch.sin(2 * t),
                        torch.sin(3 * t),
                        torch.sin(4 * t),
                        torch.sin(5 * t),
                    ),
                    axis=1,
                )

            data_t = observe_t
            data_y = y

            '''
            这里是输出转换: Output transform
            '''

            def output_transform(t, y):
                idx = n - 1
                k = (data_y[idx] - data_y[0]) / (data_t[idx] - data_t[0])
                b = (data_t[idx] * data_y[0] - data_t[0] * data_y[idx]) / (
                        data_t[idx] - data_t[0]
                )
                linear = torch.as_tensor(k) * t + torch.as_tensor(b)
                factor = torch.tanh(t) * torch.tanh(idx - t)
                y2 = linear + factor * torch.Tensor([1, 1, 1, 1, 1]) * y
                y2[0, 1] = y_true[0, 1]
                y2[0, 3] = y_true[0, 3]
                y2[idx, 1] = y_true[idx, 1]
                y2[0, 3] = y_true[0, 3]

                return y2

            if tran_in:
                self.net.apply_feature_transform(feature_transform)
            if tran_out:
                self.net.apply_output_transform(output_transform)

        model = dde.Model(data, self.net)
        model.compile(self.method, lr=self.lr, external_trainable_variables=self.variable_list)

        self.model = model
        self.x = x
        self.y = y
        self.observe_t = observe_t

    def train(self):
        fnamevar = self.fnamevar_path
        variable = dde.callbacks.VariableValue(self.variable_list, period=self.period, filename=fnamevar)
        if self.index is None:
            check_point_path = 'save_model/model_' + self.date + '.ckpt'
        else:
            check_point_path = 'save_model/model_' + self.date + '_' + str(self.index) + '.ckpt'
        checker = dde.callbacks.ModelCheckpoint(check_point_path, save_better_only=True, period=50000)
        self.model.train(iterations=self.iterations, callbacks=[variable, checker])

    def load_model(self, model_path, load_fnamevar_path=None):
        if load_fnamevar_path is None:
            try:
                load_fnamevar_path = self.fnamevar_path
                lines = open(load_fnamevar_path, "r").readlines()
                Chat = np.array(
                    [
                        np.fromstring(
                            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
                            sep=",",
                        )
                        for line in lines
                    ]
                )
                self.init_PQkp = Chat[-1, 0]
                self.init_PQki = Chat[-1, 1]
                self.init_Ikp = Chat[-1, 2]
                self.init_Iki = Chat[-1, 3]

                print('Load variable values from ' + load_fnamevar_path)

            except:
                pass
        else:
            lines = open(load_fnamevar_path, "r").readlines()
            Chat = np.array(
                [
                    np.fromstring(
                        min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
                        sep=",",
                    )
                    for line in lines
                ]
            )
            self.init_PQkp = Chat[-1, 0]
            self.init_PQki = Chat[-1, 1]
            self.init_Ikp = Chat[-1, 2]
            self.init_Iki = Chat[-1, 3]

        PQkp = dde.Variable(self.init_PQkp)
        PQki = dde.Variable(self.init_PQki)
        Ikp = dde.Variable(self.init_Ikp)
        Iki = dde.Variable(self.init_Iki)
        self.variable_list = [PQkp, PQki, Ikp, Iki]
        self.model.compile(self.method, lr=self.lr, external_trainable_variables=self.variable_list)
        self.model.restore(model_path, verbose=1)


if __name__ == "__main__":
    input_data_path = 'data/pi/0122FltQControl.csv'

    microgird = PID(path=input_data_path, index=None,
                    st=0.,
                    et=1.)

    # # microgird.build()
    microgird.build(transform='all',
                    net=dde.nn.FNN([6] + [100] * 3 + [5], "swish", "Glorot uniform"),
                    lr=1e-3,
                    iterations=5000000)
    # microgird.load_model('save_model/model_testNone.ckpt-200000.pt',
    #                      load_fnamevar_path = 'PI_variables_ModelSave_0301.dat')
    # microgird.train()
    print(microgird.input_data.shape)
    print(microgird.y)
    print(microgird.input_data)
    # y_hat = microgird.model.predict(microgird.observe_t)
    # plt.plot(y_hat, microgird.observe_t)
