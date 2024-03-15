import pandas as pd
import deepxde as dde
import numpy as np
import re
import datetime
import os
import time
import matplotlib.pyplot as plt
# from deepxde.backend import torch
# dde.backend.set_default_backend('pytorch')

class SYNC():
    def __init__(self, path=None, date=None, index=None, **kwargs):

        self.st = 0.
        self.et = 1.
        self.num_domain = 0
        self.num_boundary = 0
        self.num_test = None

        '''
        输入的维度：时间t + input transform
        '''
        self.net = dde.nn.FNN([1] + [100] * 3 + [3], "swish", "Glorot uniform")


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
            self.fnamevar_path = "SYNC_variables_" + self.date + ".dat"
        else:
            self.fnamevar_path = "SYNC_variables_" + self.date + "_" + str(index) + ".dat"

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
        input_data = input_data[int(input_data.shape[0]*self.st): int(input_data.shape[0]*self.et)]

        self.input_data = input_data

        self.rename_sync_csv()

        H = dde.Variable(self.init_H)
        D = dde.Variable(self.init_D)

        self.variable_list = [H, D]

        '''
        这里修改微分方程及其表达形式
        对于一个PI环节而言标准表达式为(Input) * Kp + (Input) * Ki * 1/s = (Output)
        i.e., Kp * s * Input + Ki * Input = s * Output
        '''

        def SYNC_ode(x, y):

            Tm, w, Te = y[:, 0:1], y[:, 1:2], y[:, 2:3]

            dw_dt = dde.grad.jacobian(y, x, i=1)


            '''
                    这里修改微分方程及其表达形式
            '''
            return [
                Tm-Te - 2*H*dw_dt - D*(w-1)
            ]



        self.pi_ode = SYNC_ode

    '''
    确认dat文件不冲突
    '''
    def check_dat_file(self):
        if os.path.exists(self.fnamevar_path):
            new_index = input(
                "The dat file already exists, please enter a new index to modify, or press Enter to continue")
            if new_index != '':
                self.index = new_index
                self.fnamevar_path = "SYNC_variables_" + self.date + "_" + str(new_index) + ".dat"
    
    def rename_sync_csv(self):
        # dict_col_sync = {
        #     'Time': 'Time',
        #     'Tm': 'Tm',
        #     'WPU': 'w',
        #     'Te_sync': 'Te',
        #     'w_sync': 'W_waiting_for_verification'
        #     }
        # self.input_data.rename(columns=lambda x: dict_col_sync[x.split('|')[-1]], inplace=True)
        # col_name = list(dict_col_sync.values())
        # col_name.remove('W_waiting_for_verification')
        # self.ver_w = self.input_data['W_waiting_for_verification']
        # self.input_data = self.input_data[col_name]
        # self.input_data.set_index('Time', inplace=True)

        # f_info = self.path.split('/')[-1].replace('.csv', '').split('_')
        # self.sync_case = f_info[1]
        # self.real_value_list = re.findall(r'[0-9]+\.?[0-9]*',f_info[2])

        dict_col_sync = {
            'Time': 'Time',
            'Tm': 'Tm',
            'TESTSPDOUT': 'w',
            'TESTTELECT': 'Te',
            }
        self.input_data.rename(columns=lambda x: dict_col_sync[x.split('|')[-1]], inplace=True)
        col_name = list(dict_col_sync.values())
        self.input_data = self.input_data[col_name]
        self.input_data.set_index('Time', inplace=True)

        self.real_value_list = [1.5, 0.25]
    
    def set_variable(self):
        self.init_H = 1.
        self.init_D = 1.
    

    def build(self, transform=None, **kwargs):
        if transform == ('input' or 'all'):
            self.net = dde.nn.FNN([6] + [100] * 3 + [3], "tanh", "Glorot uniform")
        self.__dict__.update(kwargs)
        input_data = self.input_data
        x = input_data.index.to_numpy()
        geom = dde.geometry.TimeDomain(0, x[-1])
        y = input_data.to_numpy()
        observe_t = x.reshape(-1, 1)
        n = y.shape[0]
        y_real = y
        def boundary(_, on_initial):
            return on_initial
        '''
        
        '''
        observe_y0 = dde.icbc.PointSetBC(observe_t, y[:, 0:1], component=0)
        observe_y1 = dde.icbc.PointSetBC(observe_t, y[:, 1:2], component=1)
        observe_y2 = dde.icbc.PointSetBC(observe_t, y[:, 2:3], component=2)

        ic0 = dde.icbc.IC(geom, lambda X: y[0, 0], boundary, component=0)
        ic1 = dde.icbc.IC(geom, lambda X: y[0, 1], boundary, component=1)
        ic2 = dde.icbc.IC(geom, lambda X: y[0, 2], boundary, component=2)


        '''
            这里修改观测值，选择给神经网络喂哪部分的数据
        '''
        
        data = dde.data.PDE(
            geom,
            self.pi_ode,
            [observe_y0, observe_y1, observe_y2, ic0, ic1, ic2],
            # [],
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
                y2 = linear + factor * torch.Tensor([1, 1, 1]) * y
                # y2[:4, :] = torch.Tensor(y_real[:4, :])
                # y2[-5:, :] = torch.Tensor(y_real[-5:, :])
                return y2

            if tran_in:
                self.net.apply_feature_transform(feature_transform)
            if tran_out:
                self.net.apply_output_transform(output_transform)

        model = dde.Model(data, self.net)
        model.compile(self.method, lr=self.lr,
                    #    loss_weights=[1,10,1,1,1], 
                       external_trainable_variables=self.variable_list)

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
        checker= dde.callbacks.ModelCheckpoint(check_point_path, save_better_only = True, period = 50000)
        self.model.train(iterations=self.iterations, callbacks=[variable, checker])

    def load_model(self, model_path, load_fnamevar_path = None):
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
                self.H = Chat[-1, 0]
                self.D = Chat[-1, 1]

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
            self.init_H = Chat[-1, 0]
            self.init_D = Chat[-1, 1]


        H = dde.Variable(self.init_H)
        D = dde.Variable(self.init_D)
        self.variable_list = [H, D]
        self.model.compile(self.method, lr=self.lr, external_trainable_variables=self.variable_list)
        self.model.restore(model_path, verbose=1)


if __name__ == "__main__":

    input_data_path = '../data/synchronous-machine/927testforTe2.csv'
    microgird = SYNC(path=input_data_path,
                    st=0.2,
                    et=0.5,
                    index = 'TransIn')

    # # microgird.build()
    microgird.build(transform='input',
                    net=dde.nn.FNN([6] + [100] * 3 + [3], "swish", "Glorot uniform"),
                    lr=1e-3,
                    iterations=3000000)
    # microgird.load_model('save_model/model_testNone.ckpt-200000.pt',
    #                      load_fnamevar_path = 'PI_variables_ModelSave_0301.dat')
    microgird.train()
    # print(microgird.input_data.shape)
    # y_hat = microgird.model.predict(microgird.observe_t)
    # plt.plot(y_hat, microgird.observe_t)
