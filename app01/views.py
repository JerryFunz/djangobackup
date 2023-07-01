from django.shortcuts import render, HttpResponse
from django.core.serializers import serialize
import json
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, SimpleRNN
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import tushare as ts
import datetime
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import time
# from tencentcloud.mongodb.v20190725.models import DescribeDBInstancesRequest
# from tencentcloud.mongodb.v20190725.mongodb_client import MongodbClient
from keras.layers import LSTM, SimpleRNN, Dense
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import rnn
from prophet import Prophet
import multiprocessing
import baostock as bs
from scipy.stats import spearmanr
from django.http import JsonResponse
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
# cd django && pipenv shell
# cd mysite && python manage.py runserver


# 域名地址 https://django-a7ue-58259-9-1318415499.sh.run.tcloudbase.com

# wx.cloud.callContainer({
#   "config": {
#     "env": "interlink-5g8yxa2rcdd15093"
#   },
#   "path": "/api/count",
#   "header": {
#     "X-WX-SERVICE": "django-a7ue"
#   },
#   "method": "POST",
#   "data": {
#     "action": "inc"
#   }
# })

# Create your views here.
# def index(request):
#     return HttpResponse("Hello World!")

# def user_list(request):
#     return HttpResponse("user_info")



def calc_accuracy(request):
    # startTime=time.time()


    print(request.GET)
    data = request.GET.get('data')
    data_dict = json.loads(data) # 将JSON字符串解析为Python对象

    close_pred = data_dict.get('close_pred')
    tradeDates= data_dict.get('tradeDates')
    uploadTime_day= data_dict.get('uploadTime_day')
    stockID= data_dict.get('stockID')
    print('------------------------------')
    print(stockID)
    startDate= data_dict.get('startDate')

    # print('readed ID'+ stockID)
    # print('readed startDate'+ startDate)
    # print('readed close_pred'+ close_pred)
    # print('readed tradeDates'+ tradeDates)
    # print('readed uploadTime_day'+ uploadTime_day)


    queue = multiprocessing.Queue()  # 创建队列用于接收子进程返回结果
    event = multiprocessing.Event()  # 创建事件用于同步

    kwargs = {'stockID': stockID,'startDate':startDate, 'queue': queue, 'event': event}
    p = multiprocessing.Process(target=get_baostock, kwargs=kwargs)
    p.start()
    event.wait(timeout=5)

    if event.is_set():
        # 方法已经返回，获取返回结果
        tushare = queue.get()
    else:
        # 方法还没有返回，在5秒内执行其他操作
        print('baostock timeout, use tushare')
        p.terminate()  # 终止子进程
        tushare = get_tushare(stockID=stockID, startDate=startDate)

    # print(len(tushare))
    # print(tushare)
    # print("____________________________________")
    if len(tushare) == 0:
        return HttpResponse('No Update Data')
    else:
    
        close_true=tushare['close'].tolist()
        print(close_true)
        print(close_pred)
        if len(close_pred)>len(close_true):
            close_pred=close_pred[0:len(close_true)]
        else:
            close_true=close_true[0:len(close_pred)]
        # close_pred=close_pred[0:len(close_true)]

        rho, p_value = spearmanr(close_true, close_pred)
        rho=(rho+1.001)/2
        lastmatch=1-abs(((close_true[-1]-close_pred[-1])/close_true[-1])**0.5)
        accuracy= (0.6*rho+0.4*lastmatch)*100
        accuracy= '{:.1f}'.format(accuracy)+"%"
        print(accuracy)


    response_data = {
        'accuracy': accuracy,
        'close_true': close_true
    }

    # 构造JsonResponse对象
    response = JsonResponse(response_data)

    return response





def pred(request):
    startTime=time.time()

    print(request.GET)
    activeModel = request.GET.get('activeModel')
    parameter = request.GET.get('parameter')
    selectedDay = request.GET.get('selectedDay')
    stockID = request.GET.get('stockID')
    print(activeModel, parameter, selectedDay, stockID)
    # print("activeModel: " + activeModel)


    queue = multiprocessing.Queue()  # 创建队列用于接收子进程返回结果
    event = multiprocessing.Event()  # 创建事件用于同步

    kwargs = {'stockID': stockID, 'activeModel': activeModel, 'queue': queue, 'event': event}
    p = multiprocessing.Process(target=get_baostock, kwargs=kwargs)
    p.start()
    event.wait(timeout=8)

    if event.is_set():
        # 方法已经返回，获取返回结果
        tushare = queue.get()
    else:
        # 方法还没有返回，在5秒内执行其他操作
        print('baostock timeout, use other logic')
        tushare = get_tushare(stockID=stockID, activeModel=activeModel)


        # 在这里可以使用变量 tushare 进行进一步的操作


    total_data = model_prediction(stockID, tushare, activeModel, parameter, selectedDay)

    json_data = total_data.to_json(orient='records')

    endTime = time.time() 
    elapsed_time = endTime - startTime
    time_to_wait=4

    if activeModel=='RNN':
        time_to_wait=8

    if elapsed_time<time_to_wait:
        waitTime = time_to_wait - int(elapsed_time)
        time.sleep(waitTime)

    # 返回JSON数据给微信小程序
    return HttpResponse(json_data, content_type='application/json')



def get_tushare(stockID,activeModel=None,startDate=None):

    today = datetime.date.today()   
    
    if startDate==None:
        year=1
        if activeModel=='123' or activeModel=='TCN':
            year=3
        
        startDate = today - datetime.timedelta(days = year * 365)
        startDate = startDate.strftime("%Y%m%d")
    else:
        startDate=startDate

    today = today.strftime("%Y%m%d")


    pro = ts.pro_api('480df9c4a5acaf2c46789af97ef7b4fde3d489820dbf9e9b0f17f17f')

    # 拉取数据
    tushare = pro.daily(**{
        "ts_code": stockID,
        "trade_date": "",
        "start_date": startDate,
        "end_date": today,
        "offset": "",
        "limit": ""
    }, fields=[
        "ts_code",
        "trade_date",
        "close"
    ])


    if len(tushare)!=0:
        tushare = tushare.sort_values('trade_date')
    # delete
    else:
        tushare = pd.DataFrame()
    

    return tushare



def get_baostock(stockID,queue, event, activeModel=None,startDate=None):

    today = datetime.date.today()   
    
    if startDate==None:
        year=1
        if activeModel=='123' or activeModel=='TCN':
            year=3
        
        startDate = today - datetime.timedelta(days = year * 365)
        startDate = startDate.strftime("%Y-%m-%d")
    else:
        startDate = datetime.datetime.strptime(startDate, "%Y%m%d")  # 解析为 datetime 对象
        startDate=startDate.strftime("%Y-%m-%d")

    today = today.strftime("%Y-%m-%d")

    lg = bs.login()
    print('stockID'+stockID)
    rs = bs.query_history_k_data_plus(stockID,
    "code,date,close",
    start_date=startDate, end_date=today,
    frequency="d", adjustflag="3")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
        
    print(data_list)

    if len(data_list)!=0:
        tushare = pd.DataFrame(data_list, columns=rs.fields)
        tushare.rename(columns={'date':'trade_date'}, inplace = True)
        tushare['trade_date'] = pd.to_datetime(tushare['trade_date'])
        tushare['trade_date']= tushare['trade_date'].dt.strftime('%Y%m%d')
        tushare['close'] = tushare['close'].astype(float)
    else:
        tushare=pd.DataFrame()


    bs.logout()
    queue.put(tushare)
    event.set()


def get_future_dates(selectedDay):
    calendar = mcal.get_calendar('XSHG')

# 获取当前日期
    current_date = pd.Timestamp.today()

    # 用于存储未来非节假日的日期
    future_dates = []

    # 计数器，用于控制获取日期数量
    count = 0

    # 获取未来30个非节假日的日期
    while count < selectedDay:
        current_date += pd.Timedelta(days=1)
        
        # 检查日期是否为交易日
        if calendar.valid_days(start_date=current_date, end_date=current_date).size > 0:
            future_dates.append(current_date)
            count += 1

    # 将日期转换为DatetimeIndex对象
    future_dates = pd.DatetimeIndex(future_dates)

    future_dates_formatted = future_dates.strftime('%Y%m%d')

    return future_dates_formatted

def model_prediction(stockID,tushare, activeModel, parameter, selectedDay):
    
    parameter=eval(parameter)
    selectedDay=int(selectedDay)

    if activeModel=='MA':
        tw_sliderValue = parameter['tw_sliderValue']
        weight_sliderValue=parameter['weight_sliderValue']
        

        weights = []  
        if weight_sliderValue==0:
            weights = [1] * tw_sliderValue
        else:
            for i in range(tw_sliderValue):
                i=float(i)
                # print((1+weight_sliderValue/200)**i)
                weights.append((1+weight_sliderValue/150)**i)

        predictions = []  # 存储预测值的列表
        train = tushare['close']

        for _ in range(selectedDay):
            # 计算当前的加权移动平均（WMA）值
            current_wma = train.rolling(window=tw_sliderValue).apply(lambda x: np.dot(x, weights)/sum(weights), raw=True).iloc[-1]
            
            # 将当前的预测值添加到预测列表中
            predictions.append(current_wma)
        
        # 将当前的预测值添加到当前数据的末尾
            train = pd.concat([train, pd.Series([current_wma], index=[train.index[-1] + 1])])


    elif activeModel=='AR':
        print('AR')
        order_sliderValue=parameter['order_sliderValue']
        trendValue=parameter['trendValue'].split('-')[0]
        periodValue=parameter['periodDict'][parameter['periodValue']]
        periodValue=int(periodValue)

        train = tushare['close'].values

        model = AutoReg(train, lags=order_sliderValue, trend=trendValue,   period=periodValue)
        ar_result = model.fit()
        
        forecast = ar_result.predict(start=len(train), end=len(train)+selectedDay-1) 

        # Output forecast results
        predictions = forecast.tolist()


    elif activeModel=='ARIMA':
        p = parameter['ARIMARegValue']
        d = parameter['ARIMADiffValue']
        q = parameter['ARIMAOrderValue']

        train = tushare['close'].values

        model = ARIMA(train, order=(p, d, q))
        arima_result = model.fit()
        predictions = arima_result.forecast(steps=selectedDay)


    elif activeModel=='RNN':
        RNNhiden1_Value=parameter['RNNhiden1_Value']
        RNNhiden2_Value=parameter['RNNhiden2_Value']
        RNNhiden3_Value=parameter['RNNhiden3_Value']
        RNNhiden4_Value=parameter['RNNhiden4_Value']
        RNNhiden5_Value=parameter['RNNhiden5_Value']
        epochs = 200
        lr = 0.001

        train = tushare['close'].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train.reshape(-1, 1))

        X_train = []
        y_train = []
        for i in range(30, len(train_data)):
            X_train.append(train_data[i-30:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # 调整输入数据的形状
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # 转换为PyTorch的Tensor
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()

        # 构建数据集和数据加载器
        class CustomDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        dataset = CustomDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)




        # 初始化模型和优化器
        model = RNN(1, RNNhiden1_Value, RNNhiden2_Value, RNNhiden3_Value, RNNhiden4_Value, RNNhiden5_Value)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 训练模型
        for epoch in range(epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                loss.backward()
                optimizer.step()

        # 预测
        last_30_days = train_data[-30:].reshape(1, -1, 1)
        predictions = []
        for _ in range(selectedDay):
            prediction = model(torch.from_numpy(last_30_days).float())
            predictions.append(prediction.item())
            last_30_days = np.roll(last_30_days, -1)
            last_30_days[0, -1, 0] = prediction.item()

        # 反向缩放预测结果
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # 输出预测结果
        predictions = predictions.flatten().tolist()


    elif activeModel=='CNN':   
        CNN1_Value=parameter['CNN1_Value']
        CNN2_Value=parameter['CNN2_Value']
        CNN3_Value=parameter['CNN3_Value']
        CNN4_Value=parameter['CNN4_Value']
        CNN5_Value=parameter['CNN5_Value']
        CNNsequence=parameter['CNNsequence']

        train = tushare['close'].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data = scaler.fit_transform(train.reshape(-1, 1))

        # 划分训练集和验证集
        X_train = []
        y_train = []

        for i in range(CNNsequence, len(train_data)):
            X_train.append(train_data[i - CNNsequence:i, 0])
            y_train.append(train_data[i, 0])

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(dataset, batch_size=32)

        model = CNNModel(CNN1_Value, CNN2_Value, CNN3_Value, CNN4_Value, CNN5_Value, CNNsequence)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        epochs = 400
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()



        last_sequence = torch.tensor(train_data[-CNNsequence:], dtype=torch.float32)
        predictions = []

        for _ in range(selectedDay):
            input_tensor = last_sequence.unsqueeze(0)  # 添加额外的维度作为batch维度
            input_tensor = input_tensor.permute(0, 2, 1)
            input_tensor = input_tensor.view(1, -1)
            prediction = model(input_tensor)
            prediction = prediction.item()
            predictions.append(prediction)
            last_sequence = torch.roll(last_sequence, -1)
            last_sequence[-1] = prediction


        # 将预测结果反向缩放
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # 输出预测结果
        predictions = predictions.flatten().tolist()



    elif activeModel=='LSTM':
        # LSTM参数
        LSTMhiden1_Value=parameter['LSTMhiden1_Value']
        LSTMhiden2_Value=parameter['LSTMhiden2_Value']
        LSTMhiden3_Value=parameter['LSTMhiden3_Value']
        LSTMhiden4_Value=parameter['LSTMhiden4_Value']
        LSTMhseq_Value=int(parameter['LSTMhseq_Value'])
        # epochs=int(1000/LSTMhseq_Value)
        seq_length = LSTMhseq_Value
        input_size = seq_length 
        hidden_size = 16
        num_layers = 1
        output_size = 1  # 输出特征的维度
        max_norm = 1.0

        close_prices = tushare['close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)
        x = torch.tensor(scaled_data[:-1]).unsqueeze(0).float()
        y_true = torch.tensor(scaled_data[1:]).unsqueeze(0).float()
        x = torch.transpose(x, 1, 2)

        # 创建模型和优化器
        model = LSTMModel(input_size, hidden_size, num_layers)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # 训练过程
        model.train()
        num_epochs = len(close_prices) - seq_length - 1

        for i in range(10):
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                current_x = x[:, :, epoch: epoch + seq_length]
                current_y_true = y_true[:, epoch + seq_length]

                # 梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                y_pred = model(current_x)
                loss = criterion(y_pred[:, -1], current_y_true)

                loss.backward()
                optimizer.step()



                if (epoch + 1) % 100 == 0:
                    print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
                    print('current_x' + str(current_x))
                    print('y_pred' + str(y_pred[:, -1]))
                    print('y_true' + str(current_y_true))


        # 预测结果
        future_predictions = []
        for _ in range(selectedDay):
            hist_x = x[:, :, -seq_length:] # 修改维度顺序

            prediction = model(hist_x)
            print(prediction)
            prediction = prediction.unsqueeze(2)

            x = torch.cat([x, prediction], dim=2)  # 在时间维度上进行拼接

            future_predictions.append(prediction.item())

        # 反归一化，得到预测结果
        y_pred = scaler.inverse_transform(torch.tensor(future_predictions).reshape(-1, 1))

        predictions = [item for sublist in y_pred.tolist() for item in sublist]


    # elif activeModel=='TRANSFM':
    #     num_steps = selectedDay


    #     hidden_dim = parameter['TRANSFMhiden_Value']
    #     num_layers = parameter['TRANSFMdecode_Value']
    #     dropout_rate = parameter['TRANSFMreset_Value']/100
    #     # sequence_length = parameter['TRANSFMsequence_Value']
    #     batch_size = 32

        
    #     input_dim = 1  # 输入特征维度
    #     output_dim = 1  # 输出特征维度
    #     num_heads = 1  # 多头注意力的头数
    #     learning_rate = 0.005
    #     num_epochs = 1500
        




    #     # tushare['trade_date'] = pd.to_datetime(tushare['trade_date'], format='%Y%m%d')
    #     # tushare = tushare.sort_values('trade_date')
    #     # tushare = tushare.set_index('trade_date')

    #     # 提取特征和目标数据
    #     input_dim = 1  # 输入特征维度
    #     output_dim = 1  # 输出特征维度

    #     # 特征数据X：使用前n-1个时间步的close作为特征
    #     X = tushare['close'].values[:-1].reshape(-1, 1)

    #     # 目标数据Y：使用第n个时间步的close作为目标
    #     Y = tushare['close'].values[1:].reshape(-1, 1)

    #     # 将数据转换为PyTorch张量
    #     X = torch.tensor(X, dtype=torch.float32)
    #     Y = torch.tensor(Y, dtype=torch.float32)

    #     # 将数据重新整理为合适的形状 (num_samples, seq_length, input_dim)
    #     seq_length = 1  # 序列长度
    #     num_samples = len(X)
    #     X = X.view(num_samples, seq_length, input_dim)
    #     Y = Y.view(num_samples, output_dim)

    #     # 定义Transformer模型
    #     class Transformer(nn.Module):
    #         def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout_rate):
    #             super(Transformer, self).__init__()

    #             assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
    #             self.encoder = nn.TransformerEncoder(
    #                 nn.TransformerEncoderLayer(input_dim, num_heads, hidden_dim),
    #                 num_layers
    #             )
    #             self.decoder = nn.Linear(input_dim, output_dim)

    #         def forward(self, x):
    #             x = self.encoder(x)
    #             x = self.decoder(x[:, -1, :])  # 只使用最后一个时间步的输出
    #             return x
            
        
        
    #     # 创建模型实例
    #     model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout_rate)
    #     # 定义损失函数和优化器
    #     criterion = nn.MSELoss()
    #     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #     # 划分训练集和测试集
    #     train_size = int(0.8 * num_samples)
    #     X_train, X_test = X[:train_size], X[train_size:]
    #     Y_train, Y_test = Y[:train_size], Y[train_size:]

    #     # 训练模型
    #     for epoch in range(num_epochs):
    #         optimizer.zero_grad()

    #         outputs = model(X_train)
    #         loss = criterion(outputs, Y_train)

    #         loss.backward()
    #         optimizer.step()

    #         if (epoch+1) % 10 == 0:
    #             print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    #     # 使用训练好的模型进行预测
    #     predictions = model(X_test)

    #     # 获取预测结果
    #     predicted_values = predictions.detach().numpy()

    #     # 将预测值作为新的输入来预测下一个时间步的数据
    #     # 假设你的第一个预测值为predicted_values[0]
    #     next_input = torch.tensor(X[-1], dtype=torch.float32).view(1, seq_length, input_dim)
    #     next_predicted_values = [next_input]



    #     # 循环预测下一个时间步的数据
    #     predictions = []
    #     for step in range(num_steps):
    #         next_prediction = model(next_input)
    #         next_predicted_values.append(next_prediction)
            
    #         # 从原始数据中获取下一个时间步的输入
    #         next_input = torch.tensor(Y[-num_steps + step], dtype=torch.float32).view(1, seq_length, input_dim)
            
    #         # 打印每次用于预测的序列
    #         sequence = next_input.view(-1).tolist()
    #         print(f"Sequence for step {step+1}: {sequence}")
    #         predictions.append(sequence[0])


    elif activeModel=='TCN':


        TCNkernel_Value=parameter['TCNkernel_Value']
        TCNdropout_Value=parameter['TCNdropout_Value']/100
        TCNnum_conv_Value=parameter['TCNnum_conv_Value']
        TCNseq_Value=parameter['TCNseq_Value']
        TCNchannels1_Value=parameter['TCNchannels1_Value']
        TCNchannels2_Value=parameter['TCNchannels2_Value']
        TCNchannels3_Value=parameter['TCNchannels3_Value']

        num_channels = [TCNchannels1_Value, TCNchannels2_Value, TCNchannels3_Value]  # 每个TCN层的通道数量
        kernel_size = TCNkernel_Value  # 卷积核大小
        dropout = TCNdropout_Value # Dropout概率
        num_conv_layers_per_tcn =  TCNnum_conv_Value # 每个TCN层中的额外卷积层数量
        num_predictions = selectedDay
        seq_length = TCNseq_Value
        input_size = 1  # 输入维度
        output_size = 1  # 输出维度


        close_prices = tushare['close'].values.reshape(-1, 1)


        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)

        # 将数据转换为PyTorch张量
        x = torch.tensor(scaled_data[:]).unsqueeze(0).float()
        y_true = torch.tensor(scaled_data[:]).unsqueeze(0).float()  # 目标数据形状：(1, seq_len, 1)

        x = torch.transpose(x, 1, 2) 



        model = TCNModel(input_size, output_size, num_channels, kernel_size, dropout, num_conv_layers_per_tcn)
        # model.reset_weights()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 模型训练
        num_epochs = len(close_prices) - seq_length -1 
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            current_x = x[:, :, epoch : epoch + seq_length]
            current_y_true = y_true[:,  epoch + seq_length+1]

            y_pred = model(current_x)
            loss = criterion(y_pred[:, -1], current_y_true)


            loss.backward()
            optimizer.step()
            

            if (epoch + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}, Loss: {loss.item()}')
                print('current_x' + str(current_x))
                print('y_pred' + str(y_pred[:, -1]))
                print('y_true' + str(current_y_true))
        
        future_predictions = []

        for _ in range(num_predictions):
            prediction = model(x[:, :, -seq_length:])
            
            prediction = prediction.unsqueeze(2)

            # print('prediction'+str(prediction))  # 将prediction的维度扩展到与last_sequence相同
            # print('last_sequence'+str(x[:, :, -seq_length:]))  # 将last_sequence的结汇扩展到与last_sequence目同
            
            x = torch.cat([x[:, :, :], prediction], dim=2)


            future_predictions.append(prediction.item())
            #     print(last_sequence[:, :, -7:])  # Print the last 7 values of last_sequence


        # 反归一化，得到预测结果
        y_pred = scaler.inverse_transform(torch.tensor(future_predictions).reshape(-1, 1))


        predictions = [item for sublist in y_pred for item in sublist]


    elif activeModel=='Prophet':
        data =tushare.copy()    
        data= data.rename(columns={'close': 'y'})
        data= data.rename(columns={'trade_date': 'ds'})


        growth = 'linear'
        # n_changepoints = 10
        changepoint_range = 0.8
        yearly_seasonality = True
        weekly_seasonality = False
        daily_seasonality = False
        # seasonality_mode = 'multiplicative'
        seasonality_prior_scale = 10
        
        holidays_prior_scale = 10.0
        changepoint_prior_scale = 0.05

        model = Prophet(
            growth=growth,
            # n_changepoints=n_changepoints,
            changepoint_range=changepoint_range,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,      
            # seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays=None,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale
        )

        # 拟合模型
        model.fit(data)

        # 预测未来时间点
        future = model.make_future_dataframe(periods=selectedDay)
        forecast = model.predict(future)

        # 查看预测结果
        predictions=forecast['yhat'][-selectedDay:].values.tolist()



    future_dates_formatted = get_future_dates(selectedDay)

    predictions = [round(x, 2) for x in predictions]

    pred_df = pd.DataFrame({'ts_code':stockID, 'trade_date': future_dates_formatted, 'close': predictions })

    

    total_data = pd.concat([tushare, pred_df], ignore_index=True)

    total_data['ts_code']= stockID

    total_data = total_data.sort_values('trade_date')

# 取最近的90天数据
    lastndays=selectedDay*4
    total_data = total_data.iloc[-lastndays:]



    return total_data











class CNNModel(nn.Module):
    def __init__(self, CNN1_Value, CNN2_Value, CNN3_Value, CNN4_Value, CNN5_Value, CNNsequence):
        super(CNNModel, self).__init__()
        self.conv1d = nn.Conv1d(1, CNN1_Value, CNN2_Value)
        self.maxpool1d = nn.MaxPool1d(CNN3_Value)
        self.flatten = nn.Flatten()
        self.fc_input_size = CNN1_Value * ((CNNsequence - CNN2_Value + 1) // CNN3_Value)
        self.dense1 = nn.Linear(self.fc_input_size, CNN4_Value)
        self.dense2 = nn.Linear(CNN4_Value, CNN5_Value)
        self.dense3 = nn.Linear(CNN5_Value, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1d(x))
        x = self.maxpool1d(x)
        x = self.flatten(x)
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.dense3(x)
        return x

class TCNModel(nn.Module):

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, num_conv_layers_per_tcn):
        super(TCNModel, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_conv_layers_per_tcn = num_conv_layers_per_tcn

        self.tcn_layers = self._create_tcn_layers(input_size)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def _create_tcn_layers(self, input_size):
        layers = []
        for i in range(len(self.num_channels)):
            in_channels = input_size if i == 0 else self.num_channels[i-1]
            out_channels = self.num_channels[i]
            dilation = 2 ** i
            padding = int((self.kernel_size - 1) * dilation / 2)

            layers.append(nn.Conv1d(in_channels, out_channels, self.kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

            # Add additional convolutional layers within each TCN layer
            for _ in range(self.num_conv_layers_per_tcn - 1):
                layers.append(nn.Conv1d(out_channels, out_channels, self.kernel_size, padding=padding, dilation=dilation))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.tcn_layers(x)
        out = out[:, :, -1]  # 取最后一个时间步的输出
        out = self.fc(out)
        return out
    
        # def reset_weights(self):
    #     def reset_parameters(module):
    #         if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
    #             init.xavier_uniform_(module.weight.data)
    #             if module.bias is not None:
    #                 init.zeros_(module.bias.data)

    #     self.apply(reset_parameters)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = F.relu(out)
        return out

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size1, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, hidden_size3)
        self.fc3 = nn.Linear(hidden_size3, hidden_size4)
        self.fc4 = nn.Linear(hidden_size4, hidden_size5)
        self.fc5 = nn.Linear(hidden_size5, 1)

    def forward(self, x):
        _, h = self.rnn(x)
        x = self.fc1(h)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x