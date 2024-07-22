import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter, butter, filtfilt


def create_dataset(train_df, test_df, columns, x_scaler=None, y_scaler=None):
    dataset = dict()

    if x_scaler is None:
        dataset['train_input'] = torch.tensor(train_df.iloc[:, columns].values).float()
        dataset['test_input'] = torch.tensor(test_df.iloc[:, columns].values).float()
    else:
        dataset['train_input'] = torch.tensor(x_scaler.fit_transform(train_df.iloc[:, columns].values)).float()
        dataset['test_input'] = torch.tensor(x_scaler.transform(test_df.iloc[:, columns].values)).float()

    if y_scaler is None:
        dataset['train_label'] = torch.tensor(train_df.iloc[:, 62].values.reshape(-1, 1)).float()
        dataset['test_label'] = torch.tensor(test_df.iloc[:, 62].values.reshape(-1, 1)).float()
    else:
        dataset['train_label'] = torch.tensor(y_scaler.fit_transform(train_df.iloc[:, 62].values.reshape(-1, 1))).float()
        dataset['test_label'] = torch.tensor(y_scaler.transform(test_df.iloc[:, 62].values.reshape(-1, 1))).float()

    return dataset

def plot_df(df, rows, columns):
    plt.plot(df.iloc[rows, columns], label=df.columns[columns], linewidth=1)
    plt.ylabel(r'$\Delta A_d$')
    plt.legend()
    plt.twinx()
    plt.plot(df.iloc[rows, 62], linewidth=1, color='red')
    plt.ylabel('T')
    # plt.legend()
    # plt.show()


def read_data(file_path, sheet_name=0):
    """
    Read data from a file into a pandas DataFrame.

    Args:
        file_path (str): The path to the file.
        sheet_name (int or str): The name of the sheet in the file.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the file.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")

    return df


def plot_predictions(test, predicted):
    rmse = np.sqrt(np.mean((test - predicted)**2))
    r2 = r2_score(test, predicted)
    # print("RMSE {:.4f} R2 {:.4f}".format(rmse, r2))

    plt.plot(test, label='Real Temperature')
    plt.plot(predicted, label='Predicted Temperature')
    plt.title('Human skin temperature Prediction')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()


def test_model(model, dataset, scaler=None):
    y_pred = model(dataset['test_input']).detach().numpy()
    y = dataset['test_label'].numpy()

    if scaler is not None:
        y = scaler.inverse_transform(y)
        y_pred = scaler.inverse_transform(y_pred)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    r2 = r2_score(y, y_pred)
    plt.scatter(y, y_pred, alpha=0.5)
    plt.title("RMSE {:.4f} R2 {:.4f}".format(rmse, r2))
    # 添加对角线
    plt.plot(y, y, color='red', linestyle='--')

    plt.show()


def test_model2(y_pred, y, scaler=None,color='blue'):

    if scaler is not None:
        y = scaler.inverse_transform(y)
        y_pred = scaler.inverse_transform(y_pred)

    rmse = np.sqrt(np.mean((y - y_pred)**2))
    r2 = r2_score(y, y_pred)
    # plt.subplot(2,1,1)
    plt.scatter(y, y_pred, alpha=0.5,color=color)
    plt.title("RMSE {:.4f} R2 {:.4f}".format(rmse, r2))
    # 添加对角线
    plt.plot(y, y, color='red', linestyle='--')
    # plt.subplot(2,1,2)
    # plot_predictions(y, y_pred)
    plt.show()


def validate_model(model, dataset, scaler=None):
    y_pred = model(dataset['validate_input']).detach().numpy()
    y = dataset['validate_label']

    # fix nan
    # 找到y中非nan的索引
    non_nan_indices = ~np.isnan(y_pred)
    y = y[non_nan_indices].reshape(-1, 1)
    y_pred = y_pred[non_nan_indices].reshape(-1, 1)

    if scaler is not None:
        y = scaler.inverse_transform(y)
        y_pred = scaler.inverse_transform(y_pred)

    plt.scatter(y, y_pred, alpha=0.5, color='green')
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    r2 = r2_score(y, y_pred)

    y_pred = model(dataset['test_input']).detach().numpy()
    y = dataset['test_label'].numpy()

    y = scaler.inverse_transform(y)
    if scaler is not None:
        y_pred = scaler.inverse_transform(y_pred)

    # plt.scatter(y, y_pred, alpha=0.5, color='blue')
    plt.title("RMSE {:.4f} R2 {:.4f}".format(rmse, r2))
    # 添加对角线
    plt.plot(y, y, color='red', linestyle='--')

    plt.show()


def savgol_filt(df, window_size=30, k=3):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df.iloc[0:1000, 2:8], label=df.columns[2:8])
    plt.legend()
    df.iloc[:, 2:62] = savgol_filter(df.iloc[:, 2:62].values,
                                     window_size, k, axis=0)
    plt.subplot(2,1,2)
    plt.plot(df.iloc[0:1000, 2:8], label=df.columns[2:8])
    plt.legend()
    plt.show()
    return df


def moving_average(data, window_size):
    """
    对矩阵的每列进行移动平均处理
    :param data: 输入的矩阵，每列表示一个信号
    :param window_size: 移动平均的窗口大小
    :return: 处理后的矩阵
    """
    # 确保窗口大小是正整数
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer")

    # 计算移动平均
    avg_data = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_size)/window_size, mode='same'), 
        axis=0, 
        arr=data
    )

    return avg_data


def butter_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth filter to each column of the given matrix.

    Parameters:
    data (numpy.ndarray): 2D array where each column will be filtered.
    cutoff (float): Desired cutoff frequency of the filter, Hz.
    fs (float): Sample rate, Hz.
    order (int): Order of the filter. Default is 5.

    Returns:
    numpy.ndarray: Filtered matrix.
    """
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data, axis=0)
        return y

    # Apply the filter to each column
    filtered_data = butter_lowpass_filter(data, cutoff, fs, order)
    return filtered_data


# def extract_df_in_temp(df, temp, thresh=0.1):
#     index = abs(df.iloc[:, 62] - temp) < thresh
#     return df[index], index


def extract_df_in_temp(df, temp_list, thresh=0.1):
    indices = []
    filtered_dfs = []
    for temp in temp_list:
        index = abs(df.iloc[:, 62] - temp) < thresh
        indices.append(index)
        filtered_dfs.append(df[index])
    combined_df = pd.concat(filtered_dfs).drop_duplicates().reset_index(drop=True)
    combined_index = indices[0]
    for index in indices[1:]:
        combined_index |= index
    return combined_df, combined_index


def remove_background(df, window_size=10, thresh=0.1):
    out = None
    for j in range(32, 37):
        temp, index = extract_df_in_temp(df, j, thresh)
        # temp.iloc[:,2:62] = temp.iloc[:, 2:62] - temp.iloc[0, 2:62]
        # df.loc[index, df.columns[2:62]] = moving_average(temp.iloc[:, 2:62].values, window_size)
        temp.iloc[:, 2:62] = moving_average(temp.iloc[:, 2:62].values,
                                            window_size)
        # temp.iloc[:, 2:62] = baseline_correction(temp.iloc[:,2:62].values)
        if out is None:
            out = temp
        else:
            out = pd.concat([out, temp])
    return df


def remove_basic_background(df):
    index = abs(df.iloc[:, 62] - 32) < 0.001
    df.iloc[:, 2:62] = (df.iloc[:, 2:62].values - df[index].iloc[-1, 2:62].values)/df.iloc[:, 59].values.reshape(-1,1) # best
    # df.iloc[:, 2:62] = (df.iloc[:, 2:62].values - df[index].iloc[0, 2:62].values)/df.iloc[:, 2:62].values 
    # df.iloc[:, 2:62] = (df.iloc[:, 2:62].values - df[index].iloc[0, 2:62].values) # 

    return df


def remove_average_background(df, base_temp=33):
    index = abs(df.iloc[:, 62] - base_temp) < 0.001
    base_df = df[index]
    base_spec = np.mean(base_df.iloc[:, 2:62].values, axis=0)
    df.iloc[:, 2:62] = df.iloc[:, 2:62] - base_spec
    return df, base_spec