# ================= 超参数配置区 =================
HYPERPARAMS = {
    'layers': [4, 64, 64, 128, 128, 1],  # 网络结构 - 输入4个，输出1个(RF)
    'learning_rate': 0.001,  # 初始学习率
    'N_iter': 50000,  # 训练总轮数
    'N_interv': 100,  # 打印/记录间隔
    'patience': 50000,  # 学习率衰减耐心
    'min_lr': 1e-6,  # 最小学习率
    'model_dir': './rf_Q_model_1%',  # 修改保存路径以防覆盖
    'train_model': True,  # 是否训练模型
    'resume_train': False,  # 是否断点续训
}
# ==============================================

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import r2_score
import matplotlib

matplotlib.use('Agg')  # 强制使用无界面后端

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True

sess = tf.compat.v1.Session(config=config)


###############################################################################
####################### DTNN for Q-Value Prediction     ######################
###############################################################################

def initialize_nn(layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers - 1):
        W = weights_init(size=[layers[l], layers[l + 1]])
        b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float64), dtype=tf.float64)
        weights.append(W)
        biases.append(b)
    return weights, biases


def weights_init(size):
    in_dim = size[0]
    out_dim = size[1]
    weights_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=weights_stddev, dtype=tf.float64),
                       dtype=tf.float64)


def neural_net(X, weights, biases):
    num_layers = len(weights) + 1
    H = X
    for l in range(0, num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    # 修改：不再分割P和Q，直接返回输出 Y (即 RF)
    return Y


###############################################################################
################################ DTNN Class ##################################
###############################################################################
class DTNN:
    def __init__(self, Data=None, layers=None, N_out=None, lb=None, ub=None, model_path=None):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        if model_path is not None:
            self.load_model(model_path)
        elif Data is not None and layers is not None and N_out is not None and lb is not None and ub is not None:
            # Domain Boundary - 修改索引以匹配新数据
            # 数据结构: [Depth, Mag, Span, Qval, RF]
            # 输入: 前4列 (0-3)
            # 输出: 第5列 (4)
            self.lb_input = lb[0:4]
            self.ub_input = ub[0:4]
            self.lb_output = lb[4:5]
            self.ub_output = ub[4:5]

            # Initialize neural network
            self.nn_init(Data, layers)

            # Initialize variables
            init = tf.global_variables_initializer()
            self.sess.run(init)
        else:
            raise ValueError(
                "Either provide model_path to load an existing model or provide Data, layers, N_out, lb, ub to create a new model")

    def nn_init(self, Data, layers):
        # 适配新数据格式 - 5列数据
        # Input (4 columns)
        self.Depth = Data[:, 0:1]
        self.Mag = Data[:, 1:2]
        self.Span = Data[:, 2:3]
        self.Qval = Data[:, 3:4]
        # Output (1 column)
        self.RF = Data[:, 4:5]

        # Layers for Solution
        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = initialize_nn(layers)
        self.saver = tf.train.Saver(var_list=[self.weights[l] for l in range(len(self.layers) - 1)]
                                             + [self.biases[l] for l in range(len(self.layers) - 1)])

        # tf placeholders - 修改为新变量名
        self.Depth_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.Mag_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.Span_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.Qval_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.RF_tf = tf.placeholder(tf.float64, shape=[None, 1])

        # tf graphs
        self.RF_pred = self.nn_output(self.Depth_tf, self.Mag_tf, self.Span_tf, self.Qval_tf)

        # loss for Solution
        # 归一化
        RF_tf_nr = self.min_max_norm(self.RF_tf)
        RF_pred_nr = self.min_max_norm(self.RF_pred)

        # 计算 Loss (因为只有1个输出，不需要原来的复杂权重分配，直接用MSE)
        self.nn_loss = tf.reduce_sum(tf.square(RF_tf_nr - RF_pred_nr))

        # Optimizer (保持原样)
        self.nn_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.nn_loss,
                                                                   var_list=self.weights + self.biases,
                                                                   method='L-BFGS-B',
                                                                   options={'maxiter': 50000,
                                                                            'maxfun': 50000,
                                                                            'maxcor': 100,
                                                                            'maxls': 100,
                                                                            'gtol': 1e-03})

        # 自适应学习率 (保持原样)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.placeholder(tf.float64, shape=[])
        self.nn_optimizer_Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.nn_train_op_Adam = self.nn_optimizer_Adam.minimize(
            self.nn_loss,
            global_step=self.global_step,
            var_list=self.weights + self.biases)

    def nn_output(self, d, m, s, q):
        # 拼接4个输入
        X = tf.concat([d, m, s, q], 1)
        delta = self.ub_input - self.lb_input
        delta_safe = tf.where(tf.equal(delta, 0.0), tf.ones_like(delta) * 1e-99, delta)
        H = 2.0 * (X - self.lb_input) / delta_safe - 1.0

        # 神经网络计算原始输出
        RF_raw = neural_net(H, self.weights, self.biases)

        # === 新增约束逻辑 ===

        # 约束1：RF 最大输出为 1.0
        # 使用 tf.minimum 取两者的较小值
        RF_capped = tf.clip_by_value(RF_raw, 0.0, 1.0)

        # 约束2：当 Depth (d) > 300 时，RF 默认输出为 1.0
        # 2. 条件A：当 Depth > 300 时，RF = 1.0
        condition_high = tf.greater(d, 300.0)
        RF_step1 = tf.where(condition_high, tf.ones_like(RF_capped, dtype=tf.float64), RF_capped)

        # 3. 条件B：当 Depth == 0 时，RF = 0.0
        # (注：这一步放在最后，优先级最高，确保深度为0时强制输出0)
        condition_zero = tf.equal(d, 0.0)
        RF_final = tf.where(condition_zero, tf.zeros_like(RF_capped, dtype=tf.float64), RF_step1)


        return RF_final

    def min_max_norm(self, y):
        # 修改为单变量归一化
        Y_norm = 2.0 * (y - self.lb_output) / (self.ub_output - self.lb_output) - 1.0
        return Y_norm

    def callback(self, loss):
        print('Loss: %.3e' % (loss))
        self.loss_history.append(loss)

    def save_model(self, model_dir='./saved_model'):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, "model.ckpt")
        save_path = self.saver.save(self.sess, model_path)

        metadata = {
            'layers': self.layers,
            'lb_output': self.lb_output.tolist(),
            'ub_output': self.ub_output.tolist(),
            'lb_input': self.lb_input.tolist(),
            'ub_input': self.ub_input.tolist()
        }

        metadata_path = os.path.join(model_dir, "metadata.npy")
        np.save(metadata_path, metadata)

        print(f"Model saved to {save_path}")
        print(f"Metadata saved to {metadata_path}")

        return save_path

    def load_model(self, model_dir):
        metadata_path = os.path.join(model_dir, "metadata.npy")
        metadata = np.load(metadata_path, allow_pickle=True).item()

        self.layers = metadata['layers']
        self.lb_output = np.array(metadata['lb_output'])
        self.ub_output = np.array(metadata['ub_output'])
        self.lb_input = np.array(metadata['lb_input'])
        self.ub_input = np.array(metadata['ub_input'])

        self.weights, self.biases = initialize_nn(self.layers)
        self.saver = tf.train.Saver(var_list=[self.weights[l] for l in range(len(self.layers) - 1)]
                                             + [self.biases[l] for l in range(len(self.layers) - 1)])

        # Placeholders
        self.Depth_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.Mag_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.Span_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.Qval_tf = tf.placeholder(tf.float64, shape=[None, 1])
        self.RF_tf = tf.placeholder(tf.float64, shape=[None, 1])

        self.RF_pred = self.nn_output(self.Depth_tf, self.Mag_tf, self.Span_tf, self.Qval_tf)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        model_path = os.path.join(model_dir, "model.ckpt")
        self.saver.restore(self.sess, model_path)

        print(f"Model loaded from {model_path}")

    def calculate_accuracy(self, actual, predicted, threshold=0.05):
        epsilon = 1e-10
        relative_error = np.abs((actual - predicted) / (np.abs(actual) + epsilon))
        accuracy = np.mean(relative_error < threshold) * 100
        return accuracy

    def nn_train(self, N_iter, N_interv):
        # 更新 feed_dict
        tf_dict = {self.Depth_tf: self.Depth,
                   self.Mag_tf: self.Mag,
                   self.Span_tf: self.Span,
                   self.Qval_tf: self.Qval,
                   self.RF_tf: self.RF}

        start_time = time.time()
        self.loss_history = []
        self.accuracy_history = []
        self.lr_history = []
        self.r2_history = []
        self.mae_history = []
        self.mse_history = []
        self.rmse_history = []
        self.mape_history = []

        current_lr = HYPERPARAMS['learning_rate']
        patience = HYPERPARAMS['patience']
        patience_counter = 0
        best_loss = float('inf')
        min_lr = HYPERPARAMS['min_lr']

        for it in range(N_iter):
            tf_dict[self.learning_rate] = current_lr
            self.sess.run(self.nn_train_op_Adam, tf_dict)

            if it % N_interv == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.nn_loss, tf_dict)
                self.loss_history.append(loss_value)
                self.lr_history.append(current_lr)

                RF_pred = self.sess.run(self.RF_pred, tf_dict)
                accuracy = self.calculate_accuracy(self.RF, RF_pred)
                self.accuracy_history.append(accuracy)

                y_true = self.RF
                y_pred = RF_pred
                r2 = r2_score(y_true, y_pred)
                mae = np.mean(np.abs(y_true - y_pred))
                mse = np.mean(np.square(y_true - y_pred))
                rmse = np.sqrt(mse)

                mask = np.abs(y_true) > 1
                if np.any(mask):
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100
                else:
                    mape = 0.0

                self.r2_history.append(r2)
                self.mae_history.append(mae)
                self.mse_history.append(mse)
                self.rmse_history.append(rmse)
                self.mape_history.append(mape)

                if loss_value < best_loss:
                    best_loss = loss_value
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience and current_lr > min_lr:
                    current_lr *= 0.1
                    patience_counter = 0
                    print(f'Learning rate decreased to {current_lr}')

                print(
                    'It: %d, Loss: %.3e, Accuracy: %.2f%%, LR: %.6f, R2: %.4f, MAE: %.4f, MSE: %.4f, RMSE: %.4f, MAPE: %.2f%%, Time: %.2f' %
                    (it, loss_value, accuracy, current_lr, r2, mae, mse, rmse, mape, elapsed))
                start_time = time.time()

        self.nn_optimizer.minimize(self.sess,
                                   feed_dict=tf_dict,
                                   fetches=[self.nn_loss],
                                   loss_callback=self.callback)

        RF_pred = self.sess.run(self.RF_pred, tf_dict)

        return self.loss_history, RF_pred, self.accuracy_history, self.lr_history, self.r2_history, self.mae_history, self.mse_history, self.rmse_history, self.mape_history

    def nn_predict(self, Te, N_each, N_te):
        rf_te = np.zeros((Te.shape[0], 1))

        for i in range(N_te):
            testX = Te[N_each * i: N_each * (i + 1), :]
            for j in range(0, N_each):
                # 输入列: 0, 1, 2, 3
                tf_dict = {self.Depth_tf: testX[j:j + 1, 0:1],
                           self.Mag_tf: testX[j:j + 1, 1:2],
                           self.Span_tf: testX[j:j + 1, 2:3],
                           self.Qval_tf: testX[j:j + 1, 3:4]}

                rf_te[j + N_each * i, :] = self.sess.run(self.RF_pred, tf_dict)

        return rf_te

    # 保持原有的 error_indicator 逻辑
    def error_indicator(self, actu, pred, N_out):
        names = locals()
        model_order = 1
        Indicator = np.zeros((N_out + 1, model_order * 5))

        for mi in range(1, model_order + 1):
            names['R' + str(mi)] = 0
            names['MAE' + str(mi)] = 0
            names['MSE' + str(mi)] = 0
            names['RMSE' + str(mi)] = 0
            names['MAPE' + str(mi)] = 0

            for oi in range(N_out):
                y_true = actu[:, oi]
                y_pred = pred[:, oi]
                r2 = r2_score(y_true, y_pred)
                mae = np.mean(np.abs(y_true - y_pred))
                mse = np.mean(np.square(y_true - y_pred))
                rmse = np.sqrt(mse)
                mask = np.abs(y_true) > 1
                if np.any(mask):
                    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100
                else:
                    mape = 0.0

                names['R' + str(mi)] += r2
                names['MAE' + str(mi)] += mae
                names['MSE' + str(mi)] += mse
                names['RMSE' + str(mi)] += rmse
                names['MAPE' + str(mi)] += mape

                Indicator[oi, mi - 1] = r2
                Indicator[oi, mi + model_order - 1] = mae
                Indicator[oi, mi + 2 * model_order - 1] = mse
                Indicator[oi, mi + 3 * model_order - 1] = rmse
                Indicator[oi, mi + 4 * model_order - 1] = mape

            Indicator[N_out, mi - 1] = names['R' + str(mi)] / N_out
            Indicator[N_out, mi + model_order - 1] = names['MAE' + str(mi)] / N_out
            Indicator[N_out, mi + 2 * model_order - 1] = names['MSE' + str(mi)] / N_out
            Indicator[N_out, mi + 3 * model_order - 1] = names['RMSE' + str(mi)] / N_out
            Indicator[N_out, mi + 4 * model_order - 1] = names['MAPE' + str(mi)] / N_out

        return Indicator

    # 保持原有的 AP_scatter 逻辑
    def AP_scatter(self, actu, pred, N_out, save_path=None):
        from sklearn.metrics import r2_score
        import numpy as np

        plt.rcParams["figure.figsize"] = (6 * N_out, 6)  # 调整尺寸
        fig, ax = plt.subplots(1, N_out)

        # 兼容 N_out=1 的情况（此时 ax 不是列表）
        if N_out == 1:
            ax = [ax]

        for i in range(N_out):
            y_true = actu[:, i]
            y_pred = pred[:, i]
            r2 = r2_score(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean(np.square(y_true - y_pred))
            rmse = np.sqrt(mse)
            mask = np.abs(y_true) > 1
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100
            else:
                mape = 0.0

            ax[i].scatter(y_true, y_pred, marker='o', alpha=0.7)

            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            margin = (max_val - min_val) * 0.1
            ax[i].set_xlim(min_val - margin, max_val + margin)
            ax[i].set_ylim(min_val - margin, max_val + margin)

            ax[i].plot([min_val - margin, max_val + margin],
                       [min_val - margin, max_val + margin],
                       'k--', alpha=0.75)

            metrics_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}\nMAPE = {mape:.2f}%'
            ax[i].text(0.05, 0.95, metrics_text,
                       transform=ax[i].transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

            ax[i].set_xlabel('Actual RF')
            ax[i].set_ylabel('Predicted RF')
            ax[i].set_title(f'Output {i + 1}')
            ax[i].grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        return fig

    # 保持原有的 Loss_curve 逻辑
    def Loss_curve(self, history, accuracy=None, lr_history=None, save_path=None):
        plt.rcParams["figure.figsize"] = (10, 6)
        fig, ax1 = plt.subplots(1, 1)

        color = 'tab:blue'
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(history, color=color, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_yscale('log')
        ax1.grid(True, linestyle='--', alpha=0.3)

        if accuracy is not None:
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Accuracy (%)', color=color)
            ax2.plot(accuracy, color=color, label='Accuracy')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, 100)

            if lr_history is not None:
                ax3 = ax1.twinx()
                ax3.spines["right"].set_position(("axes", 1.1))
                color = 'tab:green'
                ax3.set_ylabel('Learning Rate', color=color)
                ax3.plot(lr_history, color=color, label='LR', linestyle='-.')
                ax3.tick_params(axis='y', labelcolor=color)
                ax3.set_yscale('log')

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines3, labels3 = ax3.get_legend_handles_labels()
                ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper right')
            else:
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title('Training Progress')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
        return fig

    # 保持原有的 plot_comparison 逻辑
    def plot_comparison(self, actu, pred, N_out, title="Training Data", save_path=None):
        plt.rcParams["figure.figsize"] = (15, 6 * N_out)
        fig, axes = plt.subplots(N_out, 1)

        if N_out == 1:
            axes = [axes]

        for i in range(N_out):
            ax = axes[i]
            x = np.arange(len(actu[:, i]))

            ax.plot(x, actu[:, i], 'b-', label='Actual', linewidth=2)
            ax.plot(x, pred[:, i], 'r--', label='Predicted', linewidth=2)

            y_true = actu[:, i]
            y_pred = pred[:, i]
            r2 = r2_score(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean(np.square(y_true - y_pred))
            rmse = np.sqrt(mse)
            mask = np.abs(y_true) > 1
            if np.any(mask):
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100
            else:
                mape = 0.0
            metrics_text = f'R² = {r2:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAPE = {mape:.2f}%'

            ax.text(0.5, 0.02, metrics_text,
                    transform=ax.transAxes,
                    fontsize=10,
                    horizontalalignment='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

            ax.set_title(f'Comparison of Actual and Predicted Values for Output {i + 1}')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Value')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.fill_between(x, actu[:, i], pred[:, i], color='gray', alpha=0.3, label='Error')

        plt.suptitle(f'{title} Prediction Comparison', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path:
            plt.savefig(save_path)
        plt.close()
        return fig


###############################################################################
################################ Main Function ################################
###############################################################################

if __name__ == "__main__":

    # Data
 #   Tr = pd.read_csv('odd_rows.csv', header=None).values
  #  Tv = pd.read_csv('odd_rows.csv', header=None).values
  #  Te = pd.read_csv('even_rows.csv', header=None).values

    Tr = pd.read_csv('DATA_1%.csv', header=None).values
    Tv = pd.read_csv('DATA_1%.csv', header=None).values
    Te = pd.read_csv('DATA_1%.csv', header=None).values


    # 更新维度定义
    D_input = 4  # Depth, Mag, Span, Qval
    N_out = 1  # RF
    N_nc = 2  # (原代码保留项，此处无实际作用但保留定义)
    N_total = Te.shape[0]

    target = max(int(round(0.2 * N_total)), 1)
    N_te = None
    for candidate in range(target, 0, -1):
        if N_total % candidate == 0:
            N_te = candidate
            N_each = N_total // candidate
            break
    if N_te is None:
        N_te = 1
        N_each = N_total

    Data = np.vstack((Tr, Tv))

    lb = np.min(Data, axis=0)
    ub = np.max(Data, axis=0)

    # Model Training Mode
    train_model = HYPERPARAMS['train_model']
    resume_train = HYPERPARAMS.get('resume_train', False)
    model_dir = HYPERPARAMS['model_dir']

    if train_model and resume_train:
        print("[INFO] 加载已有模型，继续训练...")
        model = DTNN(Data, HYPERPARAMS['layers'], N_out, lb, ub, model_path=model_dir)
        history, rf_pred, accuracy_history, lr_history, r2_history, mae_history, mse_history, rmse_history, mape_history = model.nn_train(
            N_iter=HYPERPARAMS['N_iter'], N_interv=HYPERPARAMS['N_interv'])
        model.save_model(model_dir)
        N = min(len(history), len(accuracy_history), len(lr_history), len(r2_history), len(mae_history),
                len(mse_history), len(rmse_history), len(mape_history))
        training_metrics = np.column_stack((
            history[:N],
            accuracy_history[:N],
            lr_history[:N],
            r2_history[:N],
            mae_history[:N],
            mse_history[:N],
            rmse_history[:N],
            mape_history[:N]
        ))
        np.savetxt(model_dir + '/training_metrics.csv', training_metrics,
                   fmt='%0.10f', delimiter=',',
                   header='Loss,Accuracy,Learning_Rate,R2,MAE,MSE,RMSE,MAPE')
        model.Loss_curve(history, accuracy_history, lr_history,
                         save_path=os.path.join(model_dir, 'train_loss_curve.png'))
        tr_actu = Data[:, 4:5]  # RF
        tr_pred = rf_pred
        tr_error = model.error_indicator(tr_actu, tr_pred, N_out)
        model.AP_scatter(tr_actu, tr_pred, N_out, save_path=os.path.join(model_dir, 'train_scatter.png'))
        model.plot_comparison(tr_actu, tr_pred, N_out, title="Training Data",
                              save_path=os.path.join(model_dir, 'train_comparison.png'))
        np.savetxt(model_dir + '/out_training.csv', np.hstack((tr_actu, tr_pred)), fmt='%.10f', delimiter=',')
        np.savetxt(model_dir + '/out_training_error.csv', tr_error, fmt='%.10f', delimiter=',')
    elif train_model:
        model = DTNN(Data, HYPERPARAMS['layers'], N_out, lb, ub)
        history, rf_pred, accuracy_history, lr_history, r2_history, mae_history, mse_history, rmse_history, mape_history = model.nn_train(
            N_iter=HYPERPARAMS['N_iter'], N_interv=HYPERPARAMS['N_interv'])
        model.save_model(model_dir)
        N = min(len(history), len(accuracy_history), len(lr_history), len(r2_history), len(mae_history),
                len(mse_history), len(rmse_history), len(mape_history))
        training_metrics = np.column_stack((
            history[:N],
            accuracy_history[:N],
            lr_history[:N],
            r2_history[:N],
            mae_history[:N],
            mse_history[:N],
            rmse_history[:N],
            mape_history[:N]
        ))
        np.savetxt(model_dir + '/training_metrics.csv', training_metrics,
                   fmt='%0.10f', delimiter=',',
                   header='Loss,Accuracy,Learning_Rate,R2,MAE,MSE,RMSE,MAPE')
        model.Loss_curve(history, accuracy_history, lr_history,
                         save_path=os.path.join(model_dir, 'train_loss_curve.png'))
        tr_actu = Data[:, 4:5]  # RF
        tr_pred = rf_pred
        tr_error = model.error_indicator(tr_actu, tr_pred, N_out)
        model.AP_scatter(tr_actu, tr_pred, N_out, save_path=os.path.join(model_dir, 'train_scatter.png'))
        model.plot_comparison(tr_actu, tr_pred, N_out, title="Training Data",
                              save_path=os.path.join(model_dir, 'train_comparison.png'))
        np.savetxt(model_dir + '/out_training.csv', np.hstack((tr_actu, tr_pred)), fmt='%.10f', delimiter=',')
        np.savetxt(model_dir + '/out_training_error.csv', tr_error, fmt='%.10f', delimiter=',')
    else:
        model = DTNN(model_path=model_dir)

    # Testing with the model
    rf_pred_te = model.nn_predict(Te, N_each, N_te)
    te_actu = Te[:, 4:5]  # RF
    te_pred = rf_pred_te
    te_error = model.error_indicator(te_actu, te_pred, N_out)
    model.plot_comparison(te_actu, te_pred, N_out, title="Test Data",
                          save_path=os.path.join(model_dir, 'test_comparison.png'))
    model.AP_scatter(te_actu, te_pred, N_out, save_path=os.path.join(model_dir, 'test_scatter.png'))
    model.Loss_curve(history if train_model else [], accuracy_history if train_model else [],
                     lr_history if train_model else [], save_path=os.path.join(model_dir, 'test_loss_curve.png'))
    np.savetxt(model_dir + '/out_testing.csv', np.hstack((te_actu, te_pred)), fmt='%.10f', delimiter=',')
    np.savetxt(model_dir + '/out_testing_error.csv', te_error, fmt='%.10f', delimiter=',')