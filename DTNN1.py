# ================= 超参数配置区 =================
HYPERPARAMS = {
    'layers': [4, 64, 64, 128, 128, 1],  # 网络结构 - 输入4个，输出1个(RF)
    'learning_rate': 0.001,  # 初始学习率
    'N_iter': 50000,  # Adam 训练总轮数
    'N_interv': 100,  # 打印/记录间隔
    'patience': 5000,  # 学习率衰减耐心 (建议适当调小, 50000太长了)
    'min_lr': 1e-6,  # 最小学习率
    'model_dir': './rf_Q_model_TF2',  # 修改保存路径以防覆盖
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
import scipy.optimize
from sklearn.metrics import r2_score
import matplotlib

matplotlib.use('Agg')  # 强制使用无界面后端

# GPU 配置 (TF2 方式)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


###############################################################################
################################ DTNN Class (TF2) #############################
###############################################################################
class DTNN:
    def __init__(self, Data=None, layers=None, N_out=None, lb=None, ub=None, model_path=None):
        # 确保全局使用 float64 以保证精度
        tf.keras.backend.set_floatx('float64')

        if model_path is not None:
            self.load_metadata(model_path)
            self.build_model()
            self.load_model(model_path)
        elif Data is not None and layers is not None and N_out is not None and lb is not None and ub is not None:
            self.lb_input = tf.cast(lb[0:4], tf.float64)
            self.ub_input = tf.cast(ub[0:4], tf.float64)
            self.lb_output = tf.cast(lb[4:5], tf.float64)
            self.ub_output = tf.cast(ub[4:5], tf.float64)
            self.layers_config = layers

            # 提取数据
            self.X_train = tf.cast(Data[:, 0:4], tf.float64)
            self.Y_train = tf.cast(Data[:, 4:5], tf.float64)

            self.build_model()
        else:
            raise ValueError(
                "Provide model_path to load, or provide Data, layers, N_out, lb, ub to create a new model.")

    def build_model(self):
        """使用 Keras Functional API 构建带有硬约束的模型"""
        inputs = tf.keras.Input(shape=(4,), dtype=tf.float64)

        # 1. 输入归一化 [-1, 1]
        delta = self.ub_input - self.lb_input
        delta_safe = tf.where(tf.equal(delta, 0.0), tf.ones_like(delta) * 1e-99, delta)
        H = 2.0 * (inputs - self.lb_input) / delta_safe - 1.0

        # 2. 多层感知机 (MLP) - Tanh 激活函数
        for units in self.layers_config[1:-1]:
            # 初始化器：使用截断正态分布
            initializer = tf.keras.initializers.TruncatedNormal(stddev=np.sqrt(2 / (inputs.shape[1] + units)))
            H = tf.keras.layers.Dense(units, activation='tanh',
                                      kernel_initializer=initializer,
                                      dtype=tf.float64)(H)

        # 输出层 (无激活函数)
        initializer = tf.keras.initializers.TruncatedNormal(stddev=np.sqrt(2 / (self.layers_config[-2] + 1)))
        RF_raw = tf.keras.layers.Dense(1, activation=None,
                                       kernel_initializer=initializer,
                                       dtype=tf.float64)(H)

        # 3. 物理硬约束逻辑
        d = inputs[:, 0:1]  # Depth 位于第0列

        # 约束1：最大不超过 1.0，最小不低于 0.0
        RF_capped = tf.clip_by_value(RF_raw, 0.0, 1.0)

        # 约束2：Depth > 300 时，RF = 1.0
        condition_high = tf.greater(d, 300.0)
        RF_step1 = tf.where(condition_high, tf.ones_like(RF_capped, dtype=tf.float64), RF_capped)

        # 约束3：Depth == 0 时，RF = 0.0
        condition_zero = tf.equal(d, 0.0)
        RF_final = tf.where(condition_zero, tf.zeros_like(RF_capped, dtype=tf.float64), RF_step1)

        self.model = tf.keras.Model(inputs=inputs, outputs=RF_final)

    def min_max_norm(self, y):
        return 2.0 * (y - self.lb_output) / (self.ub_output - self.lb_output) - 1.0

    @tf.function
    def compute_loss(self, y_true, y_pred):
        """计算归一化后的 MSE Loss"""
        y_true_norm = self.min_max_norm(y_true)
        y_pred_norm = self.min_max_norm(y_pred)
        return tf.reduce_sum(tf.square(y_true_norm - y_pred_norm))

    @tf.function
    def train_step_adam(self, X, Y, optimizer):
        """Adam 单步训练 (使用 GradientTape)"""
        with tf.GradientTape() as tape:
            pred = self.model(X, training=True)
            loss = self.compute_loss(Y, pred)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, pred

    def nn_train(self, N_iter, N_interv):
        """混合优化训练过程 (Adam -> L-BFGS-B)"""
        start_time = time.time()
        self.loss_history = []
        self.accuracy_history = []
        self.lr_history = []
        self.r2_history = []
        self.mae_history, self.mse_history, self.rmse_history, self.mape_history = [], [], [], []

        current_lr = HYPERPARAMS['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)

        patience = HYPERPARAMS['patience']
        patience_counter = 0
        best_loss = float('inf')
        min_lr = HYPERPARAMS['min_lr']

        print("=== Phase 1: Adam Optimization ===")
        for it in range(N_iter):
            loss_value, RF_pred = self.train_step_adam(self.X_train, self.Y_train, optimizer)

            if it % N_interv == 0:
                elapsed = time.time() - start_time
                loss_val_np = loss_value.numpy()
                self.loss_history.append(loss_val_np)
                self.lr_history.append(current_lr)

                y_true = self.Y_train.numpy()
                y_pred = RF_pred.numpy()

                accuracy = self.calculate_accuracy(y_true, y_pred)
                self.accuracy_history.append(accuracy)

                r2 = r2_score(y_true, y_pred)
                mae = np.mean(np.abs(y_true - y_pred))
                mse = np.mean(np.square(y_true - y_pred))
                rmse = np.sqrt(mse)

                mask = np.abs(y_true) > 1
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / np.abs(y_true[mask]))) * 100 if np.any(
                    mask) else 0.0

                self.r2_history.append(r2)
                self.mae_history.append(mae)
                self.mse_history.append(mse)
                self.rmse_history.append(rmse)
                self.mape_history.append(mape)

                # 学习率衰减逻辑
                if loss_val_np < best_loss:
                    best_loss = loss_val_np
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience and current_lr > min_lr:
                    current_lr *= 0.1
                    optimizer.learning_rate.assign(current_lr)
                    patience_counter = 0
                    print(f'Learning rate decreased to {current_lr}')

                print('It: %d, Loss: %.3e, Acc: %.2f%%, LR: %.6f, R2: %.4f, RMSE: %.4f, Time: %.2f' %
                      (it, loss_val_np, accuracy, current_lr, r2, rmse, elapsed))
                start_time = time.time()

        print("=== Phase 2: L-BFGS-B Optimization ===")
        self.train_lbfgs()

        # 最终预测
        final_pred = self.model(self.X_train, training=False).numpy()
        return self.loss_history, final_pred, self.accuracy_history, self.lr_history, self.r2_history, self.mae_history, self.mse_history, self.rmse_history, self.mape_history

    # ==========================
    # L-BFGS-B (SciPy) 替代实现
    # ==========================
    def _set_weights_from_1d(self, weights_1d):
        """将 scipy 优化器的 1D 数组映射回 Keras 模型的权重层"""
        weights_list = []
        idx = 0
        for w in self.model.trainable_variables:
            shape = w.shape
            size = np.prod(shape)
            weights_list.append(tf.reshape(weights_1d[idx:idx + size], shape))
            idx += size
        self.model.set_weights([w.numpy() for w in weights_list])

    def _get_weights_1d(self):
        """将 Keras 模型的权重展平为 1D 数组供 scipy 使用"""
        return np.concatenate([w.numpy().flatten() for w in self.model.trainable_variables])

    @tf.function
    def _compute_loss_and_grads(self, X, Y):
        """计算损失和一维化的梯度"""
        with tf.GradientTape() as tape:
            pred = self.model(X, training=True)
            loss = self.compute_loss(Y, pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_1d = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return loss, grads_1d

    def train_lbfgs(self):
        """使用 Scipy L-BFGS-B 进行精细调优"""

        def loss_and_grads_wrapper(weights_1d):
            self._set_weights_from_1d(weights_1d)
            loss, grads_1d = self._compute_loss_and_grads(self.X_train, self.Y_train)

            # 记录 L-BFGS 过程中的 Loss
            loss_np = loss.numpy().astype('float64')
            print('L-BFGS-B Loss: %.3e' % loss_np)

            return loss_np, grads_1d.numpy().astype('float64')

        init_weights = self._get_weights_1d()
        scipy.optimize.minimize(
            fun=loss_and_grads_wrapper,
            x0=init_weights,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50000, 'maxfun': 50000, 'ftol': 1e-6, 'gtol': 1e-3}
        )

    # ==========================
    # 辅助与预测函数
    # ==========================
    def nn_predict(self, Te, N_each, N_te):
        """预测函数 (支持批次预测)"""
        rf_te = np.zeros((Te.shape[0], 1))
        for i in range(N_te):
            testX = Te[N_each * i: N_each * (i + 1), 0:4]
            # TF2 直接调用模型预测
            pred = self.model(tf.cast(testX, tf.float64), training=False)
            rf_te[j + N_each * i, :] = pred.numpy()
        return rf_te

    def calculate_accuracy(self, actual, predicted, threshold=0.05):
        epsilon = 1e-10
        relative_error = np.abs((actual - predicted) / (np.abs(actual) + epsilon))
        return np.mean(relative_error < threshold) * 100

    def save_model(self, model_dir='./saved_model'):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # TF2: 保存权重
        model_weights_path = os.path.join(model_dir, "weights.h5")
        self.model.save_weights(model_weights_path)

        metadata = {
            'layers': self.layers_config,
            'lb_output': self.lb_output.numpy().tolist(),
            'ub_output': self.ub_output.numpy().tolist(),
            'lb_input': self.lb_input.numpy().tolist(),
            'ub_input': self.ub_input.numpy().tolist()
        }
        metadata_path = os.path.join(model_dir, "metadata.npy")
        np.save(metadata_path, metadata)
        print(f"Model saved to {model_weights_path}")

    def load_metadata(self, model_dir):
        metadata_path = os.path.join(model_dir, "metadata.npy")
        metadata = np.load(metadata_path, allow_pickle=True).item()

        self.layers_config = metadata['layers']
        self.lb_output = tf.constant(metadata['lb_output'], dtype=tf.float64)
        self.ub_output = tf.constant(metadata['ub_output'], dtype=tf.float64)
        self.lb_input = tf.constant(metadata['lb_input'], dtype=tf.float64)
        self.ub_input = tf.constant(metadata['ub_input'], dtype=tf.float64)

    def load_model(self, model_dir):
        model_weights_path = os.path.join(model_dir, "weights.h5")
        self.model.load_weights(model_weights_path)
        print(f"Model loaded from {model_weights_path}")

    # ===== 下方的可视化和评估函数无需修改，直接保留即可 =====
    def error_indicator(self, actu, pred, N_out):
        # 此函数原封不动... (为缩短篇幅省略，可直接从你原代码复制过来)
        pass

    def AP_scatter(self, actu, pred, N_out, save_path=None):
        # 此函数原封不动...
        pass

    def Loss_curve(self, history, accuracy=None, lr_history=None, save_path=None):
        # 此函数原封不动...
        pass

    def plot_comparison(self, actu, pred, N_out, title="Training Data", save_path=None):
        # 此函数原封不动...
        pass


###############################################################################
################################ Main Function ################################
###############################################################################
if __name__ == "__main__":
    # 读取数据 (请确保 DATA_1%.csv 存在)
    Tr = pd.read_csv('DATA_1%.csv', header=None).values
    Tv = pd.read_csv('DATA_1%.csv', header=None).values
    Te = pd.read_csv('DATA_1%.csv', header=None).values

    D_input = 4
    N_out = 1
    N_total = Te.shape[0]

    N_te = 1
    N_each = N_total

    Data = np.vstack((Tr, Tv))
    lb = np.min(Data, axis=0)
    ub = np.max(Data, axis=0)

    model_dir = HYPERPARAMS['model_dir']

    # 初始化模型
    model = DTNN(Data, HYPERPARAMS['layers'], N_out, lb, ub)

    # 训练模型
    print("[INFO] 开始训练...")
    history, rf_pred, acc_hist, lr_hist, r2_hist, mae_hist, mse_hist, rmse_hist, mape_hist = model.nn_train(
        N_iter=HYPERPARAMS['N_iter'],
        N_interv=HYPERPARAMS['N_interv']
    )

    # 保存模型
    model.save_model(model_dir)