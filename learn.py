from sklearn import svm
import numpy as np
import pickle

# 假设你有一个数据集，其中X是特征（EAR和MAR值），y是标签（0表示警觉，1表示正常）
X = np.array([[0.3, 0.7], [0.5, 0.2], [0.1, 0.6], [0.4, 0.6]])  # 这只是一个示例，你需要使用实际的数据
y = np.array([0, 1, 0, 1])  # 这只是一个示例，你需要使用实际的数据

# 创建SVM模型
svm_model = svm.SVC()

# 使用数据集训练模型
svm_model.fit(X, y)

# 保存训练好的模型
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
