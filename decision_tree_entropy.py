import numpy as np


# Hàm tính Entropy
def entropy(labels):
    classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    # Tránh log(0) bằng cách bỏ qua các xác suất bằng 0
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


# Lớp Node cho cây quyết định
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, prob=None):
        self.feature = feature  # Chỉ số đặc trưng để phân chia
        self.threshold = threshold  # Ngưỡng phân chia
        self.left = left  # Nhánh bên trái
        self.right = right  # Nhánh bên phải
        self.value = value  # Nhãn dự đoán (lá)
        self.prob = prob  # Xác suất dự đoán (lá)


# Lớp Decision Tree
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    def find_best_split(self, X, y):
        n_features = X.shape[1]
        best_gain = -float('inf')  # Tìm Information Gain lớn nhất
        best_feature = None
        best_threshold = None

        # Tính Entropy của tập dữ liệu gốc
        parent_entropy = entropy(y)

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue

                # Tính Entropy của tập con
                entropy_left = entropy(y[left_idx])
                entropy_right = entropy(y[right_idx])

                # Tính weighted Entropy
                weighted_entropy = (len(y[left_idx]) * entropy_left + len(y[right_idx]) * entropy_right) / len(y)

                # Tính Information Gain
                gain = parent_entropy - weighted_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        majority_prob = counts[np.argmax(counts)] / n_samples

        # Điều kiện dừng
        if depth >= self.max_depth or len(np.unique(y)) == 1 or n_samples < 2:
            return Node(value=majority_class, prob=majority_prob)

        # Tìm phân chia tốt nhất
        feature, threshold = self.find_best_split(X, y)
        if feature is None:
            return Node(value=majority_class, prob=majority_prob)

        # Chia dữ liệu
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        # Đệ quy xây dựng cây
        left_node = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right_node = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_one(self, x):
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return [node.value, node.prob]

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


# Hàm chia train/test
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_size = int(test_size * n_samples)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test


# Tạo dữ liệu ngẫu nhiên
np.random.seed(42)
n_samples = 7000
n_features = 100
X = np.random.randn(n_samples, n_features)  # Dữ liệu 100 chiều
y = np.random.randint(0, 2, n_samples)  # Nhãn nhị phân (0 hoặc 1)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Khởi tạo và huấn luyện cây quyết định
tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Dự đoán trên tập test
predictions = tree.predict(X_test)

# Tính độ chính xác
accuracy = np.mean(predictions[:, 0] == y_test)
print(f"Độ chính xác trên tập test: {accuracy:.3f}")

# In một vài kết quả mẫu
print("\nKết quả giảm chiều (5 mẫu đầu tiên trong tập test):")
for i in range(5):
    print(
        f"Mẫu {i + 1}: Vector 100 chiều -> Vector 2 chiều: [Nhãn: {predictions[i, 0]}, Xác suất: {predictions[i, 1]:.3f}]")