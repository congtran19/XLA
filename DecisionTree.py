import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Hàm tính entropy
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))

# Hàm chọn đặc trưng tốt nhất dựa trên entropy
def best_split(X, y):
    best_gain = -1
    best_feature, best_threshold = None, None
    current_entropy = entropy(y)
    n_features = X.shape[1]
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            left_idx = X[:, feature] <= t
            right_idx = X[:, feature] > t
            if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                continue
            left_entropy = entropy(y[left_idx])
            right_entropy = entropy(y[right_idx])
            p_left = len(y[left_idx]) / len(y)
            gain = current_entropy - (p_left * left_entropy + (1 - p_left) * right_entropy)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = t
    if best_feature is not None:
        print(f"[DEBUG] Best split - Feature: {best_feature}, Threshold: {best_threshold}, Gain: {best_gain:.4f}")
    return best_feature, best_threshold

# Định nghĩa cây
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

# Xây dựng cây đệ quy
def build_tree(X, y, depth=0, max_depth=5):
    if len(np.unique(y)) == 1 or depth >= max_depth:
        value = np.bincount(y).argmax()
        print(f"[DEBUG] Creating leaf node at depth {depth} with value: {value}")
        return DecisionTreeNode(value=value)

    feature, threshold = best_split(X, y)
    if feature is None:
        value = np.bincount(y).argmax()
        print(f"[DEBUG] Creating leaf node at depth {depth} (no split found) with value: {value}")
        return DecisionTreeNode(value=value)

    print(f"[DEBUG] Splitting on feature {feature} at threshold {threshold} (depth {depth})")
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth)
    return DecisionTreeNode(feature, threshold, left, right)

# Hàm dự đoán
def predict_one(x, tree):
    while not tree.is_leaf_node():
        if x[tree.feature] <= tree.threshold:
            tree = tree.left
        else:
            tree = tree.right
    return tree.value

def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])

# Sinh dữ liệu ngẫu nhiên thay vì đọc từ CSV
np.random.seed(42)
X = np.random.rand(7500, 100)  # 7500 mẫu, mỗi mẫu 100 đặc trưng
y = np.random.randint(0, 2, 7500)  # Nhãn 0 hoặc 1

# Tách dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây cây và dự đoán
tree = build_tree(X_train, y_train, max_depth=5)
y_pred = predict(X_test, tree)

# Đánh giá
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
