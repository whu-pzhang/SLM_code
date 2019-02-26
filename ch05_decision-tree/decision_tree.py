import numpy as np
import pandas as pd


def calc_entropy(x):
    """Compute entropy of dataset x

    Parameters
    ----------
    x : pandas.DataFrame
        training dataset with "label"

    Returns
    -------
    out: float
        Entropy of the input dataset
    """
    prob = x["label"].value_counts(normalize=True).values
    return np.sum(-prob*np.log2(prob))


def cond_entropy(x):
    """Compute condition entropy of dataset x

    Parameters
    ----------
    x : pandas.DataFrame

    Returns
    -------
    out : dict

    """
    feature_names = x.drop("label", axis=1).columns
    cond_entropy = {}

    for name in feature_names:
        # the classes under this feature
        sub_features = set(x[name])
        prob = x[name].value_counts(normalize=True)
        temp = 0
        for f in sub_features:
            temp += prob[f] * calc_entropy(x[x[name] == f])
        cond_entropy[name] = temp

    return cond_entropy


def info_gain(x):
    """Compute the information gain of dataset x

    Parameters
    ----------
    x : pandas.DataFrame

    Returns:
    --------
    out : dict

    """
    res = {}
    ent = calc_entropy(x)
    for key, value in cond_entropy(x).items():
        res[key] = ent - value
    return res


class TreeNode():
    """Binary tree node class

    """

    def __init__(self, node_type, label=None, feature=None):
        self.type = node_type
        self.label = label
        self.feature = feature
        self.tree = {}
        self.result = {"label": self.label,
                       "feature": self.feature, "tree": self.tree}

    def __repr__(self):
        return "{}".format(self.result)

    def add_node(self, val, tree):
        self.tree[val] = tree

    def predict(self, features):
        if self.type == "leaf":
            return self.label
        return self.tree[features[self.feature]].predict(features)


class DecisionTree():
    def __init__(self, eps=0.1):
        self.eps = eps
        self.tree = {}

    def train(self, X, y):
        """ID3 algorithm

        Parameters
        ----------
        X : pandas.DataFrame
            features dataset
        y : pandas.DataFrame
            labels dataset

        Returns
        -------
        TreeNode
            decision tree
        """

        # 实例中最大的类
        max_class = y.value_counts().sort_values(ascending=False).index[0]
        # ID3 algorithm
        # (1) 若D中所有实例属于同一类Ck，则T为单节点树，并将类Ck作为该节点的类标记，返回T
        if len(set(y)) == 1:
            return TreeNode("leaf", label=y.iloc[0])
        # (2) 若A为空集，将D中实例最大的类Ck作为该节点的类标记，返回T
        if len(X) == 0:
            return TreeNode("leaf", label=max_class)

        # (3)
        train_data = pd.concat([X, y], axis=1)
        train_data.rename(columns={train_data.columns[-1]: "label"})
        feature_info_gain = info_gain(train_data)
        # 按照增益大小降序排序
        max_feature_name, max_info_gain = sorted(
            feature_info_gain.items(), key=lambda d: d[1], reverse=True)[0]

        # (4)
        if max_info_gain < self.eps:
            return TreeNode("leaf", label=max_class)

        # (5) 构建子集
        node = TreeNode("internal", feature=max_feature_name)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop(
                [max_feature_name], axis=1)
            sub_node = self.train(sub_train_df.drop(
                "label", axis=1), sub_train_df["label"])
            node.add_node(f, sub_node)

        return node

    def fit(self, X, y):
        self.tree = self.train(X, y)
        return self.tree

    def predict(self, X_test):
        return self.tree.predict(X_test)
