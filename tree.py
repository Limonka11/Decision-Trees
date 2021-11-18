from utils import *

class Node():
    """Class representing a Node in the decision tree matrix

    Parameters
        feature_index [int]  : the index of the feature the node decides to make a decision
        threshold     [float]: the value of the feature the node decides to make a decision
        left          [Node] : left child node
        right         [Node] : right child node
        info_gain     [float]: the gain gained by the devision
        label         [int]  : the label of the node if it is a leaf
        depth         [int]  : the depth of the node
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, label=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.label = label

        # for any Node
        self.parent = None


def decision_tree_learning(filepath):
    """Starts the decision tree learning and returns a tree in the form of a Node"""
    (entries, labels, rooms) = read_dataset(filepath)
    # Number of features
    num_features = entries.shape[1]
    # Concatenation of entreis and labels to make a full dataset in numpy
    dataset = np.c_[entries, labels]
    # Convert rooms type to from np.float to np.int
    rooms = np.array(rooms, dtype=np.int)

    # Generate training, validation and test datasets in ratio 80:10:10
    num_instances = entries.shape[0]
    random_generator = default_rng()
    shuffled_indices = random_generator.permutation(num_instances)
    training_dataset = \
        dataset[[shuffled_indices[i] for i in range(0, int(num_instances * 8 / 10))]]
    validation_dataset = \
        dataset[[shuffled_indices[i] for i in range(int(num_instances * 8 / 10), int(num_instances * 9 / 10))]]
    test_dataset = \
        dataset[[shuffled_indices[i] for i in range(int(num_instances * 9 / 10), int(num_instances))]]
    return build_decision_tree(training_dataset, num_features, rooms), validation_dataset, test_dataset


def build_decision_tree(training_dataset, num_features, label_set, depth = 0):
    """Return a Node which is either a leaf node or has 2 children:
       left and right"""
    # If all the labels are the same
    # this Node should be a leaf
    if np.all(training_dataset[:,-1] == training_dataset[0, -1]):
        leaf = Node()
        leaf.label = training_dataset[0, -1]
        leaf.depth = 1
        return leaf
    else:
        # Find the best split
        max_info_gain, best_split = find_split(training_dataset, num_features, label_set)
        # Then assign the info to an instance of Node
        root = Node()
        root.feature_index = best_split["best_feature_index"]
        root.threshold = best_split["best_threshold"]
        root.info_gain = max_info_gain
        root.left = build_decision_tree(best_split["best_left"], num_features, label_set, depth + 1)
        root.left.parent = root
        root.right = build_decision_tree(best_split["best_right"], num_features, label_set, depth + 1)
        root.right.parent = root
        root.depth = max(root.left.depth + 1, root.right.depth + 1)
        # Return the resulted Node
        return root


def predict(root, features):
    """Go throuh the Node "root" and return the label the set of
       features the decision tree predicts"""
    # If the node is a leaf then return its label
    # as a label has been decided
    if root.left is None:
        return root.label
    else:
        # Decide whether to traverse to the left or the right node
        feature_index = root.feature_index
        threshold = root.threshold
        if features[feature_index] <= threshold:
            return predict(root.left, features)
        else:
            return predict(root.right, features)

def accuracy(confusion_matrix):
    """Returns the accuracy measure given a confusion matrix"""

    return np.trace(confusion_matrix) / np.sum(confusion_matrix)

def confusion_matrix(test_set, root, num_labels):
    """Builds confusion matrix given a test set, the tree and
       the number of possible labels"""
    matrix = np.zeros((num_labels, num_labels))
    for row in test_set:
        features = row[:-1]
        label = int(row[-1])
        matrix[label - 1, int(predict(root, features)) - 1] += 1

    return matrix

def precision(confusion_matrix, label):
    """Returns the precision measure for a label given a confusion matrix"""
    return confusion_matrix[label - 1][label - 1] / np.sum(confusion_matrix[:, label - 1])

def recall(confusion_matrix, label):
    """Returns the recall measure for a label given a confusion matrix"""
    return confusion_matrix[label - 1][label -1 ] / np.sum(confusion_matrix[label - 1])

def F1(precision, recall):
    """Returns F1 given the precision and recall"""
    return 2 * precision * recall / (precision + recall)

def print_tree(root, graph, current_y, current_x, depth):
    """Produces a plot visualising the tree. Recursive on each node, hence
       at each iteration given current coordinates and depth"""
    # Offsets for current iteration
    x_branch_offset = 1000/(1.5**depth)
    y_branch_offset = 100

    if root.threshold is None:
        # Leaf node - add final decision to graph
        graph.text(current_x, current_y, str(int(root.label)), ha="center", va="top",
         bbox=dict(facecolor='red',boxstyle='square',edgecolor='black',pad=0.2))

        return

    # Plot current attribute and threshold on graph
    threshold_annotation = str(root.feature_index) + ": <=" + str(root.threshold)
    font_size = 30 / min(depth + 2, 7)
    ann = graph.text(current_x, current_y, threshold_annotation, ha="center",
        va="bottom",
        bbox=dict(facecolor='white',boxstyle='square',edgecolor='black',pad=0.1),
        fontsize = font_size)

    # Random color for current lines
    col = (np.random.random(), np.random.random(), np.random.random())

    if root.left is not None:
        # Draw line to and plot left path recursively
        graph.plot([current_x, current_x-x_branch_offset], [current_y, current_y -  y_branch_offset], color = col)
        print_tree(root.left, graph, current_y-y_branch_offset, current_x-x_branch_offset, depth+1)

    if root.right is not None:
        # Draw line to and plot right path recursively
        graph.plot([current_x, current_x+x_branch_offset], [current_y, current_y -  y_branch_offset], color = col)
        print_tree(root.right, graph, current_y-y_branch_offset, current_x+x_branch_offset, depth+1)

def pruning(root, current_node, validation_dataset, num_labels):
    """Prunes the tree from the current node down given a validation set"""
    parent = current_node.parent
    # If the node has two leaves as children
    if is_leaf(current_node.left) and is_leaf(current_node.right) and parent != None:
        is_pruned = False
        # Find if current node is a left child
        is_left = current_node == current_node.parent.left

        # Determine accuracies if the current node is swapped with either
        # its left or right child
        accuracy_change_left = 0
        accuracy_change_right = 0
        accuracy_no_change = accuracy(confusion_matrix(validation_dataset, root, num_labels))

        if is_left:
            parent.left = current_node.left
            accuracy_change_left = accuracy(confusion_matrix(validation_dataset, root, num_labels))
            parent.left = current_node.right
            accuracy_change_right = accuracy(confusion_matrix(validation_dataset, root, num_labels))
            parent.left = current_node
        else:
            parent.right = current_node.left
            accuracy_change_left = accuracy(confusion_matrix(validation_dataset, root, num_labels))
            parent.right = current_node.right
            accuracy_change_right = accuracy(confusion_matrix(validation_dataset, root, num_labels))
            parent.right = current_node

        # Find the best accuracy and decide if node should be pruned
        max_accuracy = max(accuracy_no_change, accuracy_change_left, accuracy_change_right)
        if max_accuracy == accuracy_change_left:
            if is_left:
                parent.left = current_node.left
            else:
                parent.right = current_node.left
            is_pruned = True
        elif max_accuracy == accuracy_change_right:
            if is_left:
                parent.left = current_node.right
            else:
                parent.right = current_node.right
            is_pruned = True

        # If node has been pruned, check if its parent can now be pruned
        if is_pruned:
            pruning(root, current_node.parent, validation_dataset, num_labels)

    # Recursive call down the tree
    if not is_leaf(current_node.right):
        pruning(root, current_node.right, validation_dataset, num_labels)
    if not is_leaf(current_node.left):
        pruning(root, current_node.left, validation_dataset, num_labels)


def is_leaf(node):
    """Determines if a node is a leaf"""
    return node.label != None

def depth(node):
    """Finds the depth of a tree"""
    if is_leaf(node):
        return 1
    else:
        return max(depth(node.left) + 1, depth(node.right) + 1)


def evaluate(dataset, labels, num_features, label_set, num_folds):
    """Performs k-fold evaluation on the model before and after pruning"""
    accuracies = []

    num_labels = label_set.size
    matrix_no_pruning = np.zeros((num_labels, num_labels))
    matrix_with_pruning = np.zeros((num_labels, num_labels))
    depth_no_pruning = 0
    depth_with_pruning = 0


    for i, (test_indices, validation_indices, train_indices) in enumerate(train_test_k_fold(num_folds, len(dataset))):
        dataset_train = dataset[train_indices, :]
        dataset_test = dataset[test_indices, :]
        dataset_validation = dataset[validation_indices, :]

        # Build tree using training set
        root = build_decision_tree(dataset_train, num_features, label_set)

        # Add results from non-pruned tree
        matrix = confusion_matrix(dataset_test, root, num_labels)
        matrix_no_pruning += matrix
        depth_no_pruning += depth(root)

        # Prune tree using the validation set
        pruning(root, root, dataset_validation, num_labels)

        # Add results from pruned tree
        matrix = confusion_matrix(dataset_test, root, num_labels)
        matrix_with_pruning += matrix
        depth_with_pruning += depth(root)

    # Average results
    num_instances = num_folds * (num_folds - 1)
    matrix_no_pruning /= num_instances
    matrix_with_pruning /= num_instances
    depth_no_pruning /= num_instances
    depth_with_pruning /= num_instances

    # Return average results for non-pruned tree and for pruned tree
    return matrix_no_pruning, matrix_with_pruning, depth_no_pruning, depth_with_pruning

def print_results(confusion_matrix):
    """Prints all evaluation metrics given a confusion matrix"""

    size = confusion_matrix[0].size

    recalls = [recall(confusion_matrix,i + 1) for i in range(size)]
    precisions = [precision(confusion_matrix,i + 1) for i in range(size)]
    F1s = [F1(precisions[i], recalls[i]) for i in range(size)]

    print(confusion_matrix)
    for i in range(size):
        print("Recall class ", i + 1, ": ", recalls[i])
    print()
    for i in range(size):
        print("Precision class ", i + 1, ": ", precisions[i])
    print()
    for i in range(size):
        print("F1 class ", i + 1, ": ", F1s[i])
    print()
    print("Accuracy: ", accuracy(confusion_matrix))
