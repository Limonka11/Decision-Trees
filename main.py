from tree import *
from numpy.random import default_rng

print("Clean set single tree results:")
root, validation_dataset, test_dataset = decision_tree_learning("wifi_db/clean_dataset.txt")

graph = plt
graph.title("Clean dataset no pruning")
print_tree(root, graph, 0, 0, 1)
graph.gcf().set_size_inches(20, 9)
graph.savefig("clean_set_no_pruning.png", dpi=500)
graph.clf()
print("Depth without pruning: ", depth(root))

matrix = confusion_matrix(test_dataset, root, 4)
print("Accuracy without pruning: ", accuracy(matrix))

pruning(root,root, validation_dataset, 4)

graph = plt
graph.title("Clean dataset with pruning")
print_tree(root, graph, 0, 0, 1)
graph.gcf().set_size_inches(20, 9)
graph.savefig("clean_set_with_pruning.png", dpi=500)
graph.clf()
print("Depth with pruning: ", depth(root))

matrix = confusion_matrix(test_dataset, root, 4)
print("Accuracy with pruning: ", accuracy(matrix))
print()

print("Noisy set single tree results:")
root, validation_dataset, test_dataset = decision_tree_learning("wifi_db/noisy_dataset.txt")

graph = plt
graph.title("Noisy dataset no pruning")
print_tree(root, graph, 0, 0, 1)
graph.gcf().set_size_inches(20, 9)
graph.savefig("noisy_set_no_pruning.png", dpi=500)
graph.clf()
print("Depth without pruning: ", depth(root))

matrix = confusion_matrix(test_dataset, root, 4)
print("Accuracy without pruning: ", accuracy(matrix))

pruning(root,root, validation_dataset, 4)

graph = plt
graph.title("Noisy dataset with pruning")
print_tree(root, graph, 0, 0, 1)
graph.gcf().set_size_inches(20, 9)
graph.savefig("noisy_set_with_pruning.png", dpi=500)
graph.clf()
print("Depth with pruning: ", depth(root))

matrix = confusion_matrix(test_dataset, root, 4)
print("Accuracy with pruning: ", accuracy(matrix))
print()
print("10-fold cross validation (option 2): ")
print("-----------------------------------------------")
print("Results for clean set:")
# Clean set
(entries, labels, rooms) = read_dataset("wifi_db/clean_dataset.txt")
num_features = entries.shape[1]
dataset = np.c_[entries, labels]

matrix_without_pruning, matrix_with_pruning, depth_no_pruning, depth_with_pruning = \
    evaluate(dataset, labels, num_features, rooms, 10)

print("***********************************************")
print("No pruning:")
print_results(matrix_without_pruning)
print("Average tree depth: ", depth_no_pruning)

print("***********************************************")
print("With pruning:")
print_results(matrix_with_pruning)
print("Average tree depth: ", depth_with_pruning)

# Noisy set
print("-----------------------------------------------")
print("Results for noisy set:")
(entries, labels, rooms) = read_dataset("wifi_db/noisy_dataset.txt")
num_features = entries.shape[1]
dataset = np.c_[entries, labels]

matrix_without_pruning, matrix_with_pruning, depth_no_pruning, depth_with_pruning = \
    evaluate(dataset, labels, num_features, rooms, 10)

print("***********************************************")
print("No pruning:")
print_results(matrix_without_pruning)
print("Average tree depth: ", depth_no_pruning)

print("***********************************************")
print("With pruning:")
print_results(matrix_with_pruning)
print("Average tree depth: ", depth_with_pruning)
