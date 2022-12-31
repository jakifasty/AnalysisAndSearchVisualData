# %%
import numpy as np
import cv2 as cv

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.4)

# SIFT parameters
cT = 0.1   # Chosen by examining distributions of resulting number of keypoints
eT = 0.2    # Chosen by examining distributions of resulting number of keypoints
detector = cv.xfeatures2d.SIFT_create(contrastThreshold=cT, edgeThreshold=eT)


###### 2.2 Compute features ######
# The features at idx-1 correspond to the features of object with index idx
print("Extracting features from images...")
server_descriptors = []
client_descriptors = []
for obj_idx in tqdm(range(1, 51)):
    # Compute and store database features
    features = np.array([])
    for version_idx in range(1, 6):
        databaseImg = cv.imread(f"data2/databaseImages/obj{obj_idx}_{version_idx}.JPG") #this reads the server images
        if img is None:
            continue
        kp, des = detector.detectAndCompute(img, None)
        features = des if features.size == 0 else np.vstack((features, des))
    server_descriptors.append(features)

    # Compute and store client features
    img = cv.imread(f"data2/queryImages/obj{obj_idx}_t1.JPG") #this reads the client images
    kp, des = detector.detectAndCompute(img, None)
    client_descriptors.append(des)

# %%

###### 2.3 Build tree ######
class Node:
    def __init__(self, depth, children, feature_vector, is_leaf, leaf_idx=None, idf_values=None, n_samples=None, obj_indices=None):
        """
        Args:
            depth (int): The depth of this node
            children (list): List of children nodes
            feature_vector (np.array): Feature vector of node (descriptor)
            is_leaf (bool): True if node is leaf
            leaf_idx (int): Global index of leaf node
            idf_values (np.array): idf value of visual word (leaf) for each object (array of size n_objects)
            n_samples (int): Number of training samples in this node cluster (only if is_leaf is true)
            obj_indices (list): List of object indices of the samples that fall into this leaf cluster
                                during training (only if is_leaf is true)
        """
        self.depth = depth
        self.children = children
        self.feature_vector = feature_vector
        self.is_leaf = is_leaf
        self.leaf_idx = leaf_idx
        self.idf_values = idf_values
        self.n_samples = n_samples
        self.obj_indices = obj_indices

class VocabularyTree:
    def __init__(self):
        self.root = None
        self.max_depth = None
        self.B = None
        self.idx2leaf = {} # Maps leaf indices to leaf nodes

        # Stats
        self.n_leaves = 0
        self.nodes_per_level = None
        self.branches_per_level = None
        self.n_samples_per_leaf = None

        # K-means hyperparameters
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1) # type, max_iter, epsilon
        self.attempts = 10
        self.flags = cv.KMEANS_RANDOM_CENTERS


    def fit(self, descriptors, max_depth, B):
        """
        Builds a hierarchical vocabulary tree with the given parameters.
        Args:
            descriptors (list of np.array): Descriptors per object / document
            max_depth (int): Maximum depth of tree
            B (int): Branching factor (K value in K-means)
        """
        self.max_depth = max_depth
        self.B = B

        ### Build the tree with all descriptors
        #   Note that descriptors is a list of np arrays (one per object)
        # Flatten the list of descriptors into one big array
        # Compute the object index for every descriptor in an array of the same size
        print("HERE the descriptors:")
        print(descriptors)
        print(np.array(descriptors).shape)

        descriptors_all = np.vstack(descriptors)
        obj_indices = [[i] * descriptors[i].shape[0] for i in range(len(descriptors))]
        obj_indices = np.hstack(obj_indices)
        _, labels, centers = cv.kmeans(descriptors_all, B, None, self.criteria, self.attempts, self.flags)
        labels = labels.reshape(-1)

        children = []
        for i in range(B):
            descr_idx = np.where(labels == i)[0]
            descr_cur_cluster = descriptors_all[descr_idx]
            obj_cur_cluster = obj_indices[descr_idx]
            
            # Build branch for this cluster
            child_node = self.build_branch(centers[i], descr_cur_cluster, max_depth, 1, B, obj_cur_cluster)
            children.append(child_node)

        self.root = Node(depth=0, children=children, feature_vector=None, is_leaf=False)
        
        # Compute the idf value for each leaf node
        self.compute_idf(descriptors) # TODO: test if makes sense?
        
        return self

    def build_branch(self, centroid, descriptors, max_depth, cur_depth, B, obj_indices):
        children = []

        # Case: Not enough descriptors to cluster or already at max depth
        # Return leaf node
        if descriptors.shape[0] < B or cur_depth == max_depth:
            leaf_idx = self.n_leaves
            self.n_leaves += 1
            leaf = Node(depth=cur_depth,
                        children=[],
                        feature_vector=centroid,
                        is_leaf=True,
                        leaf_idx=leaf_idx,
                        n_samples=descriptors.shape[0],
                        obj_indices=obj_indices)
            self.idx2leaf[leaf_idx] = leaf
            return leaf

        # Case: Create another branch
        # Perform k-means on descriptors
        _, labels, centers = cv.kmeans(descriptors, B, None, self.criteria, self.attempts, self.flags)
        labels = labels.reshape(-1)

        for i in range(B):
            descr_idx = np.where(labels == i)[0]
            descr_cur_cluster = descriptors[descr_idx]
            obj_cur_cluster = obj_indices[descr_idx]
            
            # Build branch for this cluster
            child_node = self.build_branch(centers[i], descr_cur_cluster, max_depth, cur_depth + 1, B, obj_cur_cluster)
            children.append(child_node)

        return Node(depth=cur_depth, children=children, feature_vector=centroid, is_leaf=False)

    def get_closest_leaf(self, node, query_feature_vector):
        """
        Traverses the tree from the given node to find the index of the closest leaf node.
        """
        if node.is_leaf:
            return node.leaf_idx

        # Find closest child
        min_dist = np.inf
        best_child = None
        for child in node.children:
            dist = np.linalg.norm(query_feature_vector - child.feature_vector)
            if dist < min_dist:
                min_dist = dist
                best_child = child
        return self.get_closest_leaf(best_child, query_feature_vector)


    def compute_idf(self, descriptors_all):
        n_objects = len(descriptors_all)
        
        # Compute Number of documents every word appears in
        for idx, leaf in self.idx2leaf.items():
            n_obj_of_word = len(set(leaf.obj_indices))
            leaf.idf = np.log(n_objects / n_obj_of_word)


    def compute_stats(self):
        """
        Stats that are computed:
            - Number of nodes per level
            - Number of branches on each level
        """
        self.nodes_per_level = [0 for _ in range(self.max_depth + 1)]
        self.branches_per_level = [[] for _ in range(self.max_depth + 1)]
        self.n_samples_per_leaf = []

        def traverse(node):
            self.nodes_per_level[node.depth] += 1
            self.branches_per_level[node.depth].append(len(node.children))
            if not node.is_leaf:
                for child in node.children:
                    traverse(child)
            if node.is_leaf:
                self.n_samples_per_leaf.append(node.n_samples)

        traverse(self.root)

    def get_tfidf_vector(self, descriptors):
        """
        Given a set of keypoints / descriptors that represent an image,
        compute an array of vocabulary size with tf-idf values. 
        """
        # For every descriptor, find the closest leaf node
        # and add the tf-idf value to the corresponding entry in the vector
        n_words = len(descriptors)

        # Count how often every word appears in the image
        word_indices = []
        for descriptor in descriptors:
            leaf_idx = self.get_closest_leaf(self.root, descriptor)
            word_indices.append(leaf_idx)
        word_indices = np.array(word_indices)

        # Compute tf value for every word
        tf_vector = np.zeros(self.n_leaves)
        for word_idx in set(word_indices):
            tf_vector[word_idx] = np.sum(word_indices == word_idx) / n_words

        # Compute tf-idf vector for whole image
        tfidf_vector = np.zeros(self.n_leaves)
        for leaf_idx in self.idx2leaf.keys():
            tfidf_vector[leaf_idx] = tf_vector[leaf_idx] * self.idx2leaf[leaf_idx].idf

        # Normalize with l1 norm
        tfidf_vector = tfidf_vector / np.linalg.norm(tfidf_vector, ord=1)
        return tfidf_vector
        

def hi_kmeans(data, b, depth):
    tree = VocabularyTree()
    tree.fit(data, depth, b)
    return tree



# Tree hyperparameters
B = 5
max_depth = 6

print("Building tree...")
tree = hi_kmeans(server_descriptors[:25_000], max_depth, B) # TODO: Run on full dataset

# %%

##### Compute vectors for all objects #####
print("Computing vectors for database objects...")
db_vectors = []
for descriptors in tqdm(server_descriptors):
    db_vectors.append(tree.get_tfidf_vector(descriptors))

print("Computing vectors for query objects...")
query_vectors = []
for descriptors in tqdm(client_descriptors):
    query_vectors.append(tree.get_tfidf_vector(descriptors))

# %%

##### Find closest matches and compute recall #####
is_matched_top1 = np.zeros(len(query_vectors))
is_matched_top5 = np.zeros(len(query_vectors))
for obj_q, qv in enumerate(query_vectors):
    # Find 5 closest objects
    dists = np.array([np.linalg.norm(qv - dbv, ord=1) for dbv in db_vectors])
    closest_idxs = np.argsort(dists)[:5]

    # Check if is in top 1 or top 5
    # Note that indices of query and database vectors can be seen as object names
    if obj_q == closest_idxs[0]:
        is_matched_top1[obj_q] = 1
    if obj_q in closest_idxs:
        is_matched_top5[obj_q] = 1

recall_top1 = np.sum(is_matched_top1) / len(query_vectors)
recall_top5 = np.sum(is_matched_top5) / len(query_vectors)
print(f"Recall top 1: {recall_top1}")
print(f"Recall top 5: {recall_top5}")
    
# %%

##### Compute stats #####
tree.compute_stats()
print(f"mean: {np.mean(tree.n_samples_per_leaf)}")
print(f"10% quantile: {np.quantile(tree.n_samples_per_leaf, 0.1)}")
print(f"90% quantile: {np.quantile(tree.n_samples_per_leaf, 0.9)}")

plt.figure(figsize=(10, 5))
sns.distplot(tree.n_samples_per_leaf, kde=False, bins=50)

