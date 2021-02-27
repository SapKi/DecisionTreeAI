from collections import Counter
import math

class DecisionTreeClassifier(object):
    """
    implements DTL algorithm
    """
    def __init__(self, train_tags, train_examples, features):

        self.train_examples = train_examples
        self.train_tags = train_tags
        self.features = features

        feature_domain_dictt = {}
        for feature_index in range(len(self.train_examples[0])):
            domain = set([example[feature_index] for example in self.train_examples])
            feature_domain_dictt[self.features[feature_index]] = domain

        self.feature_domain_dict = feature_domain_dictt

        tagged_examples = [(example, tag) for example, tag in zip(self.train_examples, self.train_tags)]
        self.decisionTree = DecisionTree(self.DTL(tagged_examples, features, 0,self.get_default_tag(train_tags)), features)


    def DTL(self, examplestags, features, depth, default=None):
        """
        gets list of (example, tag) tuples ,unselected features list,
        depth of the current node, default pred returns root
        """
        # check first if examples is an empty list
        if not examplestags:
            return DecisionTreeNode(depth, None, is_leaf=True, pred=default)

        # if all examples have the same tag
        examples = [example for example, tag in examplestags]
        tags = [tag for example, tag in examplestags]

        if len(set(tags)) == 1:
            return DecisionTreeNode(depth, None, is_leaf=True, pred=tags[0])

        # if features list is empty
        if not features:
            return DecisionTreeNode(depth ,None, is_leaf=True,  pred=self.get_default_tag(tags))


        node = self.best_feature(features,examples,tags,depth)

        return node

    def best_feature(self,features,examples,tags,depth):
        # for the current node find the best feature
        best_feature = self.choose_feature(features, examples, tags)
        node = DecisionTreeNode(depth, best_feature)
        feature_index = self.get_feature_index(best_feature)

        child_features = features[:]
        child_features.remove(best_feature)
        for possible_value in self.feature_domain_dict[best_feature]:
            examples_and_tags_vi = [(example, tag) for example, tag in zip(examples, tags)
                                    if example[feature_index] == possible_value]
            # create child subtree for every possibe value of the feature
            child = self.DTL(examples_and_tags_vi, child_features, depth + 1, self.get_default_tag(tags))
            node.children[possible_value] = child

        return node

    def choose_feature(self, features, examples, tags):
        """
        returns best feature by the highest gain will be chosen.
        """
        max_gain = 0
        max_feature = features[0]
        features_gains_dict = {feature : self.get_gain(examples, tags, feature) for feature in features}
        for feature in features:
            if features_gains_dict[feature] > max_gain:
                max_gain = features_gains_dict[feature]
                max_feature = feature

        # return the feature with the highest gain
        return max_feature

    def calc_entropy(self, tags):
        """
        finds the entropy of the given tags
        """
        tags_counter = Counter()

        if not tags:
            return 0

        for tag in tags:
            tags_counter[tag] += 1
        classes_probs = [tags_counter[tag] / float(len(tags)) for tag in tags_counter]
        if 0.0 in classes_probs:
            return 0

        entropy = 0
        for prob in classes_probs:
            entropy -= prob * math.log(prob, 2)

        return entropy

    def get_gain(self, examples, tags, feature):
        """
        finds the gain for the given feature according to the given examples and tags.
        """
        initial_entropy = self.calc_entropy(tags)
        relative_entropy_per_feature = []
        feature_index = self.get_feature_index(feature)
        for possible_value in self.feature_domain_dict[feature]:
            examples_and_tags_vi = [(example, tag) for example, tag in zip(examples, tags)
                                            if example[feature_index] == possible_value]
            tags_vi = [tag for example, tag in examples_and_tags_vi]
            entropy_vi = self.calc_entropy(tags_vi)
            if not examples:
                pass
            relative_entropy = (float(len(examples_and_tags_vi)) / len(examples)) * entropy_vi
            relative_entropy_per_feature.append(relative_entropy)

        return initial_entropy - sum(relative_entropy_per_feature)



    def get_feature_index(self, feature):
        return self.features.index(feature)


    def get_default_tag(self, tags):
        """
        returns the most common tag in tags list.
        """
        tags_counter = Counter()
        for tag in tags:
            tags_counter[tag] += 1

        if len(tags_counter) == 2 and list(tags_counter.values())[0] == list(tags_counter.values())[1]:
            return self.find_positive_tag(tags_counter.keys())

        return tags_counter.most_common(1)[0][0]

    def find_positive_tag(tags):
        for tag in tags:
            if tag in ["yes", "true"]:
                return tag

        return tag[0]
    def predict(self, example):
        """
        gets an example and returns a predicted tag for the example
        """
        return self.decisionTree.traverse_tree(example)

    def write_tree_to_file(self, output_file_name):
        """
        writes the tree structure to output_file_name
        """
        with open(output_file_name, "w") as output:
            tree_string = self.decisionTree.tree_string(self.decisionTree.root)
            output.write(tree_string[:len(tree_string) - 1])


class DecisionTree(object):
    """
    root of the tree and traverse tree functionality.
    """
    def __init__(self, root, features):
        self.features = features
        self.root = root

    def traverse_tree(self, example):
        """
        prediction for the example after traversing the tree and reaching to a leaf
        """
        current_node = self.root
        while not current_node.is_leaf:
            feature_value = example[self.get_feature_index(current_node.feature)]
            current_node = current_node.children[feature_value]

        return current_node.pred

    def get_feature_index(self, feature):
        return self.features.index(feature)

    def tree_string(self, node):
        """
        return the string of the tree structure
        """
        string = ""
        for child in sorted(node.children):
            string += node.depth * "\t"
            if node.depth > 0:
                string += "|"
            string += node.feature + "=" + child
            if node.children[child].is_leaf:
                string += ":" + node.children[child].pred + "\n"
            else:
                string += "\n" + self.tree_string(node.children[child])

        return string


class DecisionTreeNode(object):
    """
    non-leaf node contains the feature, depth and children dict that maps every possible value
    of the feature to a child node and each leaf node contains pred attribute.
    """
    def __init__(self,  depth, feature, is_leaf=False, pred=None):
        self.depth = depth
        self.feature = feature
        self.is_leaf = is_leaf
        self.pred = pred
        self.children = {}