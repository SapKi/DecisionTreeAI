import math
from functools import reduce
from collections import Counter
from DecisionTree import DecisionTreeClassifier

class KNNClassifier(object):
    def __init__(self, train_tags, train_examples, k=5):
        """
        gets list of tags for train_examples and
        list of examples. each example is a list of feature values
        and number of k-closest-examples to choose from in prediction
        """
        self.train_examples = train_examples
        self.train_tags = train_tags
        self.k = k

    def predict(self, example):
        """
        gets an example and returns a predicted tag for the example
        compute hamming distance between the example and all the train examples
        and extract just the tags
        """
        distances = []
        examples_and_tags = [(ex, tag) for ex, tag in
                             zip(self.train_examples, self.train_tags)]
        for example_and_tag in examples_and_tags:
            distance = self.hamming_distance(example_and_tag[0], example)
            distances.append((example_and_tag, distance))
        closest_k = sorted(distances, key=lambda x : x[1])[:self.k]

        closest_k = [item[0][1] for item in closest_k]
        res = self.get_common_tag(closest_k)
        return res


    def hamming_distance(self, first, second):
        """
        calculate difference between feature values between first and second
        """
        distance = 0
        for feature_1, feature_2 in zip(first, second):
            if feature_1 != feature_2:
                distance += 1
        return distance


    def get_common_tag(self, tags):
        """
        returns the most common tag in tags
        """
        tagscounter = Counter()
        for tag in tags:
            tagscounter[tag] += 1
        return tagscounter.most_common(1)[0][0]

class NaiveBayesClassifier(object):
    def __init__(self, train_tags, train_examples):
        self.train_examples = train_examples
        self.train_tags = train_tags

        feature_domain_size_dict = {}
        #a dict in which the key is a feature index and the value is the size of the feature domain
        for feature_index in range(len(self.train_examples[0])):
            domain = set([example[feature_index] for example in self.train_examples])
            feature_domain_size_dict[feature_index] = len(domain)
        self.feature_size_dict = feature_domain_size_dict

        #creates and a dict in which the key is a tag and the value is list of the examples that have this tag
        examples_tag_dict = {}
        for example, tag in zip(self.train_examples, self.train_tags):
            if tag in examples_tag_dict:
                examples_tag_dict[tag].append(example)
            else:
                examples_tag_dict[tag] = [example]
        self.examples_tag_dict = examples_tag_dict


    def predict(self, example):
        """
        calculates the probability for each class, keep tracking the maximum prob
        returns a predicted tag for the example
        """
        probs = []
        max_prob = 0
        #assuming we will get back the one most seen
        max_tag = list(self.examples_tag_dict.keys())[0]
        for tag in self.examples_tag_dict:
            prob = self.calculate_prob(self.examples_tag_dict[tag], example)
            probs.append(prob)
            if prob > max_prob:
                max_prob, max_tag = prob, tag

        if probs[0] == probs[1] and len(probs) == 2:
            pos = self.find_positive_tag(self.examples_tag_dict.keys())
            return pos

        # return the tag with the highest probability
        return max_tag

    def find_positive_tag(self, tags):
        for tag in tags:
            if tag in ["yes", "true"]:
                return tag

        return tag[0]

    def calculate_prob(self, tag_group, example):
        """
        calculates the probability that the example belongs to the class of the examples in tag_group
        computes conditioned probability for each feature
        """
        tag_group_size = len(tag_group)
        prob_list = []
        for feature_index in range(len(example)):
            counter = 1
            domain_size = self.feature_size_dict[feature_index]
            for train_example in tag_group:
                if train_example[feature_index] == example[feature_index]:
                    counter += 1
            prob_list.append(float(counter)/ (tag_group_size + domain_size))
        class_prob = float(len(tag_group)) / len(self.train_examples)

        a = lambda x, y: x * y
        return reduce(a , prob_list) * class_prob


def write_output_files(test_tags, preds_per_classifier, accuracy_per_classifier, dt_classifier):
    """
    print results on the test set to output.txt and output_tree.txt file
    preds_per_classifier is a list of predictions of each classifier
    accuracy_per_classifier is a list of accuracy of each classifier
    dt_classifier is DecisionTreeClassifier object for print
    """
    dt_preds, knn_preds, nb_preds = preds_per_classifier[0], preds_per_classifier[1], preds_per_classifier[2]
    dt_acc, knn_acc, nb_acc = accuracy_per_classifier[0], accuracy_per_classifier[1], accuracy_per_classifier[2]
    with open("output.txt", "w") as output:
        lines = []
        lines.append("Num\tDT\tKNN\tnaiveBayes")
        i = 1
        for true_tag, dt_pred, knn_pred, nb_pred in zip(test_tags, dt_preds, knn_preds, nb_preds):
            lines.append("{}\t{}\t{}\t{}".format(i, dt_pred, knn_pred, nb_pred))
            i += 1
        lines.append("\t{}\t{}\t{}".format(dt_acc, knn_acc, nb_acc))
        output.writelines("\n".join(lines))

    dt_classifier.write_tree_to_file("output_tree.txt")

def read_file(file_name):
    features = []
    tags = []
    examples = []
    with open(file_name, "r") as file:
        content = file.readlines()
        features += content[0].strip("\n").strip().split("\t")
        for line in content[1:]:
            line = line.strip("\n").strip().split("\t")
            example , tag = line[:len(line) - 1], line[-1]
            examples.append(example)
            tags.append(tag)
    return features, tags, examples

def get_accuracy(predicted_tags, true_tags):
    goodtag = 0.00
    badtag = 0.00
    for true_tag, pred in zip(true_tags, predicted_tags):
        if true_tag == pred:
            goodtag += 1
        else:
            badtag += 1
        accuracy = float(goodtag)/(goodtag + badtag)
    #requested to upper above
    return math.ceil(accuracy * 100) / 100

def find_positive_tag(tags):
    for tag in tags:
        if tag in ["yes", "true"]:
            return tag
    #returning the default
    return tag[0]

def main():
    features, train_tags, train_examples = read_file("train.txt")
    _,test_tags,test_examples  = read_file("test.txt")

    dt_classifier = DecisionTreeClassifier(train_tags, train_examples, features[:len(features) - 1])
    knn_classifier = KNNClassifier(train_tags, train_examples, k=5)
    nb_classifier = NaiveBayesClassifier(train_tags, train_examples)


    classifiers = [dt_classifier, knn_classifier, nb_classifier]
    preds_of_classifier = []
    accuracy_per_classifier = []

    for classifier in classifiers:
        preds = []
        for example, tag in zip(test_examples, test_tags):
            pred = classifier.predict(example)
            preds.append(pred)
        preds_of_classifier.append(preds)
        accuracy_per_classifier.append(get_accuracy(preds, test_tags))

    write_output_files(test_tags, preds_of_classifier, accuracy_per_classifier, dt_classifier)

if __name__ == "__main__":
    main()