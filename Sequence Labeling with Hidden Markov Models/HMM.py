import json
import nltk
import numpy as np
from tqdm import tqdm
from nltk.corpus import brown
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import copy
import os
from collections import Counter


def get_folds(tagged_sentences_all):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    return kf.split(tagged_sentences_all)


def get_unique_tags(tagged_sentences_all):
    tags = []
    for sentence in tagged_sentences_all:
        for _, tag in sentence:
            tags.append(tag)
    return list(set(tags))


def get_preprocessed_sentences(tagged_sentences_train):
    for i, sentence in enumerate(tagged_sentences_train):
        if sentence[0] != ("#####", "#####"):
            sentence.insert(0, ("#####", "#####"))
        if sentence[-1] != ("$$$$$", "$$$$$"):
            sentence.append(("$$$$$", "$$$$$"))
    return tagged_sentences_train


def make_tag_given_tag(tagged_sentences_train, tags, fold=None):
    tag_given_tag = {}
    for sentence in tagged_sentences_train:
        for i in range(len(sentence) - 1):
            pos1 = sentence[i][1]
            pos2 = sentence[i + 1][1]
            if pos2 not in tag_given_tag:
                tag_given_tag[pos2] = {}
                for tag in tags:
                    if tag != "$$$$$":
                        tag_given_tag[pos2][tag] = 0
            tag_given_tag[pos2][pos1] += 1

    for tag in tags:
        if tag != "$$$$$":
            total = sum(tag_given_tag[key1][tag] for key1 in tag_given_tag)
            for key1 in tag_given_tag:
                tag_given_tag[key1][tag] = (tag_given_tag[key1][tag] + 1) / (
                    total + len(tag_given_tag)
                )

    if fold is not None:
        with open(f"data/fold{fold}/tag_given_tag.json", "w") as f:
            json.dump(tag_given_tag, f)
    else:
        with open("data/full_data/tag_given_tag.json", "w") as f:
            json.dump(tag_given_tag, f)


def make_word_given_tag(tagged_sentences_train, tags, fold=None):
    word_given_tag = {}
    for sentence in tagged_sentences_train:
        for word, tag in sentence:
            word = word.lower()
            if word not in word_given_tag:
                word_given_tag[word] = {}
                for tag in tags:
                    word_given_tag[word][tag] = 0
            word_given_tag[word][tag] += 1

    total_given_tag = {}
    for tag in tags:
        total = sum(word_given_tag[word][tag] for word in word_given_tag)
        total_given_tag[tag] = total
        for word in word_given_tag:
            word_given_tag[word][tag] = (word_given_tag[word][tag] + 1) / (
                total + len(word_given_tag)
            )

    if fold is not None:
        with open(f"data/fold{fold}/word_given_tag.json", "w") as f:
            json.dump(word_given_tag, f)
        with open(f"data/fold{fold}/total_given_tag.json", "w") as f:
            json.dump(total_given_tag, f)
    else:
        with open("data/full_data/word_given_tag.json", "w") as f:
            json.dump(word_given_tag, f)
        with open("data/full_data/total_given_tag.json", "w") as f:
            json.dump(total_given_tag, f)


def make_tag_index_map(tags, fold=None):
    tags.remove("#####")
    tag_to_index = {}
    for tag in tags:
        if tag != "$$$$$":
            tag_to_index[tag] = len(tag_to_index)

    if fold is not None:
        with open(f"data/fold{fold}/tag_to_index.json", "w") as f:
            json.dump(tag_to_index, f)
    else:
        with open("data/full_data/tag_to_index.json", "w") as f:
            json.dump(tag_to_index, f)


def get_POS_tags(sentence, tag_given_tag, word_given_tag, total_given_tag, tag_to_index):
    primary_prob = []
    temp_prob = np.zeros((len(tag_to_index), len(tag_to_index)))

    for tag in tag_to_index:
        primary_prob.append(([tag], tag_given_tag[tag]["#####"]))

    for word in sentence[:-1]:
        for tag in tag_to_index:
            for tag2 in tag_to_index:
                try:
                    temp_prob[tag_to_index[tag], tag_to_index[tag2]] = (
                        tag_given_tag[tag2][tag]
                        * word_given_tag[word][tag]
                        * primary_prob[tag_to_index[tag]][1]
                    )
                except:
                    temp_prob[tag_to_index[tag], tag_to_index[tag2]] = (
                        tag_given_tag[tag2][tag]
                        * 1 / (total_given_tag[tag] + len(word_given_tag))
                        * primary_prob[tag_to_index[tag]][1]
                    )

        temp_list = []
        for tag in tag_to_index:
            max_prob = np.max(temp_prob[:, tag_to_index[tag]])
            max_index = np.argmax(temp_prob[:, tag_to_index[tag]])
            temp_list.append((primary_prob[max_index][0] + [tag], max_prob))
        primary_prob = temp_list

    for i, elem in enumerate(primary_prob):
        prob = elem[1]
        try:
            new_prob = (
                prob * tag_given_tag["$$$$$"][elem[0][-1]] * word_given_tag[sentence[-1]][elem[0][-1]]
            )
        except:
            new_prob = (
                prob * tag_given_tag["$$$$$"][elem[0][-1]] * 1 / (total_given_tag[elem[0][-1]] + len(word_given_tag))
            )
        primary_prob[i] = (elem[0], new_prob)

    max = -1
    for elem in primary_prob:
        if elem[1] > max:
            max = elem[1]
            max_elem = elem[0]

    return max_elem


def confmat(predicted, actual, tag_to_index):
    return metrics.confusion_matrix(actual, predicted, labels=list(tag_to_index.keys()))


def full_train(tagged_sentences_all, tags):
    tagged_sentences_all = copy.deepcopy(tagged_sentences_all)
    tags = copy.deepcopy(tags)
    tagged_sentences_all = get_preprocessed_sentences(tagged_sentences_all)
    make_tag_given_tag(tagged_sentences_all, tags)
    make_word_given_tag(tagged_sentences_all, tags)
    make_tag_index_map(tags)


def main():
    nltk.download("brown")
    nltk.download("universal_tagset")
    tagged_sentences_all = list(brown.tagged_sents(tagset="universal"))
    tags_org = get_unique_tags(tagged_sentences_all)
    tags_org.append("$$$$$")
    tags_org.append("#####")

    predicted_all = []
    actual_all = []

    if not os.path.exists(f"data/full_data"):
        os.makedirs(f"data/full_data")

    folds = get_folds(tagged_sentences_all)
    for fold, (train_index, test_index) in enumerate(folds, 1):
        print(f"Fold {fold}")
        tagged_sentences_train = [tagged_sentences_all[i] for i in train_index]
        tagged_sentences_test = [tagged_sentences_all[i] for i in test_index]

        tagged_sentences_train = copy.deepcopy(tagged_sentences_train)
        tagged_sentences_test = copy.deepcopy(tagged_sentences_test)
        tags = copy.deepcopy(tags_org)

        if not os.path.exists(f"data/fold{fold}"):
            os.makedirs(f"data/fold{fold}")

        tagged_sentences_train = get_preprocessed_sentences(tagged_sentences_train)
        make_tag_given_tag(tagged_sentences_train, tags, fold)
        make_word_given_tag(tagged_sentences_train, tags, fold)
        make_tag_index_map(tags, fold)

        predicted = []
        actual = []

        with open(f"data/fold{fold}/tag_given_tag.json", "r") as f:
            tag_given_tag = json.load(f)
        with open(f"data/fold{fold}/word_given_tag.json", "r") as f:
            word_given_tag = json.load(f)
        with open(f"data/fold{fold}/total_given_tag.json", "r") as f:
            total_given_tag = json.load(f)
        with open(f"data/fold{fold}/tag_to_index.json", "r") as f:
            tag_to_index = json.load(f)

        for sentence in tqdm(tagged_sentences_test):
            actual.extend([tag for _, tag in sentence])
            predicted.extend(
                get_POS_tags(
                    [word.lower() for word, _ in sentence],
                    tag_given_tag,
                    word_given_tag,
                    total_given_tag,
                    tag_to_index,
                )
            )

        predicted_all.extend(predicted)
        actual_all.extend(actual)

    # Calculate and print metrics
    accuracy = metrics.accuracy_score(actual_all, predicted_all)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    tag_counts = Counter(actual_all)
    most_common_tag = tag_counts.most_common(1)[0][0]
    baseline_predicted = [most_common_tag] * len(actual_all)
    baseline_accuracy = metrics.accuracy_score(actual_all, baseline_predicted)
    print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
    print(f"Accuracy Improvement: {accuracy - baseline_accuracy:.4f}")

    confusion_matrix = confmat(predicted_all, actual_all, tag_to_index)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        cmap="Reds",
        fmt="d",
        xticklabels=list(tag_to_index.keys()),
        yticklabels=list(tag_to_index.keys()),
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")

    print(
        metrics.classification_report(
            actual_all, predicted_all, labels=list(tag_to_index.keys()), digits=4
        )
    )

    for beta in [0.5, 1, 2]:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            actual_all,
            predicted_all,
            average="weighted",
            labels=list(tag_to_index.keys()),
            beta=beta,
        )
        print(f"Weighted F-beta Score (β={beta}): {fscore:.4f}")

    full_train(tagged_sentences_all, tags_org)


if __name__ == "__main__":
    main()
