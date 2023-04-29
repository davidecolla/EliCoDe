from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import argparse
from pathlib import Path
import numpy as np
import os

def main(language_base_path, output_folder):

    language_base_path = str(language_base_path) + "/"
    output_folder = str(output_folder) + "/"
    
    # BUild Output Folder
    os.makedirs(output_folder, exist_ok=True)

    htrain = open(language_base_path + 'train.tsv', 'r').readlines()
    hdev = open(language_base_path + 'dev.tsv', 'r').readlines()
    htest = open(language_base_path + 'test.tsv', 'r').readlines()

    train_sent = []
    dev_sent = []
    train_labels = []
    dev_labels = []
    test_sent = []

    # Load train set
    for line in htrain:
        if line != '\n':
            if line not in train_sent:
                train_sent.append(line.strip().split()[0])
                train_labels.append(line.strip().split()[1])

    # Load dev set
    for line in hdev:
        if line != '\n':
            if line not in dev_sent:
                dev_sent.append(line.strip().split()[0])
                dev_labels.append(line.strip().split()[1])

    # Load test set
    for line in htest:
        if line != '\n':
        #     if line not in test_sent:
            test_sent.append(line.strip().split()[0])
        else:
            test_sent.append("\n")

    # Init Tokenizer as count vectorizer
    vectorizer = CountVectorizer()

    # Fit the tokenizer on train sentences
    vectorizer.fit(train_sent)

    # Tokenize sentences
    X_train = vectorizer.transform(train_sent)
    X_dev = vectorizer.transform(dev_sent)
    X_test = vectorizer.transform(test_sent)

    # Tokenize lables
    y_train = np.array(train_labels)
    y_dev = np.array(dev_labels)

    # Build Multinomial Naive Bayes classifier
    nb = MultinomialNB()

    # Fit the classifier on train texts
    nb.fit(X_train, y_train)

    # Run prediction on test set
    y_pred = nb.predict(X_dev)

    # Compute the accuracy on dev set
    accuracy = accuracy_score(y_dev, y_pred)

    # Print accuracy
    print("Accuracy on dev set:", accuracy)

    # Run prediction on test set
    y_test_pred = nb.predict(X_test).tolist()
    test_tuples = zip(test_sent, y_test_pred)

    open(output_folder + "baseline_preds.txt", "w").write("\n".join([(x[0] + "\t" + x[1]) if x[0] != "\n" else "" for x in test_tuples]) + "\n")

    y_dev_pred = nb.predict(X_dev).tolist()
    dev_report = classification_report(y_dev, y_dev_pred)
    open(output_folder + 'baseline_dev_report.txt', 'w').write(dev_report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language_folder", type=Path)
    parser.add_argument("--output_folder", type=Path)
    p = parser.parse_args()
    
    main(p.language_folder, p.output_folder)