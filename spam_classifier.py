"""
Spam/Ham Message Classifier — Naive Bayes, built from scratch (no sklearn).

Trains on the UCI SMS Spam Collection dataset. Classifies a message as
spam or ham using Multinomial Naive Bayes with Laplace (add-1) smoothing,
implemented directly from Bayes' theorem and word frequency counts.
"""

import os
import re
import math
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "SMSSpamCollection")


# ---------- Data loading ----------

def load_dataset(filepath):
    """Load tab-separated (label, message) pairs."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at: {filepath}\n"
            f"Download it from https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/SMSSpamCollection "
            f"and place it in data/SMSSpamCollection"
        )
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, message = line.split('\t', 1)
            data.append((label, message))
    return data


def tokenize(text):
    """Lowercase, strip punctuation, split into words."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def train_test_split(data, test_ratio=0.2, seed=42):
    random.seed(seed)
    shuffled = data[:]
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


# ---------- Naive Bayes ----------

class NaiveBayesSpamClassifier:
    """
    Multinomial Naive Bayes classifier for text.

    Trained probabilities:
      P(class)                — prior, fraction of training messages in that class
      P(word | class)         — likelihood, with Laplace smoothing to handle
                                 unseen words (add 1 to every count so no
                                 probability is ever exactly zero)
    """

    def __init__(self):
        self.class_priors = {}          # P(spam), P(ham)
        self.word_counts = {}           # word_counts[class][word] = count
        self.class_word_totals = {}     # total word count per class
        self.vocab = set()

    def train(self, training_data):
        class_message_counts = {}
        self.word_counts = {"spam": {}, "ham": {}}
        self.class_word_totals = {"spam": 0, "ham": 0}

        for label, message in training_data:
            class_message_counts[label] = class_message_counts.get(label, 0) + 1
            for word in tokenize(message):
                self.vocab.add(word)
                self.word_counts[label][word] = self.word_counts[label].get(word, 0) + 1
                self.class_word_totals[label] += 1

        total_messages = len(training_data)
        self.class_priors = {
            label: count / total_messages
            for label, count in class_message_counts.items()
        }

    def _log_likelihood(self, message, label):
        """log P(message | class), using Laplace smoothing."""
        vocab_size = len(self.vocab)
        total_words_in_class = self.class_word_totals[label]
        log_prob = 0.0
        for word in tokenize(message):
            word_count = self.word_counts[label].get(word, 0)
            # Laplace (add-1) smoothing: every word gets +1 count,
            # and the denominator grows by the vocab size to compensate,
            # so an unseen word never produces a zero probability.
            prob = (word_count + 1) / (total_words_in_class + vocab_size)
            log_prob += math.log(prob)
        return log_prob

    def predict(self, message):
        """Return (predicted_label, scores_dict)."""
        scores = {}
        for label in self.class_priors:
            log_prior = math.log(self.class_priors[label])
            log_likelihood = self._log_likelihood(message, label)
            scores[label] = log_prior + log_likelihood

        predicted = max(scores, key=scores.get)
        return predicted, scores

    def most_indicative_words(self, label, top_n=10):
        """
        Words most strongly associated with a class, by likelihood ratio.
        Useful for explaining *why* the model classified something a
        certain way, not just what it predicted.
        """
        other_label = "ham" if label == "spam" else "spam"
        vocab_size = len(self.vocab)
        ratios = []
        for word in self.vocab:
            p_word_given_label = (self.word_counts[label].get(word, 0) + 1) / \
                                  (self.class_word_totals[label] + vocab_size)
            p_word_given_other = (self.word_counts[other_label].get(word, 0) + 1) / \
                                  (self.class_word_totals[other_label] + vocab_size)
            ratios.append((word, p_word_given_label / p_word_given_other))
        ratios.sort(key=lambda x: x[1], reverse=True)
        return ratios[:top_n]


# ---------- Evaluation ----------

def evaluate(classifier, test_data):
    tp = fp = tn = fn = 0
    for true_label, message in test_data:
        predicted, _ = classifier.predict(message)
        if predicted == "spam" and true_label == "spam":
            tp += 1
        elif predicted == "spam" and true_label == "ham":
            fp += 1
        elif predicted == "ham" and true_label == "ham":
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / len(test_data)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def print_evaluation(results):
    print("-" * 40)
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall:    {results['recall']:.2%}")
    print(f"F1 score:  {results['f1']:.2%}")
    cm = results['confusion_matrix']
    print(f"\nConfusion matrix:")
    print(f"  True Positive (spam correctly caught):  {cm['tp']}")
    print(f"  False Positive (ham wrongly flagged):   {cm['fp']}")
    print(f"  True Negative (ham correctly passed):   {cm['tn']}")
    print(f"  False Negative (spam missed):           {cm['fn']}")
    print("-" * 40)


# ---------- Main ----------

if __name__ == "__main__":
    print("Loading dataset...")
    data = load_dataset(DATA_PATH)
    print(f"Loaded {len(data)} messages "
          f"({sum(1 for l, _ in data if l == 'spam')} spam, "
          f"{sum(1 for l, _ in data if l == 'ham')} ham)")

    train_data, test_data = train_test_split(data)
    print(f"Train: {len(train_data)}  Test: {len(test_data)}\n")

    classifier = NaiveBayesSpamClassifier()
    classifier.train(train_data)

    results = evaluate(classifier, test_data)
    print_evaluation(results)

    print("\nTop 10 words most indicative of SPAM:")
    for word, ratio in classifier.most_indicative_words("spam"):
        print(f"  {word:15s} ratio={ratio:.1f}")

    print("\nTop 10 words most indicative of HAM:")
    for word, ratio in classifier.most_indicative_words("ham"):
        print(f"  {word:15s} ratio={ratio:.1f}")

    print("\n" + "=" * 40)
    while True:
        msg = input("\nEnter a message to classify (or 'quit'): ")
        if msg.lower() == "quit":
            break
        label, scores = classifier.predict(msg)
        print(f"Prediction: {label.upper()}")
        print(f"  log-score spam: {scores['spam']:.2f}")
        print(f"  log-score ham:  {scores['ham']:.2f}")
