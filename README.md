# Spam/Ham Message Classifier (Naive Bayes, from scratch)

A text classifier that labels SMS messages as spam or legitimate ("ham"),
using Multinomial Naive Bayes implemented directly from probability theory —
no `sklearn`, no ML framework, just Bayes' theorem and word frequency counts.

## How it works

Naive Bayes classifies by comparing, for each class (spam/ham), how probable
the message is under that class, then picking whichever class is more likely:

```
P(class | message) ∝ P(class) × P(message | class)
```

- **P(class)** — the prior: what fraction of training messages were spam/ham
- **P(message | class)** — the likelihood: assuming words are independent
  (the "naive" assumption), this is the product of P(word | class) for every
  word in the message

Because multiplying many small probabilities together underflows quickly,
the implementation works in **log-space** — sums of logs instead of products
of probabilities — which is standard practice, not a shortcut.

### Laplace smoothing

If a word never appeared in the training data for a class, its raw
probability would be zero — and multiplying anything by zero would wipe out
the whole prediction. Every word count is smoothed with **add-1 (Laplace)
smoothing**:

```
P(word | class) = (count(word, class) + 1) / (total_words(class) + vocab_size)
```

This guarantees no probability is ever exactly zero, while barely affecting
words that appear frequently.

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
— 5,574 real SMS messages, labeled spam or ham, standard benchmark dataset
for text classification.

Download: https://raw.githubusercontent.com/adityaiiitmk/Datasets/master/SMSSpamCollection

Save it as `data/SMSSpamCollection` (tab-separated, no file extension).

## Usage

```bash
python spam_classifier.py
```

On startup, it trains on 80% of the dataset, evaluates on the remaining 20%,
prints accuracy/precision/recall/F1 and a confusion matrix, then shows the
words most strongly associated with each class. You can then type your own
messages to classify interactively.

## Evaluation metrics explained

- **Accuracy** — overall fraction correct
- **Precision** — of messages flagged as spam, how many actually were spam
  (low precision = too many legitimate messages wrongly blocked)
- **Recall** — of actual spam messages, how many were caught
  (low recall = spam getting through)
- **F1** — harmonic mean of precision and recall, a single balanced score

Precision and recall are reported separately (not just accuracy) because
spam filtering has asymmetric costs: a missed spam message (low recall) is
annoying, but a legitimate message wrongly blocked (low precision) can be
worse — accuracy alone hides this trade-off.

## Explainability

`most_indicative_words()` reports the words with the highest likelihood
ratio between classes — i.e. words that are disproportionately common in
spam vs ham (or vice versa). This isn't just a nice-to-have: it's a basic,
honest way to inspect *why* the model behaves as it does, which is exactly
what a black-box `.fit()` call from a library would hide.

## Project structure

```
├── spam_classifier.py
├── data/
│   └── SMSSpamCollection
└── README.md
```

## Limitations

- The "naive" independence assumption (that words appear independently of
  each other given the class) is technically false — word order and context
  are completely ignored — but works surprisingly well in practice for spam
  detection, which is exactly why Naive Bayes remains a standard baseline.
- Tokenization is simple (lowercase, strip punctuation, split on whitespace)
  — no stemming, lemmatization, or handling of misspellings/obfuscation
  (e.g. "fr33" instead of "free"), which real spam often uses to evade
  filters.
- The dataset is English-only SMS messages from ~2012; a model trained on it
  would need retraining to generalize well to email spam or non-English text.
