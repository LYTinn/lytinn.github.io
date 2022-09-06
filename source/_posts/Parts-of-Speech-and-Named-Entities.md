---
title: Parts of Speech and Named Entities
date: 2022-09-05 08:56:06
tags:
- NLP
categories:
- Natural Language Processing
---
- [POS and NER: Sequence Labeling](#pos-and-ner-sequence-labeling)
  - [Parts of speech (POS)](#parts-of-speech-pos)
  - [Named Entity Recognition (NER)](#named-entity-recognition-ner)
  - [Sequence labeling:](#sequence-labeling)
- [POS Tagging](#pos-tagging)
  - [Why POS tagging is challenging?](#why-pos-tagging-is-challenging)
  - [POS Tagging with Hidden Markov Model (HMM)](#pos-tagging-with-hidden-markov-model-hmm)
    - [POS Tagging in probabilistic view:](#pos-tagging-in-probabilistic-view)
    - [Markov Chains](#markov-chains)
    - [Hidden Markov Model](#hidden-markov-model)
    - [POS tagging with HMM](#pos-tagging-with-hmm)
- [Named Entity Recognition](#named-entity-recognition)

# POS and NER: Sequence Labeling
## Parts of speech (POS) 
The parts of speech (POS) refers to **word classes** such as Noun, Verb, Adjective, etc. 

It also known as lexical categories, word classes, morphological classes, lexical tags. Knowing word class tells us more about neighboring words and syntactic structure. E.g. nouns in English are often preceded by determiners and adjectives. Verbs have dependency links to nouns. POS tagging is a key aspect of parsing sentence structure.

## Named Entity Recognition (NER)
The named entity recognition (NER) is proper name for person, location, organization, etc. 

NEs are useful clues to sentence structure and meaning understanding. Knowing if a named entity like Washington is a name of person, a place, or a university is important to tasks like question answering and information extraction.

## Sequence labeling:
- **POS tagging** takes a sequence of words and assigns each word a POS like NOUN or VERB.
- **Named Entity Recognition (NER)** assigns words or phrases tags like PERSON, LOCATION, or ORGANIZATION.

# POS Tagging
There are multiple POS tagsets defined. For example, the following are 17 parts of speech in the Universal Dependecies tagset.
![pos tagging](/figures/POStagging.png)

**Closed classes** are those with relatively **fixed membership**, such as prepositions; new prepositions are rarely coined.

Closed class words are generally **function words** (e.g. of, it, and) which tend to be very short, occur frequently, and often have structuring uses in grammar.

Nouns and verbs are maong **open classes**; new nouns and verbs like *iPhone* or *fax* are continually being created or borrowed.

**Prepositions** indicate spatial ot temporal relations, and relations.

**Determiners** like *this* and *that* can mark the start of article English noun phrase.

**Pronouns** acts as a shorthand for referring to an entity or event.
- **Persional pronouns** refer to person or entities (you, she, I, it, me, etc.)
- **Possessive pronouns** are forms of personal pronouns that indicate either actual possession ot more often just an abstract relation between the person and some object (my, your, his, her, its, one's, our, their, etc.)
- **Wh-pronouns** (What, who, whom, whoever, etc.) are used in certain question forms, or act as complementizers.

The 45-tag Penn Treebanc tagset is another tagset example.
![45-tag](/figures/45-tag.png)

Part-of-speech tagging is the process of assigning a part-of-speech to each word in a text.

## Why POS tagging is challenging?
Words are **ambiguous** -- have more than one possible part-of-speech. And the example words may contain multiple parts-of-speech. For example:
> **Book** that flight; Hand me that **book**.

> The **back** door; On my **back**; Win the voters **back**; Promised to **back** the bill.

How to determine the tag of word is a hard problem, there are many algorithms discusing and trying to solve it. The accuracy of POS tagging algorithms is extremely high, about 97%. However, the most-frequent-tag baseline has an accuracy of about 92%.

Here, we won't discuss tagging algorithm in detail. We will focus on predicting and generating sentence using tags.

## POS Tagging with Hidden Markov Model (HMM)
A sequence labeler assigns a label to each unit (e.g. word) in a sequence (i.e. sentence), thus mapping a sequence of observations to a sequence of labels of the same length.

The HMM is a classic probabilitic sequence model. Given a sequence of words, it computes a probability distribution over possible sequences of labels and chooses that best label sequence.

### POS Tagging in probabilistic view:
- Consider all possible sequences of tag (each tag for one word)
- Out of this universe of sequences, choose the tag sequence which is most probable given the observation sequence of $n$ words $w_1\dots w_n$.

### Markov Chains
The HMM is based on augmenting the **Markov chain**. A Markov chain is a model on the probabilities of sequences of random variables (or states), each of which can take on values from some set. These sets can be tags, words, or symbols representing anything. For example, a set of possible weather states includes HOT, COLD, WARM.

Markov assumption: The probability of next state only depends on the current state, e.g. predict tomorrow's weather only based on todays' weather.

$$
P(q_i = a | q_1\dots q_{i-1}) = P(q_i = a|q_{i-1})
$$
where $q_1\dots q_{i-1}$ is a sequence of states. The probability of next state $q_i=a$ only depends on the state $q_{i-1}$.

$$
Q = q_1q_2\dots q_N\qquad
$$ 
$Q$ is a set of N **states**.
$$
A = \left(\begin{matrix}
    a_{11}&\dots&a_{1N}\\
    \vdots&\ddots\\
    a_{N1}&\dots&a_{NN}
\end{matrix}\right)\qquad
$$
$A$ is **transition probability matrix** A, each $a_{ij}$ representing the probability of moving from state $i$ to state $j$, s.t. $\sum_{j=1}^na_{ij} = 1\quad \forall i$
$$
\pi = \pi_1, \pi_2, \dots, \pi_N
$$
$\pi$ is the initial probability distribution over states. $\pi_i$ is the probability that the Markov chain will start in state $i$. Some states $j$ may have $\pi_i = 0$, meaning that they cannot be initial states. Also, $\sum_{i=1}^n\pi_i = 1$.
### Hidden Markov Model
A Markov chain is useful when we need to compute a probability for a sequence of observable events, e.g., based on today's weather to predict tomorrow's weather, and based on current word to predict next word (as in bigram model).

In many cases, however, the events we are interested in are **hidden**. For example, in POS tagging, we can only observe words, but not there tags. We cannot use the current tag to predict the next tag for a word sequence. We call the tags hidden because they are not observed.

A hidden Markov model (HMM) allows us to talk about both **observed events** and **hidden events**. For POS tagging:
- Observed event are the words in the input sentence. 
- Hidden events are the part-of-speech tags for these words.
- The observed events are considered as causal factors in this probabilistic model.

$$
Q = q_1q_2\dots q_N\qquad
$$ 
$Q$ is a set of N **states**.
$$
A = \left(\begin{matrix}
    a_{11}&\dots&a_{1N}\\
    \vdots&\ddots\\
    a_{N1}&\dots&a_{NN}
\end{matrix}\right)\qquad
$$
$A$ is **transition probability matrix** A, each $a_{ij}$ representing the probability of moving from state $i$ to state $j$, s.t. $\sum_{j=1}^na_{ij} = 1\quad \forall i$
$$
O = o_1o_2\dots o_T
$$
$O$ is a sequence of $T$ **observations**, each one drawn from a vocabulary $V = v_1v_2\dots v_V$

$$
B = b_i(o_t)
$$
$B$ is a sequence of **observation likelihoods**, also called **emission probabilities**, each expressing the probability of an observation $o_t$ being generated from a state $q_i$.

$$
\pi = \pi_1, \pi_2, \dots, \pi_N
$$
$\pi$ is the initial probability distribution over states. $\pi_i$ is the probability that the Markov chain will start in state $i$. Some states $j$ may have $\pi_i = 0$, meaning that they cannot be initial states. Also, $\sum_{i=1}^n\pi_i = 1$.

A first-order hidden Markov model instantiates two simplifying assumptions:

**Markov Assumption**: the probability of a particular state depends only on the previous state
$$
P(q_i=a|q_1q_2\dots q_{i-1}) = P(q_i=a|q_{i-1})
$$
**Output Independence Assumption**: the probability of an output observation $o_i$ depends only on the state $q_i$ that produced the observation and not on any other states or any otehr observations
$$
P(o_i|q_1,\dots, q_T; o_1,\dots, o_T) = P(o_i|q_i)
$$

**Decoding**: For any model, such as an HMM, that contains hidden variables, the task of determining the hidden variables sequence corresponding to the sequence of observations is called decoding.

### POS tagging with HMM
Out of all possible sequences of $n$ tags $t_1\dots t_n$ the single tag sequence such that $P(t_1\dots t_n|w_1\dots w_n)$  is highest.
$$
\hat{t}_{1:n} = \arg \max_{t_{1:n}}P(t_{1:n}|w_{1:n})
$$
where the $\hat{t}$ means **our estimate** of the best one.

Use Bayes rule $P(y|x) = \frac{P(x|y)P(y)}{P(x)}$ to transform this equation into a set of other probabilities that are easier to compute:
$$
\hat{t}_{1:n} = \arg \max_{t_{1:n}}P(t_{1:n}|w_{1:n}) = \arg \max_{t_{1:n}}\frac{P(w_{1:n}|t_{1:n})P(t_{1:n})}{P(w_{1:n})} = \arg\max_{t_{1:n}}P(w_{1:n}|t_{1:n})P(t_{1:n})
$$
Applying the two assumptions of HMM, we can get:
$$
\hat{t}_{1:n} = \arg\max_{t_{1:n}}\prod_{i=1}^nP(w_i|t_i)P(t_i|t_{i-1})
$$
# Named Entity Recognition