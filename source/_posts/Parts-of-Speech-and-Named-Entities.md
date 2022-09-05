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
  - [Closed classes](#closed-classes)
  - [Open classes](#open-classes)

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
![pos tagging](../figures/POStagging.png)
## Closed classes
Closed classes are those with relatively **fixed membership**, such as prepositions; new prepositions are rarely coined.

Closed class words are generally **function words** (e.g. of, it, and) which tend to be very short, occur frequently, and often have structuring uses in grammar.
## Open classes