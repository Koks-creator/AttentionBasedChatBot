# Attention Based Chat Bot

## Short description
The goal of this project was to make generative chat bot based on attention mechanism. It's huge progress when compering to my <a href="https://github.com/Koks-creator/IntentsChatBot">last chat bot project</a> which was built on simple text classification. 

[![video](https://img.youtube.com/vi/jeiZiWP6Lzk/0.jpg)](https://www.youtube.com/watch?v=jeiZiWP6Lzk)

## Setup
  - python 3.13+ or sth like that
  - ```pip install -r requirements.txt```

## Packages
```
numpy==2.4.3
flask==3.1.3
keras==3.13.2
numpy==2.4.3
tensorflow==2.21.0
```

## Datasets

### DailyDialog
Multi-turn dialog dataset which uses less noisy and human-written languag a high-quality designed for natural language understanding, conversational modeling, and emotion/intent recognition.

Splits:
 - train: 11118
 - validation: 1000
 - test: 1000

https://arxiv.org/pdf/1710.03957

### Cornell Movie-Dialogs Corpus
A large metadata-rich collection of fictional conversations extracted from raw movie scripts. (220,579 conversational exchanges between 10,292 pairs of movie characters in 617 movies).

Splits:
 - train: 178887
 - validation: 9237
 - test: 76052

https://arxiv.org/pdf/2502.16274
https://huggingface.co/datasets/cornell-movie-dialog/cornell_movie_dialog

### Synthetic-Persona-Chat
Datasets which contains 20000 synthetic dialogues based on user profiles (personas).

Splits:
- train: 213428
- validation: 24347

https://arxiv.org/pdf/2312.10007

### Blended Skill Talk
This dataset contains conversations between two personas with additional context previous utterances free messages guided messages suggestions and guided chosen suggestions allowing for the creation of natural multi-modal conversations with personality empathy and knowledge.

Splits:
- train: 54020
- validation: 11103

https://huggingface.co/datasets/anezatra/blended-skill-talk


## Tokenization

## Model
The model is based on the self-attention mechanism, which is faster than traditional RNNs and enables a better understanding of language. This technique has been described in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) paper.

### Positional Embedding

### What is attention mechanism?
The attention mechanism in deep learning is a technique that allows a neural network to dynamically select and weight the most important parts of the input data at any given moment, rather than treating all elements equally. How does the attention mechanism work?

This technique is inspired by human vision and the human mind, which focus only on a selected part of the environment at any given moment. The process consists of three main steps:

- Importance assessment: The model calculates a probability distribution that determines the importance of individual input elements (e.g., words in a sentence or pixels in an image).

- Scaling: Relevant elements are amplified, while less important ones are attenuated or ignored. - Focused processing: The scaled input data is then fed into the next part of the network, yielding a more precise result. Why is attention so important?

- Context and long sequences: It helps models remember relationships between words that are far apart in long texts. Parallel processing: In modern architectures, this mechanism has replaced older sequential networks (RNN/LSTM), allowing entire sentences to be analyzed at once. Better interpretability: It allows us to see what the model “focused on” when making a decision.

![alt text](assets/attention.png)

https://medium.com/@prakhargannu/attention-mechanism-in-deep-learning-simplified-d6a5830a079d

#### GlobalSelfAttention

#### CausalSelfAttention

#### CrossAttention 
