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
Instead of splitting by words (a huge vocabulary, lots of `[UNK]`) or characters (no meanings), we use **subwords**—a compromise: rare words are broken down into pieces (`##ing`, `##ed`).
We train a single **shared vocabulary** on the training texts (questions and answers are in the same language and domain). Special tokens: `[PAD]` (index 0), `[UNK]`, `[START]`, `[END]`.

For example if 'play' is popular word but 'playing' not, normally 'playing' would end up as `[UNK]`, but thanks to subword tokenization we have 'play' and (`##ing'), so we can create 'playing' out of it - in the result we can create all needed words using smaller vocabulary.

## Model
The model is based on the self-attention mechanism, which is faster than traditional RNNs and enables a better understanding of language. This technique has been described in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) paper.

### Positional Embedding
Traditional RNN models process tokens one by one, so order is known, but Self-attention allows model to process tokens paralely which speeds up processing time but model loses order of tokens, that's why regular embedding is not enough because it only gives information about what token it is but doesn't know where it stands, so Positional Embedding also provides information about token position.

Attention Is All You Need 3.4, 3.5

### What is attention mechanism?
The attention mechanism in deep learning is a technique that allows a neural network to dynamically select and weight the most important parts of the input data at any given moment, rather than treating all elements equally. How does the attention mechanism work?

This technique is inspired by human vision and the human mind, which focus only on a selected part of the environment at any given moment. The process consists of three main steps:

- Importance assessment: The model calculates a probability distribution that determines the importance of individual input elements (e.g., words in a sentence or pixels in an image).

- Scaling: Relevant elements are amplified, while less important ones are attenuated or ignored. - Focused processing: The scaled input data is then fed into the next part of the network, yielding a more precise result. Why is attention so important?

- Context and long sequences: It helps models remember relationships between words that are far apart in long texts. Parallel processing: In modern architectures, this mechanism has replaced older sequential networks (RNN/LSTM), allowing entire sentences to be analyzed at once. Better interpretability: It allows us to see what the model “focused on” when making a decision.

![alt text](assets/attention.png)

https://medium.com/@prakhargannu/attention-mechanism-in-deep-learning-simplified-d6a5830a079d

#### GlobalSelfAttention

GlobalSelfAttention is part of the encoder and helps the model understand the relationships between words in context. Since the entire input sequence is available, the encoder can attend to every token, including both preceding and following words. This is important because the meaning of a word often depends on its context. For example, in “I sat on the bank of the river,” the word “river” helps determine which meaning of “bank” is intended. If the encoder could only look backward, it would miss important information from subsequent words.

Another advantage of self-attention is parallel processing. Unlike traditional RNNs, which process tokens sequentially, self-attention allows the model to process representations of all input tokens in parallel. However, this means that the model needs positional embeddings to capture information about token order, since self-attention alone does not inherently encode word positions.

Attention Is All You Need 3.2.3

#### CausalSelfAttention
Part of decoder, during training model can see future tokens because of teacher forcing which provides the decoder with the entire response all at once. Without a mask, the model at position t would see the token it is supposed to predict and would learn to copy. The triangular mask ensures that, during training, the model faces exactly the same conditions as during generation. In other words, we force model to see only already generated response without seeing next tokens.

Attention Is All You Need 3.2.3

#### CrossAttention
Cross-attention is part of the decoder. CausalSelfAttention focuses on the target sequence, but it does not directly access the context provided by the input question. CrossAttention acts as a bridge between the encoder's GlobalSelfAttention and the decoder's CausalSelfAttention, allowing the decoder to use information from the input when generating a response. This helps the model generate answers that are relevant to the question rather than producing text without considering the input context.

Attention Is All You Need 3.2.3

#### Attention heads

#### Add + LayerNormalization

#### Encoder

#### Decoder

#### FeedForward

#### Transformer
