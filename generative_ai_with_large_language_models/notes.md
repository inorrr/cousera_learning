# Generative AI with Large Language Models

## Module 1: Generative AI use cases, project lifecycle, and model pre-training

### Learning Objectives:
1. Discuss model pre-training and the value of continued pre-training vs fine-tuning
2. Define the terms Generative AI, large language models, prompt, and describe the transformer architecture that powers LLMs
3. Describe the steps in a typical LLM-based, generative AI model lifecycle and discuss the constraining factors that drive decisions at each step of model lifecycle
4. Discuss computational challenges during model pre-training and determine how to efficiently reduce memory footprint
5. Define the term scaling law and describe the laws that have been discovered for LLMs related to training dataset size, compute budget, inference requirements, and other factors.

### Introduction to LLMs and the generative AI project lifecycle

#### Generative AI & LLMs

- Gen AI is a subset of traditional ML.
- Base models: GPT = BLOOM > FLAN-T5 > LLaMa > PaLM > BERT, in terms of parameter size.
- prompt -> model -> completion(output)

#### LLM use cases and tasks
- next word perdiction is the base logic behind many tasks in text generation. For example, write essay with given prompt, summarization, translation, natural languate to machine code, entity extraction,
- Connect LLM to external source so that it can perform tasks that it has not be pre-trained on is an ongoing research area.

#### Text generation before transformer
- RNN: recurent neural networks. Look at the previous couple of words to predict the next word. Performance improves as model get larger, but still not good enough as human language is complex and sometimes contains ambiguity.

"Attentional is All You Need": the transformer paper published by Google and the University of Toronto.
Transformers can be:
- scaled efficiently to use multicore GPU
- parallel process to use larger tranining datasets.
- able to make attention to input meaning - crutoial

#### Transformers Architecture

- Ability to learn the relevlence between words in a sentence. 
- To apply *attention weights* to those relationships so that the model learns the relevance of each word to each other words no matter where they are in the input.
- Attention map: used to illustrate the attention weights between each word and every other word.
- self attention.

Simplified diagram of the transformer: two distinct parts: encoder and decoder.

![Screenshot 2024-06-10 at 3 05 08 PM](https://github.com/inorrr/cousera_learning/assets/94703030/8d4c0a08-692b-45ca-ac62-4be4d7464d5f)

- Tokenize: convert words to numbers, with each number representing a position in a dictionary of all the possible words that the model can work with.
- Each token ID in the vocabulary is matched to a multi-dimensional vector, and the intuition is that these vectors learn to encode the meaning and context of individual tokens in the input sequence
- there are many set of heads(self attention), each self-attention head will learn a different aspect of language.

#### Generating text with transformers
Example: translation (sequence to sequence task) from French to English
1. First, you'll tokenize the input words using this same tokenizer that was used to train the network.
2. These tokens are then added into the input on the encoder side of the network, passed through the embedding layer, and then fed into the multi-headed attention layers.
3. The outputs of the multi-headed attention layers are fed through a feed-forward network to the output of the encoder.
4. At this point, the data that leaves the encoder is a deep representation of the structure and meaning of the input sequence.
5. This representation is inserted into the middle of the decoder to influence the decoder's self-attention mechanisms.
6. Next, a start of sequence token is added to the input of the decoder. This triggers the decoder to predict the next token, which it does based on the contextual understanding that it's being provided from the encoder.
7. The output of the decoder's self-attention layers gets passed through the decoder feed-forward network and through a final softmax output layer. At this point, we have our first token

### LLM pre-training and scaling laws

## Module 2: Fine-tuning and evaluating large language models

### Learning Objectives:
1. Describe how fine-tuning with instructions using prompt datasets can improve performance on one or more tasks
2. Define catastrophic forgetting and explain techniques that can be used to overcome it
3. Define the term Parameter-efficient Fine Tuning (PEFT)
4. Explain how PEFT decreases computational cost and overcomes catastrophic forgetting
5. Explain how fine-tuning with instructions using prompt datasets can increase LLM performance on one or more tasks

### Fine-tuning LLMs with instruction

### Parameter efficient fine-tuning

## Module 3: Reinforcement learning and LLM-powered applications

### Reinformcement Learning from human feedback

### LLM-powered applications
