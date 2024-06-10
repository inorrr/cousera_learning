# Generative AI with Large Language Models

## Module 1: Generative AI use cases, project lifecycle, and model pre-training

### Learning Objectives:
1. Discuss model pre-training and the value of continued pre-training vs fine-tuning
2. Define the terms Generative AI, large language models, prompt, and describe the transformer architecture that powers LLMs
3. Describe the steps in a typical LLM-based, generative AI model lifecycle and discuss the constraining factors that drive decisions at each step of model lifecycle
4. Discuss computational challenges during model pre-training and determine how to efficiently reduce memory footprint
5. Define the term scaling law and describe the laws that have been discovered for LLMs related to training dataset size, compute budget, inference requirements, and other factors.

### Introduction to LLMs and the generative AI project lifecycle

- Gen AI is a subset of traditional ML.
- Base models: GPT = BLOOM > FLAN-T5 > LLaMa > PaLM > BERT, in terms of parameter size.
- prompt -> model -> completion(output)
- next word perdiction is the base logic behind many tasks in text generation. For example, write essay with given prompt, summarization, translation, natural languate to machine code, entity extraction,
- Connect LLM to external source so that it can perform tasks that it has not be pre-trained on is an ongoing research area.

#### Text generation before transformer
- RNN: recurent neural networks. Look at the previous couple of words to predict the next word. Performance improves as model get larger, but still not good enough as human language is complex and sometimes contains ambiguity.

"Attentional is All You Need": the transformer paper published by Google and the University of Toronto.
Transformers can be:
- scaled efficiently to use multicore GPU
- parallel process to use larger tranining datasets.
- able to make attention to input meaning - crutoial


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
