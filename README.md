# LLM Preference Arena - Kaggle

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](#)
[![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)](#)
[![HuggingFace](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white)](#)

## 🚀 About This Project

This project is my experimentation & solution for the Kaggle Competition: [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning/overview)

The competiotion challenges participants to predict which responses users will prefer in head-to-head battles between chatbots powered by large language models (LLMs). The dataset consists of real-world conversations from the Chatbot Arena, where users interact with two anonymous LLMs and select their preferred answer.

The goal is to develop a machine learning model that can accurately predict user preferences between competing chatbot responses. This task is closely related to building "reward models" or "preference models" in the context of reinforcement learning from human feedback (RLHF). The competition highlights the importance of overcoming common biases such as position bias, verbosity bias, and self-enhancement bias, which can affect preference predictions.

## 🔎 Exploration

As we can see the dataset is already balanced as no evidence of biased selection is made from users due to boredom of selecting the first answer as presented in some papers.

![Preference](/assets/winners.png)

And the dataset mainly utilized OpenAI GPT variants for generation as well as Claude 2.1!
Further more regarding the win rate again models from OpenAI GPT topped the leaderboard probably due to techinques used like RLHF or DPO.
![Models Usage](/assets/models_usage.png)
![Win Rate](/assets/models_win_rate.png)

In perspective of text there is no obvious significant leaning towards more on lengthier or smaller outputs.
Nor more or less punctuation play an important role, neither new lines.

![Winner Length Diff](/assets/winner_length_diff_count.png)
![Winner Length Diff per Prompt Length](/assets/winner_length_prompt_and_diff_length.png)
![Winner More Punctuation](/assets/more_punctuation_winner.png)
