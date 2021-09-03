# Multimodal Sentiment Analysis in Conversational Videos Based on Recurrent Neural Networks
## Overview
Multimodal Sentiment Analysis has become an essential research topic in Artificial Intelligence due to its potential applications in various challenging tasks, such as intelligent human-computer interaction, user behavior understanding and others. 

The goal of this project is to utilize different modalities(text, audio and visual) in a conversational video for emotion recognition.

## Related Dataset
The project uses Multimodal EmotionLines Dataset(MELD), which is a multimodal dataset extended from the EmotionLines dataset. EmotionLines dataset contains dialogues from the popular American TV sitcom *Friends*, where each dialogue contains utterances from multiple speakers. MELD not only includes textual dialogues available in EmotionLines, but also their corresponding visual and audio features.

Please visit https://github.com/declare-lab/MELD for more information about MELD.


## Model Architecture
In this project, I designed Multimodal Conversational Recurrent Neural Network(MC-RNN) based on Gated Recurrent Unit(GRU) and Global Attention to model context in conversations, so as to accurately predict the sentiment label and emotion label of every utterance in a conversational video.

<img width="638" alt="截屏2021-09-03 下午4 31 15" src="https://user-images.githubusercontent.com/37060800/131975662-04154bea-3898-4a81-925e-2ba12bb5ed4d.png">


## Example Dialgue and its Predictions Using MC-RNN
<img width="786" alt="Example Dialogue" src="https://user-images.githubusercontent.com/37060800/131972610-f5f4e067-d8d5-4c2b-aa0c-373ac1305b83.png">
 <img width="690" alt="截屏2021-09-03 下午4 03 36" src="https://user-images.githubusercontent.com/37060800/131972856-1efcaeab-98a5-4286-8355-c4b1f24c2ebe.png">


## Results
<img width="948" alt="截屏2021-09-03 下午8 47 35" src="https://user-images.githubusercontent.com/37060800/132007659-8a0c12b9-c95e-492e-b420-2d1bbbcdd1ec.png">





