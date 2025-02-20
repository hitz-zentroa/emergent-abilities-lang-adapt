<p align="center">
    <br>
    <p>Code for the Paper</p>
    <br>
    <h1 align="center">Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation</h1>
.
<p align="center">
<!--     <a href="https://github.com/hitz-zentroa/latxa/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/hitz-zentroa/latxa"></a> -->
    <a href="https://huggingface.co/collections/ahmedselhady/emergent-abilities-of-large-language-models-under-continued-679a2472a8f0931fa6c422c7"><img alt="Models" src="https://img.shields.io/badge/🤗 -Models-ff3333"></a>
    <a href="https://huggingface.co/datasets/ahmedselhady/CoPain"><img alt="Models" src="https://img.shields.io/badge/🤗 -Dataset-blueviolet"></a>    
    <a href="https://arxiv.org/abs/2403.20266"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-blue"></a>
<!-- <br>
     <a href="http://www.hitz.eus/"><img src="https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet"></a>
    <a href="http://www.ixa.eus/?language=en"><img src="https://img.shields.io/badge/IXA-%20NLP%20Group-ff3333"></a>
    <br>
     <br> -->
</p>

Continued pretraining (CPT) is a popular approach to adapt existing large language models (LLMs) to new languages. When doing so, it is common practice to include a portion of English data in the mixture, but its role has not been carefully studied to date. In this work, we show that including English does not impact validation perplexity, yet it is critical for the emergence of downstream capabilities in the target language. We introduce a language-agnostic benchmark for in-context learning (ICL), which reveals catastrophic forgetting early on CPT when English is not included. This in turn damages the ability of the model to generalize to downstream prompts in the target language as measured by perplexity, even if it does not manifest in terms of accuracy until later in training, and can be tied to a big shift in the model parameters. Based on these insights, we introduce curriculum learning and exponential moving average (EMA) of weights as effective alternatives to mitigate the need for English. All in all, our work sheds light into the dynamics by which emergent abilities arise when doing CPT for language adaptation, and can serve as a foundation to design more effective methods in the future.



