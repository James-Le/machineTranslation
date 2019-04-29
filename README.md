# machineTranslation

Baselines are based on different seq2seq models with pre-trained word2vec embeddings, attention mechanism, beam search strategy.

![image](https://github.com/James-Le/machineTranslation/blob/master/baseline_loss.png)

![image](https://github.com/James-Le/machineTranslation/blob/master/BLEU.png)

The Dual learning model is contructed by Ruosen Li: https://github.com/lrscy/Unsupervised-Dual-Learning-Neural-Machine-Translation-Model/tree/master/pytorch-dual-learning

For the model's overall architecture, code is simulated from https://github.com/jmhIcoding/machine_translation. I have made some improvements including fixing exceptions, trying pre-trained embeddings and changing its generative strategy.
