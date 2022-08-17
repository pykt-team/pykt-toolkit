# Models


##  AKT(Attentive knowledge tracing)
Attentive knowledge tracing (AKT) is a model which couples flexible attention-based neural network models with a series of novel, interpretable model components inspired by cognitive and psychometric models.

![AKT](../pics/akt.png)

[Ghosh, Aritra, Neil Heffernan, and Andrew S. Lan. "Context-aware attentive knowledge tracing." Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 2020.](https://dl.acm.org/doi/pdf/10.1145/3394486.3403282)

##  ATKT
Adversarial training (AT) based KT method (ATKT) to enhance KT model's generalization and thus push the limit of KT. To better implement AT, ATKT proposes an efficient attentive-LSTM model as KT backbone.

![ATKT](../pics/atkt.png)


[Guo, Xiaopeng, et al. "Enhancing Knowledge Tracing via Adversarial Training." Proceedings of the 29th ACM International Conference on Multimedia. 2021.](https://arxiv.org/pdf/2108.04430)



##  DKT-Forget
 DKT-Forget extends the deep knowledge tracing model to consider forgetting by incorporating multiple types of information related to forgetting.

![DKT-Forget](../pics/dkt_forget.png)


[Nagatani, Koki, et al. "Augmenting knowledge tracing by considering forgetting behavior." The world wide web conference. 2019.](https://dl.acm.org/doi/10.1145/3308558.3313565)


##  DKT(Deep Knowledge Tracing)
DKT is the first model uses Recurrent Neural Networks (RNNs) to solve Knowledge Tracing.

![DKT](../pics/dkt.png)

[Piech, Chris, et al. "Deep knowledge tracing." Advances in neural information processing systems 28 (2015).](https://proceedings.neurips.cc/paper/2015/file/bac9162b47c56fc8a4d2a519803d51b3-Paper.pdf)

##  DKT+
DKT+ introduces regularization terms that correspond to reconstruction and waviness to the loss function of the original DKT model to enhance the consistency in prediction.

<!-- ![DKT+](../pics/dkt+.png) -->


[Yeung, Chun-Kit, and Dit-Yan Yeung. "Addressing two problems in deep knowledge tracing via prediction-consistent regularization." Proceedings of the Fifth Annual ACM Conference on Learning at Scale. 2018.](https://arxiv.org/pdf/1806.02180)

##  DKVMN (Dynamic key-value memory networks)
DKVMN has one static matrix called "key" which stores the knowledge concepts, and the other dynamic matrix called value, which stores and updates the mastery levels of corresponding concepts.

![DKVMN](../pics/dkvmn.png)


[Zhang, Jiani, et al. "Dynamic key-value memory networks for knowledge tracing." Proceedings of the 26th international conference on World Wide Web. 2017.](https://arxiv.org/abs/1611.08108)



##  GKT(graph-based knowledge tracing)
GKT is a GNN-based knowledge tracing method that casts the knowledge structure as a graph, enabling us to reformulate the Knowledge tracing task as a time-series node-level classification problem in the GNN.

![GKT](../pics/gkt.png)

[Nakagawa, Hiromi, Yusuke Iwasawa, and Yutaka Matsuo. "Graph-based knowledge tracing: modeling student proficiency using graph neural network." 2019 IEEE/WIC/ACM International Conference On Web Intelligence (WI). IEEE, 2019.](https://ieeexplore.ieee.org/abstract/document/8909656/)

##  HawkesKT
HawkesKT is the first to introduce Hawkes process to model temporal cross effects in KT.
![HawkesKT](../pics/hawkes.png)

[Wang, Chenyang, et al. "Temporal cross-effects in knowledge tracing." Proceedings of the 14th ACM International Conference on Web Search and Data Mining. 2021.](http://www.thuir.cn/group/~mzhang/publications/WSDM2021-WangChenyang.pdf)

##  IEKT(Individual Estimation Knowledge Tracing)
Individual Estimation Knowledge Tracing (IEKT) estimates the students' cognition of the question before response prediction and assesses their knowledge acquisition sensitivity on the questions before updating the knowledge state.

![IEKT](../pics/iekt.png)

[Long, Ting, et al. "Tracing knowledge state with individual cognition and acquisition estimation." Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2021.](https://wnzhang.net/papers/2021-sigir-iekt.pdf)

##  KQN

KQN uses neural networks to encode student learning activities into knowledge state and skill vectors, and models the interactions between the two types of vectors with the dot product. 

![KQN](../pics/kqn.png)

[Lee, Jinseok, and Dit-Yan Yeung. "Knowledge query network for knowledge tracing: How knowledge interacts with skills." Proceedings of the 9th international conference on learning analytics & Knowledge. 2019.](https://arxiv.org/pdf/1908.02146)


##  LPKT(Learning Processconsistent Knowledge Tracing)

LPKT monitors students' knowledge state by directly modeling their learning process.

![LPKT](../pics/lpkt.png)

[Shen, Shuanghong, et al. "Learning process-consistent knowledge tracing." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.](http://staff.ustc.edu.cn/~huangzhy/files/papers/ShuanghongShen-KDD2021.pdf)

##  SAINT(Separated Self-AttentIve Neural Knowledge Tracing)

SAINT has an encoder-decoder structure where the exercise and response embedding sequences separately enter, respectively, the encoder and the decoder. This is the first work to suggest an encoder-decoder model for knowledge tracing that applies deep self-attentive layers to exercises and responses separately.

![SAINT](../pics/saint.png)

[Choi, Youngduck, et al. "Towards an appropriate query, key, and value computation for knowledge tracing." Proceedings of the Seventh ACM Conference on Learning@ Scale. 2020.](https://arxiv.org/pdf/2002.07033.pdf)

##  SAKT(Self Attentive Knowledge Tracing)

SAKT model is proposed to solve the problem of sparse data processing. SAKT identifies the KCs from the student's past activities that are relevant to the given KC and predicts their mastery based on the relatively few KCs that it picked.

![SAKT](../pics/sakt.png)

[Pandey, Shalini, and George Karypis. "A self-attentive model for knowledge tracing." arXiv preprint arXiv:1907.06837 (2019).](https://arxiv.org/pdf/1907.06837.pdf)

##  SKVMN

This model unifies the strengths of recurrent modeling capacity and memory capacity of the existing deep learning KT models for modeling student learning.

![SKVMN](../pics/skvmn.png)

[Abdelrahman, Ghodai, and Qing Wang. "Knowledge tracing with sequential key-value memory networks." Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 2019.](https://arxiv.org/pdf/1910.13197.pdf)
