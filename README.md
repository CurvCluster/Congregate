# CONGREGATE: Contrastive Graph Clustering in Curvature Space.

## Noteworthy Contribution

In this paper, we are the __FIRST__ to introduce __a novel curvature space__, supporting fine-grained curvature modeling, to graph clustering.

## Datasets

We evaluate our model on 4 benchmark datasets, and all the datasets are publicly available. Please refer to the following papers for further details on the datasets.

1. Cora and Citeseer.    
   
   > Sen, P.; Namata, G.; Bilgic, M.; Getoor, L.; Gallagher, B.; and Eliassi-Rad, T. 2008. Collective Classification in Network Data. AI Mag., 29(3): 93–106.
   > 
   > Devvrit, F.; Sinha, A.; Dhillon, I,; and Jain. P. S3GC: Scalable self-supervised graph clustering. In Advances in 36th NeurIPS, 2022.

2. MAG-CS
   
   > Park, N.; Rossi, R.; Koh, E.; Burhanuddin, I. A.; Kim, S.; Du, F.; Ahmed, N. K.; and Faloutsos, C. CGC: Contrastive graph clustering for community detection and tracking. In Proceedings of The ACM Web Conference, pages 1115–1126.  ACM, 2022.

3. Amazon-Photo
   
   > Li, B.; Jing, B.; and Tong, H. Graph communal contrastive learning. In Proceedings of The ACM Web Conference, pages 1203–1213. ACM, 2022.

## Implementation

Our model consists of M restricted manifolds and 1 free manifold.
The number of M, and the dimension of manifolds need to be configured for an specific instantiation. (Note that, the restricted manifolds have __learnable curvatures__, which is another novelty of our model.)
Also, the weighting coefficient alpha's are the hyperparameters of the loss function.

We give a sample implementation of CONGREGATE here.
We will release all the source code of project of Geometric Graph Clustering-Curvature after publication. 

The requirements is listed below.

+ Python 3.7
+ Pytorch >= 1.1
+ numpy
+ scikit-learn
+ networkx

## Baselines

We compare with 19 state-of-the-art baselines in total. The baselines are introduced in the Technical Appendix, and all the baselines are implemented according to the original papers. 

1. IJCAI'19 DAEGC
   
   > Chun Wang, Shirui Pan, Ruiqi Hu, Guodong Long, Jing Jiang, and Chengqi Zhang. Attributed graph clustering:  A deep attentional embedding approach. In Proceedings of  the 28th IJCAI, pages 3670–3676. ijcai.org, 2019.

2. WWW'20 SDCN
   
   > Deyu Bo, Xiao Wang, Chuan Shi, Meiqi Zhu, Emiao Lu, and Peng Cui. Structural deep clustering network. In Proceedings of WWW, pages 1400–1410. ACM / IW3C2, 2020.

3. SIGKDD'20 AGE
   
   > Ganqu Cui, Jie Zhou, Cheng Yang, and Zhiyuan Liu. Adaptive graph encoder for attributed graph embedding. In Proceedings of the 26th ACM SIGKDD, pages 976–985. ACM, 2020.

4. AAAI'20 GMM-VGAE
   
   > Binyuan Hui, Pengfei Zhu, and Qinghua Hu. Collaborative graph convolutional networks: Unsupervised learning meets semi-supervised learning. In Proceedings of the 34th AAAI, pages 4215–4222. AAAI Press, 2020.

5. ACM MM'21 AGCN
   
   > Zhihao Peng, Hui Liu, Yuheng Jia, and Junhui Hou. Attention-driven graph clustering network. In Proceedings of ACM MM, pages 935–943. ACM, 2021. 

6. NeurIPS'22 S3GC 
   
   > Fnu Devvrit, Aditya Sinha, Inderjit Dhillon, and Prateek Jain. S3GC: Scalable self-supervised graph clustering. In Advances in 36th NeurIPS, 2022.

7. WWW'22 CGC
   
   > Namyong Park, Ryan A. Rossi, Eunyee Koh, Iftikhar Ahamath Burhanuddin, Sungchul Kim, Fan Du, Nesreen K. Ahmed, and Christos Faloutsos. CGC: Contrastive graph clustering for community detection and  tracking. In Proceedings of the Web Conference, 2022.

8. WWW'22 gCooL
   
   > Bolian Li, Baoyu Jing, and Hanghang Tong. Graph communal contrastive learning. In Proceedings of The ACM Web Conference, pages 1203–1213. ACM, 2022.

9. CIKM'22 HostPool
   
   > Alexandre Duval and Fragkiskos D. Malliaros. Higher-order clustering and pooling for graph neural networks. In Proceedings of the 31st CIKM, pages 426–435. ACM, 2022.

10. IJCAI'22 AGC-DRR
    
    > Lei Gong, Sihang Zhou, Wenxuan Tu, and Xinwang Liu. Attributed graph clustering with dual redundancy reduction. In Proceedings of the 31st IJCAI, pages 3015–3021. ijcai.org, 2022.

11. IJCAI'22 FT-VGAE
    
    > Nairouz Mrabah, Mohamed Bouguessa, and Riadh Ksantini. Escaping feature twist: A variational graph auto-encoder for node clustering. In Proceedings of the 31st IJCAI, pages 3351–3357. ijcai.org, 2022.

12. AAAI'23 HSAN
    
    > Yue Liu, Xihong Yang, Sihang Zhou, Xinwang Liu, Zhen Wang, Ke Liang, Wenxuan Tu, Liang Li, Jingcan Duan, and Cancan Chen. Hard sample aware network for contrastive deep graph clustering. In Proceedings of the AAAI, 2023.

13. GAE

14. VGAE
    
    > Thomas N Kipf and Max Welling. Variational graph autoencoders.

15. ICLR'19 DGI
    
    > Petar Velickovic, William Fedus, William L. Hamilton, Pietro Lio, Yoshua Bengio, and R. Devon Hjelm. Deep graph infomax. In Proceedings of ICLR, pages 1–24, 2019.

16. TCYB'20 ARGA
    
    > Shirui Pan, Ruiqi Hu, Sai-Fu Fung, Guodong Long, Jing Jiang, and Chengqi Zhang. Learning graph embedding with adversarial training methods. IEEE Trans. on Cybern., 50(6):2475–2487, 2020.

17. ICML'20 MVGRL
    
    > Kaveh Hassani and Amir Hosein Khas Ahmadi. Contrastive multi-view representation learning on graphs. In Proceedings of ICML, volume 119, pages 4116–4126, 2020.

18. RicciCom
    
    > Chien-Chun Ni, Yu-Yao Lin, Feng Luo, and Jie Gao. Community detection on networks with ricci flow. Nature Scientific Reports, 9(9984), 2019.

19. IJCAI'21 GDCL
    
    > Han Zhao, Xu Yang, Zhenru Wang, Erkun Yang, and Cheng Deng. Graph debiased contrastive learning with joint representation clustering. In Proceedings of the 30th IJCAI, pages 3434–3440. ijcai.org, 2021.

**********

Please refer to Technical Appendix for further details.
