# Parallel Implementation of SM4-CTR Algorithm based on General Computing Platform

Data transmission security issues have become increasingly prominent with the rapid development of big data, cloud computing, and 5G communication technologies. The design and efficient implementation of cryptographic algorithms have become particularly important. Domestic cryptographic algorithms that can run at high speeds have become the key to protecting national security. At the same time, the hardware GPU, which was initially used for image computing, became a universal and popular computing power resource after the release of the programming model CUDA. Based on a general computer platform, this paper proposes a parallel implementation and optimization scheme for using its local GPU to perform high-speed encryption and decryption of SM4 algorithm in CTR mode. Experiments show that the SM4-CTR parallel encryption and decryption scheme proposed in this paper can effectively improve the operating efficiency of the SM4 algorithm. On a general computer platform, it can achieve 40 times the speedup, and the encryption and decryption rate has reached 14.192Gbps. The experiment also analyzed the effect of thread block division on the GPU parallel acceleration effect. The optimal thread block size is 128 to 512, and must be an integral multiple of 32. Finally, based on the results of the experiments in this article, compare the optimized SM4 solutions of other teams, including the solutions optimized by CPU and GPU in the traditional working mode and the solutions quickly implemented by software. The comparison results show that even other team’s solution runs on the better platform hardware conditions, the operating speed of the scheme proposed in this article can still achieve a significant lead. Therefore, the solution in this paper has a broader application platform while improving security and computing speed. It will play a huge role in the security protection of big data and personal data in real life.

MIT license

Programmer: Yiming Hu、Chenhao Ji

Email: zjy@besti.edu.cn

Accepted by Journal of Cryptologic Reseatch




# 基于CUDA通用GPU计算平台SM4-CTR算法并行实现与优化

原本只用于图像计算的硬件GPU，在编程模型CUDA发布后就成为通用的、普及化的算力资源。本代码基于通用的计算机平台，利用本地GPU进行CTR工作模式下SM4算法高速加解密的并行实现。实验表明，本方案提出的SM4-CTR并行加解密方案能够有效提高SM4算法的运行效率，在通用的计算机平台上，能够达到40倍加速比，加解密速率达到了14.192Gbps。线程块划分对GPU并行加速效果的影响，最优线程块大小为128到512，且必须为32的整倍数。与其他团队的优化SM4方案进行对比，包括传统工作模式下利用CPU、GPU优化的方案和利用软件快速实现的方案，对比结果显示即便之前团队的方案运行的平台硬件条件好于本文实验环境，文中提出的方案运行速率依然能做到大幅领先。


代码遵循MIT协议。

代码作者：胡一鸣、吉晨昊

联系方式：zjy@besti.edu.cn

本研究已被《密码学报》录取

北京电子科技学院CSP实验室
