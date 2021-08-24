# 基于CUDA通用GPU计算平台SM4-CTR算法并行实现与优化

原本只用于图像计算的硬件GPU，在编程模型CUDA发布后就成为通用的、普及化的算力资源。本代码基于通用的计算机平台，利用本地GPU进行CTR工作模式下SM4算法高速加解密的并行实现。实验表明，本方案提出的SM4-CTR并行加解密方案能够有效提高SM4算法的运行效率，在通用的计算机平台上，能够达到40倍加速比，加解密速率达到了14.192Gbps。线程块划分对GPU并行加速效果的影响，最优线程块大小为128到512，且必须为32的整倍数。与其他团队的优化SM4方案进行对比，包括传统工作模式下利用CPU、GPU优化的方案和利用软件快速实现的方案，对比结果显示即便之前团队的方案运行的平台硬件条件好于本文实验环境，文中提出的方案运行速率依然能做到大幅领先。
