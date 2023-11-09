Alternating Local Enumeration (TnALE):
Solving Tensor Network Structure Search with Fewer Evaluations (ICML, 2023) [https://proceedings.mlr.press/v202/li23ar/li23ar.pdf]
===================================

Introduction
-------------------------------
This repository is the implementation of TnALE under the ring constraint.



Requirements
----------------------
 * Python 3.7.3<br/>
 
 * Tensorflow 1.13.1
 
Usage
---------------------
First, you need to start agents with

     CUDA_VISIBLE_DEVICES=0 python agent.py 0
     
The last 0 stands for the id of the agent. You can spawn multiple agents with each one using one GPU by modifying the visible device id. <br/>

Then start the main script by

     python TNALE_TR.py 'data.npz' 2 1 1 2
     
The argvs stand for the name of data, the rank-related radius in the initial phase, the rank-related radius in the main phase, the switch that decides whether or not to include the initial phase and the $L_{0}$ in the paper, respectively. Here we provide a demo of learning the low-dimensional TR format representation of a tensor. The running details will be saved in a `.log` file.

Acknowledgment
-------------------------
 * The code is modified based on the [TNGA](https://github.com/minogame/icml2020-TNGA). Thanks for their great efforts.
