Alternating Local Enumeration (TnALE):
Solving Tensor Network Structure Search with Fewer Evaluations (ICML, 2023)
===================================

Intro
-------------------------------
This repository is the implementation of TnALE under the ring constraint ((https://proceedings.mlr.press/v202/li23ar/li23ar.pdf)).



Requirements
----------------------
 * Python 3.7.3<br/>
 
 * Tensorflow 1.13.1
 
Usage
---------------------
First you need to start agents with

     CUDA_VISIBLE_DEVICES=0 python agent.py 0
     
The last 0 stands for the id of the agent. You can spawn multiply agents with each one using one gpu by modifying the visible device id. <br/>

Then start the main script by

    python TNALE_TR.py ‘data.npz’ 2 1 1 2

The argvs stands for the name of data, the rank-related radius in Init Phase, rank-related radius in Main Iteration, Whether or not to use Initialization with objective estimation, Times of Initialization respectively. Here we provide a demo of learning the low-dimensional representation TR format of the tensor. The running details of the algorithm will be saved in a `.log` file.

Acknowledgement
-------------------------
 * The code is modified based on the [TNGA](https://github.com/minogame/icml2020-TNGA). Thanks for their great efforts.
