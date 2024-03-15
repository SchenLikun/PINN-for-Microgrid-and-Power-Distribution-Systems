# Introduction
This project aims for solving the ODEs in mircrogird, especially for forward (simulation) and inverse (parameters estimation) calculation of synchronous- and inverter-based DERs.

# Env. Info.
Python Version 3.11 backends PyTorch 2.0.1 + CU117 on NVIDIA A30 × 4

# Basic principles plz refer to our papers

>LATEX CITATION:
>
>@unpublished{LikunSchenPESGM2024,  
  author = {Chen, Likun and Dong, Xuzhu and Wang, Yifan and Sun, Wei and Wang, Bo and Harrison, Gareth},  
  title  = {Physics-Informed Neural Network for Microgrid Forward/Inverse Ordinary Differential Equations},  
  note   = {IEEE PES General Meeting},  
  year   = {2024},  
>}
>
>@unpublished{LikunSchenTPS2024,  
  author = {Chen, Likun and Dong, Xuzhu and Wang, Yifan and Sun, Wei and Harrison, Gareth},  
  title  = {Improved PINN-Based Parameter Estimation for Optimizing Microgrid DER Analysis-Part I},  
  note   = {Submitted to IEEE Trans. on Power Systems},  
  year   = {2024},  
>}
  
# how to use
The example codes for both sync- and inverter-based DERs are './PINN-for-Synchronous/SYNC_MAIN.py'[../PINN-for-Synchronous/SYNC_MAIN.py](url) and './PINN-for-inverter-control-loop/PID_MAIN.py'.

Before running, plz remember to install Torch and CUDA correctly.
