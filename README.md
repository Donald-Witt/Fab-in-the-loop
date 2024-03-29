# Fab-in-the-loop
Code for Fab-in-the-loop Reinforcement Learning

The optimization of silicon photonic components is a key challenge. The performance of simulated
devices is significantly different than their measured performance due to the fabrication
process. Fabrication effects such as sidewall angle and roughness are difficult to account
for in conventional simulations; to do so, the simulation mesh must be made finer, 
resulting in dramatically increased simulation time. I came up with the idea of using machine
learning to optimize these devices for a fabrication process. I call this approach fab-in-the-loop
reinforcement learning. The key idea is to have the algorithm propose improved designs based on the
measured results of previous fabrication runs. A new generation of improved devices based on these 
results is fabricated and measured and the best performing designs are found. The algorithm can be repeated until 
a device optimized to desired performance is generated. 

If you find this useful, please cite our APL photonics paper “Reinforcement learning for photonic component design“ https://doi.org/10.1063/5.0159928

Copyright ©Donald Witt 2023. All rights reserved.
