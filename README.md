# False Data Injection Attacks in Internet of Things and Deep Learning enabled Predictive Analytics

# Introduction
Industry 4.0 is the latest industrial revolution primarily merging automation with advanced manufacturing to reduce direct human effort and resources. Predictive maintenance (PdM) is an industry 4.0 solution, which facilitates predicting faults in a component or a system powered by state-of-the-art machine learning (ML) algorithms (especially deep learning algorithms) and the Internet-of-Things (IoT) sensors. However, IoT sensors and deep learning (DL) algorithms, both are known for their vulnerabilities to cyber-attacks. In the context of PdM systems, such attacks can have catastrophic consequences as they are hard to detect due to the nature of the attack. To date, the majority of the published literature focuses on the accuracy of the IoT and DL enabled PdM systems and often ignores the effect of such attacks. In this paper, we demonstrate the effect of IoT sensor attacks (in the form of false data injection attack) on a PdM system. At first, we use three state-of-the-art DL algorithms, specifically, Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Convolutional Neural Network (CNN) for predicting the Remaining Useful Life (RUL) of a turbofan engine using NASA's C-MAPSS dataset. Our obtained results show that the GRU-based PdM model outperforms some of the recent literature on RUL prediction using the C-MAPSS dataset. Afterward, we model and apply two different types of false data injection attacks (FDIA), specifically, continuous and interim FDIAs on turbofan engine sensor data and evaluate their impact on CNN, LSTM, and GRU-based PdM systems. Our results demonstrate that attacks on even a small number of IoT sensors can strongly defect the RUL prediction in all cases. However, the GRU-based PdM model performs better in terms of accuracy and FDIA resiliency.

# Dataset
To evaluate the performance of the CNN, LSTM, and GRU DL algorithms, we use a well-known dataset, NASA's turbofan engine degradation simulation dataset C-MAPSS (Commercial Modular Aero-Propulsion System Simulation). This dataset includes 21 sensor data with different number of operating conditions and fault conditions. In this dataset, there are four sub-datasets (FD001-04). Every subset has training data and test data. The test data has run to failure data from several engines of the same type. Each row in test data is a time cycle which can be defined as an hour of operation. A time cycle has 26 columns where the 1st column represents engine ID, and the 2nd column represents the current operational cycle number. The columns from 3 to 5 represent the three operational settings and columns from 6-26 represent the 21 sensor values. The time series data terminates only when a fault is encountered.

# DL algorithms for RUL prediction
For this work, we utilize LSTM, GRU, and CNN algorithms and compare their performance to an FDIA attack. The Table I gives summary of all the hyperparameters used. Table 2 gives summary of architectures of DL algorithms employed in this work and also RMSE comparision of LSTM, CNN and GRU.

From Table II The notation GRU(100,100,100) lh(80) refers to a network that has 100 nodes in the hidden layers of the first GRU layer, 100 nodes in the hidden layers of the second GRU layer, 100 nodes in the hidden layers of the third GRU layer, and a sequence length of 80. In the end, there is a 1-dimensional output layer.

From Fig. 3 and Table II it is evident that the DL algorithm GRU(100, 100, 100) with a sequence length 80 has the least RMSE of 7.26. It means that GRU is very accurate in predicting accurate RUL for this dataset.

<img src="https://github.com/dependable-cps/False-Data-Injection-Attacks-in-Internet-of-Things-and-Deep-Learning-enabled-Predictive-Analytics/blob/master/images/Hyper.PNG" height="300" width="480">

<img src="https://github.com/dependable-cps/False-Data-Injection-Attacks-in-Internet-of-Things-and-Deep-Learning-enabled-Predictive-Analytics/blob/master/images/Comp.PNG" height="250" width="800">

# FDIA signature: 
To model the FDIA on sensors, we add a vicious vector to the original vector, which modifies the sensor output by a very small margin (0.01% to 0.05%) for random FDIA and 0.02% for biased FDIA. Here, random FDIA means the noise added to the sensor output has a range (0.01% to 0.05%). Whereas, biased FDIA has a constant amount of noise added to the sensor output. Fig. 1 shows the comparison between the original and FDIA attacked output signal of sensor 2 for engine ID 3 for continuous FDIA. In continuous FDIA, we attack the sensor output from time cycles 130 to the end of life of the engine. In the case of interim FDIA as shown in Fig. 2, the attack duration is only for 20 time cycles (130 to 150 time cycles). Note, in the constrained attack the adversary has limited access to sensors. As shown in Figure 1 and 2, the attack signature is very similar to the original signal, making it stealthy and harder to detect even with common defense mechanisms in place.

<img src="https://github.com/dependable-cps/False-Data-Injection-Attacks-in-Internet-of-Things-and-Deep-Learning-enabled-Predictive-Analytics/blob/master/images/ContinuousSignature.PNG" height="210" width="380">

<img src="https://github.com/dependable-cps/False-Data-Injection-Attacks-in-Internet-of-Things-and-Deep-Learning-enabled-Predictive-Analytics/blob/master/images/InterimSignature.PNG" height="200" width="380">

# Impact of FDIA on a PdM system
The average degradation point of the engine is considered as 130 for the FD001 dataset, and we assume that the Engine Health Monitoring (EHM) system of the aircraft sends 20 time cycles of data to the ground at a time. The train and test dataset have 21 sensor data. The FDIA can be performed on 21 sensors, but to make the attack more realistic, we perform FDIA on only 3 sensors (specifically, T24, T50, and P30). In FDIA continuous scenario, the attacker has initiated the attacks after 130 time cycles (one time cycle is equivalent of one flight hour), and the attack duration is until end of life of the engine. In FDIA interim scenario, the attacker has initiated the attacks after 130 time cycles, and the attack duration is 20 hours (20 time cycles). Since the attack is initiated after 130 time cycles, we only consider the engines which have data for more than 130 cycles which gives us 37 engines in the FD001 dataset.

It is evident from Fig. 4 and 5 that LSTM, GRU, and CNN are greatly affected by the continuous and interim FDI attack. In the case of random and biased FDIA, random FDIA showed a considerable impact on all PdM models.

<img src="https://github.com/dependable-cps/False-Data-Injection-Attacks-in-Internet-of-Things-and-Deep-Learning-enabled-Predictive-Analytics/blob/master/images/AttackScenario.PNG" height="500" width="800">

# Piece-wise RUL prediction
To show the impact of FDIA attacks on a specific engine data, we apply the piece-wise RUL prediction. The piece-wise RUL prediction gives a better visual representation of degradation in an aircraft engine. Figure 6(a) shows an example of an engine data from the dataset of 100 engines, and depicts the predicted RUL using GRU at each time step of that engine data. From Figure 6(a), it is evident that as the time series approaches the end of life, the predicted RUL (red line) is close to the true RUL (blue dashes), because the DL model has more time series data to accurately predict the RUL. Figure 6 and 7 gives piece wise RUL prediction after both continuous and interim FDIA.

<img src="https://github.com/dependable-cps/False-Data-Injection-Attacks-in-Internet-of-Things-and-Deep-Learning-enabled-Predictive-Analytics/blob/master/images/PieceWise.PNG" height="500" width="800">

# Citation
If this is useful for your work, please cite our <a href="https://ieeexplore.ieee.org/abstract/document/9110395"> NOMS paper</a>, and/or the <a href="https://arxiv.org/pdf/1910.01716.pdf">arXiv paper (extended version)</a>:<br>
<div class="highlight highlight-text-bibtex"><pre>
@INPROCEEDINGS{hoque2020fdia,
  author={G. R. {Mode} and P. {Calyam} and K. A. {Hoque}},
  booktitle={NOMS 2020 - 2020 IEEE/IFIP Network Operations and Management Symposium}, 
  title={Impact of False Data Injection Attacks on Deep Learning Enabled Predictive Analytics}, 
  year={2020},
  volume={},
  number={},
  pages={1-7}
  }
</pre></div>

<div class="highlight highlight-text-bibtex"><pre>
@article{mode2019false,
title={False Data Injection Attacks in Internet of Things and Deep Learning enabled Predictive Analytics},
  author={Mode, Gautam Raj and Calyam, Prasad and Hoque, Khaza Anuarul},
  journal={arXiv preprint arXiv:1910.01716},
  year={2019}
}
</pre></div>
