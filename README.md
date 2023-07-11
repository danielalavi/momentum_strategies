# momentum_strategies

This repository contains code and data used for my bachelor's thesis that analyzed momentum strategies on the FTSE 100 from 1996 to 2021. 

# Methodology
The methodology follows the paper by Jegadeesh and Titman on momentum strategies. Different evaluation and holding periods are selected and analyzed for their return and volatility. 
Additionally, returns are decomposed by classical risk factors introduced by Fama and French. 

# Data
The data is sourced from Refinitiv Eikon, for which reason I cannot include it in this repository. The data cleaning process, however, was extremely rigorous compared to other studies in order to ensure reproducibility of results in real time. All delisted companies were included until the time of delisting in order to avoid survivorship bias which is especially prone to affecting results of momentum strategies. 

# Conclusion
Significant outperformance of winner stocks can still be observed for the sample period. A profitable trading strategy is still hard to implement due to trading costs. 
