# Emotions - Emotion contagion model for dynamical crowd path planning

## Description
Emotion contagion model for dynamical crowd path
planning. Realistic and plausible simulation of crowd path planning and crowd behaviour. We want to change the paths considering drive to use the optimal (shortest vs fastest) path of each agent and also modeling their personalities using OCEAN pesonality trait model. The emotions are contagious.

![image](https://github.com/user-attachments/assets/1d281391-d3cb-4baa-925e-7d3fb1d7da3d)


## Team members
* Andrej Sušnik (AndrejSusnik)
* Timotej Zgonik (zgontimo99)
* Ema Leila Grošelj (elgroselj)

  
## Starting Point:
Emotion Contagion Model for Dynamical Crowd Path Planning: Yunpeng Wu et al. [link](https://doi.org/10.53941/ijndi.2024.100014)

## Project stages
* First (26.11.2024)
Working simulation (simplified redo of the article), room, maybe not all parameters, collision avoidance (optional).
Initial project structure: Agent class, Environment class (room), Main class (visualization)

DONE:
calculate_distance_preference(Agent)
calculate_velocity_preference(Agent)
(is in) relationship(Agent, Agent) TODO
collective_density(Agent in Environment)


* Second (7.12.2024)
Working simulation from the article. Parallelize agent computations.

* Final (11.1.2025)
Add parameter of panic (=> agressivness) to agents (slows down evacuation)
Use historic data of agent ... (max agressivness until now)
Working simulation
Screens from simulation, and all traces


## Running the project
To run the project, you need to have Python installed on your computer. You can download it [here](https://www.python.org/downloads/).

After you have installed Python, you can clone the repository by running the following command in your terminal:
```bash
git clone git@github.com:AndrejSusnik/Emotions.git
```

Then, navigate to the project folder:
```bash

```

In the src folder create .env file with the following content:
```bash
    ENVIRONMENTS_PATH=path/to/environments
```


Create virtual environment and install the required packages:
```bash

```

To run the project, run the following command:
```bash

```