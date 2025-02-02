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
Emotion Contagion Model for Dynamical Crowd Path Planning: Yunpeng Wu et al. available at [https://doi.org/10.53941/ijndi.2024.100014](https://doi.org/10.53941/ijndi.2024.100014).

## Project stages
* First (26.11.2024)

    - Working simulation (simplified redo of the article), room, maybe not all parameters, collision avoidance (optional).
    - Initial project structure: Agent class, Environment class (room), Main class (visualization)

    - REALIZED: visualization, basic structure, clustering, contagion of preferences




* Second (7.12.2024)

    - Working simulation from the article.
    - Parallelize agent computations.

* Final (11.1.2025)

    - Add parameter of panic to agents (slows down evacuation)
    <!-- - Use historic data of agent (max agressivness until now) -->
    - Working simulation
    - Screens from simulation, and all traces
    - Try out different clustering methods

* Postfinal (28.2.2025)
    - Collision avoidance: in case of conflict (intersecting paths/lines inside one time step), a random agent gets to procede, other(s) have to stay still
    - Increase agent count to 150-200
    - Visualize panic parameter (size of the agent)
    - Try out different clustering methods - debug


## Realization of goals
We have implemented a simulation workflow where we can define the environment in the text format and then run the simulations. We tried out three different clustering algorithms, but the change of clustering algorithm appears to have no visible influence on the result. The introduction of panic parameter was a fruitful decision, as panic like behavior can be seen in the paths of agents. We did not manage to incorporate the historic data of the agent into the model.


## Running the project
To run the project, you need to have Python installed on your computer. You can download it [here](https://www.python.org/downloads/).

After you have installed Python, you can clone the repository by running the following command in your terminal:
```bash
git clone git@github.com:AndrejSusnik/Emotions.git
```

Then, navigate to the project folder:
```bash
cd Emotions
```

In the src folder create .env file with the following content:
```bash
ENVIRONMENTS_PATH=path/to/environments
```

if you ar running the project from the root of the project only run
```bash
mv example.env .env
```


Install the required packages:
```bash
python3 -m pip install -r requirements.txt
```

To run the project, run the following command:
```bash
python3 src/main.py
```