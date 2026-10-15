# Problem Statement

## Introduction   
The aim of this project is to develop a method for designing optimal bicycle networks that is usable for real-world policy-makers and urban planners.

"Optimal", because we want the bicycle network to be optimised for demand; bicycle paths should be placed where the users most need them.\* The network should also be optimised for cost, so that public spending can be kept to a minimum. In short, the produced bicycle networks should have an optimal balance between two (or more?) criteria; this requires a multi-objective optimisation approach. Since this is merely a *method,* users of the method should preferably be able to plug in their own optimisation criteria.

"Usable", because there is a need to close the gap between theoretical research on how to solve network design problems and the practical applicability in a real-world context. Some papers propose mathematical models that indeed solves the bicycle network design problem; often, these papers readily state that authorities can use the proposed solution in transportation planning. But it is not clear how policy-makers and planners are supposed to apply the often very complex and computationally demanding theoretical models to their specific city and use-case.

*We must take into account that this approach risks favouring affluent neighbourhoods where bicycling is already big.*   

## Context and state-of-the-art
*WIP*

How can we design optimal and connected bicycle networks that are optimised for demand and travel time - from scratch?

This problem is necessary to think about, because 1) cycling is a cheap and healthy mode of transportation that can help reduce emissions and , and 2) most cities have very little or no existing bicycling infrastructure.

The Network Design Problem (NDP) has been studied for decades… NP-hard…

## Idea, scope, methods and deliverables   
This project will develop and evaluate a method for designing optimal bicycle networks using Multi-Objective Optimisation (MOO) and Genetic Algorithms (GA) and make this method usable for planners and policy-makers with open-source code and possibly a web app or GUI.

The project scope includes:

- Street network data processing on a chosen case city, using Python   
- Choosing seed points   
- Defining optimisation criteria   
- Calculating demand for every seed point pair   
- Defining in pseudocode and implementing in Python a genetic algorithm, which builds a bicycle network that connects the selected seed points   
- Applying the algorithm to the case city   
- Visualising the produced bicycle networks on a background map of the case city   
- Possibly some front-end programming of a GUI, where planners and policy-makers can enter their own city and get an optimal bicycle network out; both as visualisation and as downloadable geo data   

Deliverables will be:

- Thesis report incl. visualisations   
- Code on Github   
- Possibly a simple web app or GUI  