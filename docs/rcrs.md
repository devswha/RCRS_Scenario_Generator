---
layout: default
title: "RoboCup Rescue Simulation (RCRS)"
permalink: /rcrs/
---

# RoboCup Rescue Simulation (RCRS)

[The RoboCup Rescue Simulation (RCRS)](https://rescuesim.robocup.org/) is an official simulator used RoboCup competition, a world-class robot competition. It is a large-scale multi-agent system aim to study earthquake disaster response and support the emergency decision making by the rescue crew. In the RCRS, the simulator simulates the earthquake occurs in the city according to a specific disaster scenario. The city where the earthquake will occur can be selected from the [Open Street Map (OSM)](https://www.openstreetmap.org) because the RCRS has a program that converts maps on the OSM into maps to be used for simulation. And the earthquake disaster scenario in the simulator consists of time steps, buildings, roads, and people.


![Robocup Rescue Simulation]({{ site.baseurl }}/assets/images/RCRS.png)


In a disaster scenario, a fire starts at certain buildings, and some randomly selected buildings collapse and create debris. A fire cause injury and the debris cause the interference of movement of the rescue team. And scenario progress a unit of time called time step and the time step progresses, the fire spreads and people move. In the simulation, there are two groups of people: rescue teams and civilians. The rescue team consists of firefighters, police and emergency teams. The firefighters are responsible for firefighting, police removing debris from buildings, and emergency teams for transporting the injured civilians to shelters. Each rescue crew can communicate within a certain range, and according to the competition participants' disaster response plans and policies, rescue crew are moved. In RCRS, there are civilians in the disaster zone random location. And as the disaster progresses, the building collapses or fires cause injured civilians. Therefore, we predict the location of hidden injured civilians in RCRS. And the civilians have information called Health Point (HP) that shows how much they are injured status. Due to the disaster situation in simulators, likes fires spread or buildings collapse, civilians' HP going to lower. We set the HP threshold which determines the civilians is injured or not. If one civilian' HP lower than this threshold, we determine this civilian is injured.


[The RoboCup Rescue Simulation (RCRS) GitHub Page](https://github.com/roborescue)
