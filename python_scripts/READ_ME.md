# Python Scripts
Author: **Harrison LaBollita**

Department of Mathematics and Physics, Piedmont College, Demorest, GA

Email: hlabollita0219@lions.piedmont.edu

This is the accompanying read me documentation for all of the python codes found in the python_scripts folder. Contact me at the provided email address or follow this [link](https://github.com/CompPhysics/MachineLearning) for more information on Machine Learning.

**NOTE**: The data files: "BetaScint2DEnergy.csv" and "BetaScint2DEnergy2Electron.csv" are neeeded in addition to run any of these scripts.

1. *neural_nets*

Dependencies: ProcessingData and setup_electron_densities

This folder contains the neural networks that were created:
  * *one_e_cnn_python.py*

    CNN for single electron cases to predict origin of electron

  * *two_e_cnn_python.py*

    CNN for distinguishing one and two electron events

2. *extras*

Contains old files

3. *ProcessingData.py*

This file contains some of the functions that I use over and over throughout all of the other files. It contains:

   * get_data

     load Sean Liddick data into input data (grids) and output data (outputs)

   * max_energy_matrix

     find the highest energy pixel on the grid and return its row and column and energy value.

   * distance_formula

     compute the distance between two points

   * find_starting_pixels

     finds the exact pixel that the electron started in.

   * print_stats

     used in electron_densities to print the information that was calculated.

4. *electron_densities.py*

Dependencies: ProcessingData and setup_electron_densities

This script finds the highest energy pixel and then checks to see if the electron for that event started in that pixel. It then plots a two dimensional histogram of all the electrons starting the 100 possible starting pixels.
At the end of the script, all of the stats will be printed out, these include the number of electrons that started inside of the pixel with the highest energy as well as the total number of events that had that pixel as the one with the highest energy. We find that this number is about 70% of the time the electron started in the pixel with the highest energy.

5. *setup_electron_densities.py*

Dependencies: ProcessingData

This script contains functions that the main script electron_densities calls.

6. *sort_easy_hard.py*

Dependencies: ProcessingData

This script sorts the two electron data into easy and hard cases. We deemed easy cases that had two clusters, where it was obvious that there were two electrons here. A hard case was one where there were one or three or more clusters. We found that out of the entire two electron data set only 15% of it was "hard" cases.


7. *one_electron_origins.py*

Dependencies: ProcessingData

Calculates how many electrons started in the pixel with the highest energy and how many times the electron started in one of the neighboring pixels.

8. *two_electron_origins.py*

Dependencies: ProcessingData

Calculates how many times the electron started in the pixel with the highest energy or the second highest energy and how many times the electron started in one of the neighbors of the pixel with the highest energy or the second highest energy.
