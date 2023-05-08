Creating the cvs:
The primary function of the Cu_reader.py script is to read all the calculations and choose those that give a minimum
energy for a given concentration for each facet, creating a csv dataset for ease of import for other scripts.
It can also plot (matplotlib) the dft adsorption energies via versvs the concentration or theta.

Inorder to recognise folders containing the calculations file they must follow this regex pattern, were {ad_str} will be
replaced with the keyword argument value from -Ad:
\d{3}(-\dX\d)?_{ad_str}_\d+per\d+\w?\Z

it has the following commandline keyword arguments:

-dir
Path to the folder containing the folder with the calculations.

-f
A keyword used if only a set amount of facets are of interest, if not stated the script will take every facet that fits
its regex.

-plot
A bool that denotes if the adsorptions plots are desired.

-csv
A bool that states if the data should be written into a csv file. 2 csv files will be created one for the adsorbate
data and one containing the clean slab data

-Ad
Denotes what adsorbates should be considered, will default to Cl.

If neither -plot or -csv then there will be no output.
An example of use could be:
python Cu_reader.py -dir ./si_for_cl/. -plot -csv


Plotting the gibbs integrals:
The script gibbs_integral_plotter.py is used to plot the gibbs integrals, as seen in figure 4A of the article.
The script can plot both in plotly, creating an interactive html and in matplotlib.

It has the following commandline positional arguments:

Facet_csv
Path to csv containing the minima energies, as created by Cu_reader.py

It has the following commandline keyword arguments:

-U
A narg, were each value given should be a potential value to plot the gibbs energy with. The potential is vs. SHE

-plt
A bool that will get the script to plot with matplotlib, as were used for publication.

An example of use could be:
python .\gibbs_integral_plotter.py .\Cl_minima_energies.csv -U 0 0.848 -plt


Creating figure 4B, 4C and the wulff structures.
potential_plotter.py creates everything that is plotted against the potential.
The script can plot both in plotly, creating an interactive html and in matplotlib.

It has the following commandline positional arguments:

Facet_csv
Path to csv containing the minima energies, as created by Cu_reader.py

It has the following commandline keyword arguments:

-wulff
A bool that dictates that the wullf constructions will be created as png files.

-plotly
A bool that if called will plot in plotly instead of plt.

An example of use could be:
python .\potential_plotter.py .\Cl_minima_energies.csv -wulff