# MSI_plotting_GUI
GUI for loading and plotting MSI data

# Installation: Executable
Download the UI.zip-file (https://seafile.zfn.uni-bremen.de/f/8fd49d6f94d54e099529/?dl=1).

Unpack the zip-file and open the folder. Locate the UI.exe-file and double-click it. The GUI should pop up shortly afterwards. 

You may want to create a shortcut to the UI.exe as it can be tedious to locate

# Installation: Python
Get Python 3 (developed under python 3.11 but should work for older and newer versions). 

Download everything except the UI.zip. 

You may have to install the necessary packages (PyQt5, Numpy, Pandas)

Run the UI.py file.


# How to use
Testfiles are included in the folder: Open the test_data.txt as the file spectra txt and hit read and any of the other txt- or csv-file as the mass list file and hit read.
Now you should see an entry in the field next to plot mz [Da]. Hit plot. 
You can also directly enter a mass or up to three seperated by semicolons. The widgets on the right of the plot area allow to modify plotting options. You can save the current options and afterwards load them with the buttons in the bottom right corner.
If you loaded in a file of masses, you can also navigate the entries with the left and right button or plot all compounds. 
You can also choose to plot all compounds in the mass list file. The pannels below the plot allow you to save the plots. You can also choose to save any plot that shows up by enableing autosaving. 

If you want to create a list of compounds yourself, take a look at the example files. Mass and name columns will be detected automatically, if they have appropriate names. 
To plot multiple (2 or 3) compounds together in one plot, you have to create an index column where each compound has an index. Compounds with the same index will be plotted together.
