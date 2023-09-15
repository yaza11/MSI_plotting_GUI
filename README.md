# MSI_plotting_GUI
GUI for loading and plotting MSI data

# How to use
Get Python 3 (developed under python 3.11 but should work for older and newer versions). 
Download the zip-file. 
You may have to install the necessary packages (PyQt5, Numpy, Pandas)
Unpack the zip-file. 
Run the UI.py file.
Testfiles are included in the folder: Open the test_data.txt as the file spectra txt and hit read and any of the other txt- or csv-file as the mass list file and hit read.
Now you should see an entry in the field next to plot mz [Da]. Hit plot. 
You can also directly enter a mass or up to three seperated by semicolons. The widgets on the right of the plot area allow to modify plotting options. You can save the current options and afterwards load them with the buttons in the bottom right corner.
If you loaded in a file of masses, you can also navigate the entries with the left and right button or plot all compounds. 
You can also choose to plot all compounds in the mass list file. The pannels below the plot allow you to save the plots. You can also 
