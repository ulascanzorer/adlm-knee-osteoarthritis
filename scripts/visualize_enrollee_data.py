import pandas as pd
import dtale
import os

# This will work regardless of the directory we are in, since the
# path is relative to the location of the current python script.
current_dir = os.path.dirname(__file__)

data = pd.read_csv(f"{current_dir}/Enrollees.txt", sep="|") # You need the Enrollees.txt file from "OAICompleteData_ASCII" in the same directory as this file for this to work.

dtale.show(data, subprocess=False)  # This shows the dataframe in a web UI, whose link you can find in the output of the program.