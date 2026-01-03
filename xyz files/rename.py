import os

# Define the directory path
directory = '.'

# Iterate over all files in the specified directory
for filename in os.listdir(directory):
    # Check if the file ends with the double extension
    if filename.endswith(".niflosic.xyz.xyz"):
        # Construct the new filename by replacing the suffix
        new_filename = filename.replace(".niflosic.xyz.xyz", ".niflosic.xyz")
        
        # Get full paths for the old and new filenames
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
