import json

# Read the notebook
with open('EDA_Hector.ipynb', 'r') as f:
    notebook = json.load(f)

# Update the kernel metadata
notebook['metadata'] = {
    "kernelspec": {
        "display_name": "Python 3.14.2 (suicide_detection)",
        "language": "python",
        "name": "suicide_detection"
    },
    "language_info": {
        "name": "python",
        "version": "3.14.2",
        "mimetype": "text/x-python",
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py"
    }
}

# Write back the notebook
with open('EDA_Hector.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Notebook kernel updated successfully!")
