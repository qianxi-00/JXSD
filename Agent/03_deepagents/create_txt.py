print("Python script started: Creating a txt file...")

# Create a txt file
with open("example.txt", "w", encoding="utf-8") as file:
    file.write("This is an example text file created by a Python script.\n")
    file.write("It was executed successfully!")

print("Text file 'example.txt' has been created in the current directory.")
print("File content:")
with open("example.txt", "r", encoding="utf-8") as file:
    print(file.read())