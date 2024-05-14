new_size = 100
input_file = "s1.txt"
output_file = f"s1_{new_size}.txt"

def replace_commas(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            data = f.readlines()
        
        # modified_data = [line.replace(',', ' ') for line in data]
        
        with open(output_file, 'w') as f:
            f.writelines(data[:new_size])
        
        print("File successfully modified and saved as", output_file)
    
    except FileNotFoundError:
        print("File not found.")


replace_commas(input_file, output_file)
