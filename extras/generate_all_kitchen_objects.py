import json

def read_json_file(file_path):
    """Read JSON data from a file and return the parsed data."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def combine_data(file1_path, file2_path):
    """Combine data from two JSON files into a single dictionary."""
    data1 = read_json_file(file1_path)
    data2 = read_json_file(file2_path)
    
    combined_data = {}
    combined_data.update(data1)
    combined_data.update(data2)
    
    return combined_data

def process_combined_data(combined_data):
    """Perform operations on the combined data."""
    all_objects = set()
    
    for key in combined_data:
        all_objects.update(combined_data[key])
    
    # Add additional objects
    all_objects.update(['Human', 'Fork'])
    
    # Convert set to a sorted list of object names
    sorted_objects = sorted(all_objects)
    
    # Create formatted string
    formatted_string = '. '.join(sorted_objects) + '.'
    
    # Replace 'spatula' with 'Fork'
    formatted_string = formatted_string.replace('spatula', 'Fork')
    
    return formatted_string

def save_formatted_data(formatted_data, output_file_path):
    """Save formatted data to a text file."""
    with open(output_file_path, 'w') as file:
        file.write(formatted_data)
    print(f"Formatted data saved to '{output_file_path}'.")

if __name__ == "__main__":
    # Paths to input JSON files
    file1_path = '/data/aryan/Seekg/nesca-pytorch/datasets/detected_objects_50salads.json'
    file2_path = '/data/aryan/Seekg/nesca-pytorch/datasets/detected_objects_breakfast.json'
    
    # Path to output text file
    output_file_path = 'formatted_objects.txt'
    
    # Combine data from two JSON files
    combined_data = combine_data(file1_path, file2_path)
    
    # Process combined data
    formatted_data = process_combined_data(combined_data)
    
    # Save formatted data to a text file
    save_formatted_data(formatted_data, output_file_path)
