def process_numbers(input_file, output_file):
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()

        results = []
        for line in lines:
            try:
                number = float(line.strip())  # Attempts to convert the line to a floating point number
                result = number * 5  # Multiplies the number by 5
                results.append(result)
            except ValueError:
                print(f"Skipping invalid number line: {line.strip()}")  # Skips lines that cannot be converted to float

        with open(output_file, 'w') as file:
            for result in results:
                file.write(f"{result}\n")  # Writes each result to a new line

        print(f"Processed {len(results)} numbers. Results saved to {output_file}.")
    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
process_numbers('nlpcloud_similarity_mojtext.txt', 'nlpcloud_similarity_mojtext_correct.txt')
