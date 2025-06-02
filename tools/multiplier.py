def process_numbers(input_file, output_file):
    try:
        with open(input_file, 'r') as file:
            lines = file.readlines()

        results = []
        for line in lines:
            try:
                number = float(line.strip())
                result = number * 5
                results.append(result)
            except ValueError:
                print(f"Skipping invalid number line: {line.strip()}")

        with open(output_file, 'w') as file:
            for result in results:
                file.write(f"{result}\n")

        print(f"Processed {len(results)} numbers. Results saved to {output_file}.")
    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

process_numbers('nlpcloud_similarity_mojtext.txt', 'nlpcloud_similarity_mojtext_correct.txt')
