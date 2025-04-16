import openai
import re
import os
import json

# Nastavte svoj OpenAI API kľúč
openai.api_key = ''

def extract_similarity_score(response):
    """ Extrahuje čísla z odpovede a vráti najväčšie číslo. """
    matches = re.findall(r"\b[0-4](?:\.\d+)?|5(?:\.0)?\b", response)
    if matches:
        return max(map(float, matches))  # Vyberie najväčšie číslo
    return None

def get_chatgpt_response(messages):
    """ Získa odpoveď od ChatGPT. """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.7,
        )
        content = response.choices[0].message['content'].strip()
        messages.append({"role": "assistant", "content": content})
        return content, messages
    except Exception as e:
        return f"Error: {str(e)}", messages

def main():
    input_file = '../mojtext.txt'
    numbers_file = f"{os.path.splitext(input_file)[0]}_numbers.txt"
    results_dir = 'chatgpt_questions_results'
    os.makedirs(results_dir, exist_ok=True)
    base_results_file = os.path.join(results_dir, f"{os.path.splitext(input_file)[0]}_results.json")

    question = input("Otázka: ")

    # Kontrola spracovaných riadkov
    processed_lines = 0
    if os.path.exists(numbers_file):
        with open(numbers_file, 'r', encoding='utf-8') as num_file:
            processed_lines = sum(1 for _ in num_file)

    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            total_lines = sum(1 for _ in infile)

        results = {
            "question": question,
            "responses": []
        }

        with open(input_file, 'r', encoding='utf-8') as infile, \
                open(numbers_file, 'a', encoding='utf-8') as num_file:

            for index, line in enumerate(infile, start=1):
                if index <= processed_lines:
                    continue

                line = line.strip()
                if not line:
                    continue

                print(f"Spracovávam riadok [{index}/{total_lines}]...")

                messages = [{"role": "user", "content": f"{line}\n\nOtázka: {question}"}]
                response, messages = get_chatgpt_response(messages)
                print(f"Response od GPT: {response}")

                similarity_score = extract_similarity_score(response)
                additional_question_needed = False

                # Ak odpoveď obsahuje text aj číslo, pýtame si len číslo
                if similarity_score is None or not (
                        isinstance(response, str) and response.strip().replace('.', '', 1).isdigit()):
                    additional_question_needed = True
                    messages.append({"role": "user", "content": "Potrebujem v odpovedi len číslo. Môžeš uviesť iba číslo?"})
                    response, messages = get_chatgpt_response(messages)
                    print(f"Response od GPT (po vyžiadaní čísla): {response}")
                    similarity_score = extract_similarity_score(response)

                # Ak stále nie je číslo, nastavíme -1
                if similarity_score is None:
                    similarity_score = -1

                num_file.write(f"{similarity_score}\n")

                results["responses"].append({
                    "original_line": line,
                    "response": response,
                    "numerical_response": similarity_score,
                    "additional_question_needed": additional_question_needed
                })

                # Ukladanie ".json" po každých 10 riadkoch
                if index % 10 == 0:
                    with open(base_results_file, 'w', encoding='utf-8') as json_file:
                        json.dump(results, json_file, ensure_ascii=False, indent=4)
                    print(f"Medzičasné výsledky boli uložené do '{base_results_file}'.")

        # Uloženie konečných výsledkov
        with open(base_results_file, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)

        print(f"Spracovanie dokončené. Výsledky boli uložené do '{base_results_file}'.")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Došlo k chybe: {str(e)}")

if __name__ == "__main__":
    main()
