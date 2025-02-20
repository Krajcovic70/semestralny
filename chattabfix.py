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
    input_file = 'temp.txt'
    numbers_file = f"{os.path.splitext(input_file)[0]}_numbers.txt"
    results_dir = 'chatgpt_questions_results'
    os.makedirs(results_dir, exist_ok=True)
    base_results_file = os.path.join(results_dir, f"{os.path.splitext(input_file)[0]}_results.json")

    question = "Aká je podobnosť medzi týmito dvoma vetami na stupnici od 0 do 5?"

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

                # **Rozdelenie riadku na tri časti podľa tabulátora (`\t`)**
                parts = line.split('\t')

                if len(parts) < 3:
                    print(f"Preskakujem neplatný riadok [{index}]: {line}")
                    continue

                # Prvé číslo (ignorujeme pri porovnávaní, ale uložíme)
                original_score = parts[0].strip()

                # Dve vety na porovnanie
                sentence1 = parts[1].strip()
                sentence2 = parts[2].strip()

                if not sentence1 or not sentence2:
                    print(f"Preskakujem riadok [{index}]: {line} (prázdne vety)")
                    continue

                print(f"Porovnávam: '{sentence1}' VS '{sentence2}'")

                # Pripravíme správu pre GPT
                messages = [{"role": "user", "content": f"What is semantics text similarity of those two sentences? They are in slovak language and please respond in number float or int on scale 0-5.\n\n1. {sentence1}\n2. {sentence2}"}]
                response, messages = get_chatgpt_response(messages)
                print(f"Response od GPT: {response}")

                similarity_score = extract_similarity_score(response)
                additional_question_needed = False

                if similarity_score is None or not response.strip().replace(".", "").isdigit():
                    additional_question_needed = True
                    messages.append({"role": "user", "content": "Potrebujem v odpovedi len číslo. Môžeš uviesť iba číslo?"})
                    response, messages = get_chatgpt_response(messages)
                    print(f"Response od GPT (po vyžiadaní čísla): {response}")
                    similarity_score = extract_similarity_score(response)

                if similarity_score is None:
                    similarity_score = -1

                num_file.write(f"{original_score}\t{similarity_score}\n")

                results["responses"].append({
                    "original_score": original_score,
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "response": response,
                    "numerical_response": similarity_score,
                    "additional_question_needed": additional_question_needed
                })

                if index % 10 == 0:
                    with open(base_results_file, 'w', encoding='utf-8') as json_file:
                        json.dump(results, json_file, ensure_ascii=False, indent=4)
                    print(f"Medzičasné výsledky boli uložené do '{base_results_file}'.")

        with open(base_results_file, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)

        print(f"Spracovanie dokončené. Výsledky boli uložené do '{base_results_file}'.")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Došlo k chybe: {str(e)}")

if __name__ == "__main__":
    main()
