import pandas as pd
from scipy.stats import pearsonr

# Subory
original_scores_path = 'sick_sk_slovak_gpt.txt'
computed_scores_path = 'sick_sk_slovak_gpt_numbers.txt'
final_file = 'final.txt'  # Súbor pre uloženie finálnych výsledkov
redo_file = 'redo.txt'  # Súbor pre znovuspracovanie neplatných riadkov

def compute_pearson_correlation(original_scores_path, computed_scores_path, question):
    """
    Vypočíta Pearsonov korelačný koeficient a uloží výsledky do final.txt.
    Zároveň spracuje neplatné hodnoty (-1) a uloží ich do redo.txt.
    """
    try:
        # Načítanie originálnych skôre
        original_data = pd.read_csv(original_scores_path, sep='\t', header=None)
        original_scores = original_data[0].replace(',', '.', regex=True).astype(float)

        # Načítanie vypočítaných skôre
        computed_data = pd.read_csv(computed_scores_path, header=None)
        computed_scores = computed_data[0].astype(float)

        # Kontrola, či majú obe sady rovnakú dĺžku
        if len(original_scores) != len(computed_scores):
            result = "Error: Počet originálnych a vypočítaných skôre sa nezhoduje."
        else:
            # Identifikácia riadkov s neplatnými hodnotami (-1)
            invalid_indices = computed_scores[computed_scores == -1].index

            # Uloženie neplatných riadkov do redo.txt
            with open(redo_file, 'w', encoding='utf-8') as redo_out:
                for idx in invalid_indices:
                    full_sentence = '\t'.join(map(str, original_data.iloc[idx].values))
                    redo_out.write(f"{full_sentence}\n")

            # Odstránenie neplatných hodnôt z oboch sád
            valid_original_scores = original_scores.drop(invalid_indices)
            valid_computed_scores = computed_scores.drop(invalid_indices)

            # Kontrola, či zostali nejaké platné dáta
            if len(valid_original_scores) == 0 or len(valid_computed_scores) == 0:
                result = "Error: Po odstránení neplatných hodnôt neostali žiadne dáta na porovnanie."
            else:
                # Výpočet Pearsonovho korelačného koeficientu
                correlation, p_value = pearsonr(valid_original_scores, valid_computed_scores)
                result = (f"Pearson correlation coefficient: {correlation}\n")

    except FileNotFoundError as e:
        result = f"Error: {str(e)}"
    except pd.errors.EmptyDataError:
        result = "Error: Jeden z CSV súborov je prázdny alebo neplatný."
    except ValueError as ve:
        result = f"Error pri konverzii dát: {str(ve)}"
    except Exception as e:
        result = f"Error: {str(e)}"

    # Uloženie výsledku do final.txt
    with open(final_file, 'a', encoding='utf-8') as final_out:
        final_out.write(f"Otázka: {question}\n")
        final_out.write(f"{result}")
        final_out.write("="*50 + "\n")  # Oddelenie jednotlivých záznamov

if __name__ == "__main__":
    # Získanie otázky od užívateľa
    question = input("Otázka: ")

    compute_pearson_correlation(original_scores_path, computed_scores_path, question)
    print("Finálne výsledky boli uložené do 'final.txt'.")
