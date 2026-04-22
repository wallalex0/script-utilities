from pypdf import PdfReader
import pandas as pd
import os

def clean_value(field):
    if not field:
        return ""

    value = field.get("/V")

    if value is None:
        return ""

    # Convert PDF NameObject like '/Yes' → 'Yes'
    value = str(value).lstrip("/")

    # Handle checkboxes
    if value.lower() in ["off", "false", "none"]:
        return ""

    return value


def extract_forms():
    print('\nExtracting PDF forms...')
    
    input_folder = "input/"
    output_folder = "output/"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in input folder.")
        return

    data = []

    for file in pdf_files:
        print(f" Processing {file}...")
        pdf_path = os.path.join(input_folder, file)
        try:
            reader = PdfReader(pdf_path)
            fields = reader.get_fields()
            row = {"filename": file}

            if fields:
                for key, field in fields.items():
                    row[key] = clean_value(field)

            data.append(row)
        except Exception as e:
            print(f" Error processing {file}: {e}")

    if data:
        df = pd.DataFrame(data)
        output_path = os.path.join(output_folder, "pdf_forms_export.xlsx")
        try:
            df.to_excel(output_path, index=False)
            print(f"\nDone! Saved to {output_path}\n")
        except PermissionError:
            print(f"\nPermission denied: Could not save to {output_path}. Is it open?\n")
        except Exception as e:
            print(f"\nAn error occurred while saving: {e}\n")
    else:
        print("\nNo data extracted.\n")
