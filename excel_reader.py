import pandas as pd

class ExcelReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sheet_names = []
    
    def get_sheet_names(self):
        try:
            excel_file = pd.ExcelFile(self.file_path)
            self.sheet_names = excel_file.sheet_names
        except Exception as e:
            print(f"Error getting sheet names: {e}")

    def read_specific_sheet(self, sheet_name):
        try:
            data = pd.read_excel(self.file_path, sheet_name=sheet_name)
            return data
        except Exception as e:
            print(f"Error reading sheet {sheet_name}: {e}")
            return None

def test_excel_reader(test_file):
    reader = ExcelReader(test_file)
    reader.get_sheet_names()
    if reader.sheet_names:
        print(f"Found sheets: {reader.sheet_names}")
        first_sheet = reader.sheet_names[1]
        sheet_data = reader.read_specific_sheet(first_sheet)
        if sheet_data is not None:
            print(f"Successfully read sheet: {first_sheet}")
        else:
            print(f"Failed to read sheet: {first_sheet}")
    else:
        print("Failed to get sheet names")

if __name__ == "__main__":
    test_file = "./DataScientist_CaseStudy_Dataset.xlsx"
    test_excel_reader(test_file)

