import pandas as pd

class DataSetSplitter:
    def __init__(self, input_file='./merged_output.xlsx', training_file='./training.xlsx', testing_file='./testing.xlsx'):
        self.input_file = input_file
        self.training_file = training_file
        self.testing_file = testing_file
        self.data = None
        self.training_data = None
        self.testing_data = None

    def load_data(self):
        try:
            self.data = pd.read_excel(self.input_file)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def split_data_set(self):
        try:
            if self.data is None:
                raise ValueError("No data loaded. Call load_data() first")
            
            last_6_cols = self.data.iloc[:, -6:]
            non_empty_mask = last_6_cols.notna().all(axis=1)
            self.training_data = self.data.iloc[non_empty_mask.values]
            
            empty_mask = last_6_cols.isna().all(axis=1)
            self.testing_data = self.data.iloc[empty_mask.values]
       
            return True
        except Exception as e:
            print(f"Error splitting training set: {e}")
            return False

    def save_data_set(self):
        try:
            if self.training_data is None:
                raise ValueError("No training data available to save")
            
            self.training_data.to_excel(self.training_file, index=False)
            print(f"Successfully saved training data to {self.training_file}")
            self.testing_data.to_excel(self.testing_file, index=False)
            print(f"Successfully saved testing data to {self.testing_file}")
            return True
        except Exception as e:
            print(f"Error saving training data: {e}")
            return False

def test_splitter():
    splitter = DataSetSplitter()
    if splitter.load_data():
        if splitter.split_data_set():
            splitter.save_data_set()
    
if __name__ == "__main__":
    test_splitter()
