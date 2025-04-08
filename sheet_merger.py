import pandas as pd
import numpy as np
from excel_reader import ExcelReader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SheetMerger:
    def __init__(self, file_path):
        self.excel_reader = ExcelReader(file_path)
        self.merged_data = None

    def merge_sheets(self):
        try:
            self.excel_reader.get_sheet_names()
            self.merged_data = self.excel_reader.read_specific_sheet(self.excel_reader.sheet_names[1])
            if self.merged_data is None:
                raise ValueError(f"Could not read sheet {self.excel_reader.sheet_names[1]}")
            key_column = self.merged_data.columns[0]

            for sheet_name in self.excel_reader.sheet_names[2:]:
                current_sheet = self.excel_reader.read_specific_sheet(sheet_name)
                if current_sheet is not None:
                    self.merged_data = pd.merge(
                        self.merged_data,
                        current_sheet,
                        on=key_column,
                        how='outer'
                    )
            
            self.merged_data = self.merged_data.drop_duplicates()
            return self.merged_data

        except Exception as e:
            print(f"Error merging sheets: {e}")
            return None

    def save_merged_data(self, output_path):
        try:
            if self.merged_data is None:
                raise ValueError("No merged data available to save")
            self.merged_data.to_excel(output_path, index=False)
            print(f"Successfully saved merged data to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving merged data: {e}")
            return False

    def summary_statistics(self, file_name):
        try:
            if self.merged_data is None:
                raise ValueError("No merged data available to describe")
            description = self.merged_data.describe().T
            description.to_excel(file_name)
            print(f"Successfully saved data description to {file_name}")
            return description
        except Exception as e:
            print(f"Error getting data description: {e}")
            return None
    
    def plot_distributions(self, output_dir):
        try:
            if self.merged_data is None:
                raise ValueError("No data available to plot")
            os.makedirs(output_dir, exist_ok=True)
            plt.style.use('default')  
            sns.set_theme()
            numeric_cols = self.merged_data.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=self.merged_data, x=col)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{col}_distribution.png'))
                plt.close()

            print(f"Successfully saved distribution plots to {output_dir}")
            return True

        except Exception as e:
            print(f"Error plotting distributions: {e}")
            return False
        
    def remove_anomalies(self):
        try:
            if self.merged_data is None:
                raise ValueError("No data available to clean")
            initial_rows = len(self.merged_data)

            valid_data_mask = (
                (self.merged_data['Age'] * 12 >= self.merged_data['Tenure']) & 
                (self.merged_data['Age'] >= 18) &  
                (self.merged_data['Age'] <= 100) & 
                (self.merged_data['Tenure'] >= 0) 
            )
            self.merged_data = self.merged_data[valid_data_mask]
            
            print(f"Rows after outlier removal: {len(self.merged_data)}")
            for col in self.merged_data.columns:
                if col.startswith(('ActBal', 'Volume')):
                    self.merged_data[col] = np.log1p(self.merged_data[col])
            
            # numeric_cols = self.merged_data.select_dtypes(include=['int64', 'float64']).columns
            # for col in numeric_cols:
            #     if not self.merged_data[col].isna().all():
            #         Q1 = self.merged_data[col].quantile(0.25)
            #         Q3 = self.merged_data[col].quantile(0.75)
            #         IQR = Q3 - Q1
            #         lower_bound = Q1 - 1.5 * IQR 
            #         upper_bound = Q3 + 1.5 * IQR
            #         self.merged_data = self.merged_data[(self.merged_data[col].isna()) | 
            #                                           ((self.merged_data[col] >= lower_bound) & 
            #                                            (self.merged_data[col] <= upper_bound))]
            
            removed_rows = initial_rows - len(self.merged_data)
            print(f"\nData Cleaning Summary:")
            print(f"Initial rows: {initial_rows}")
            print(f"Rows removed: {removed_rows}")
            print(f"Remaining rows: {len(self.merged_data)}")
            
            
            return self.merged_data

        except Exception as e:
            print(f"Error cleaning data: {e}")
            return None
    
    def check_missing_consistency(self):
        try:
            if self.merged_data is None:
                raise ValueError("No data available to check")
            
            ca_inconsistent = (
                (self.merged_data['Count_CC'].isna() & self.merged_data['ActBal_CC'].notna()) |
                (self.merged_data['ActBal_CC'].isna() & self.merged_data['Count_CC'].notna())
            )
            
            inconsistent_rows = self.merged_data[ca_inconsistent]
            
            if len(inconsistent_rows) > 0:
                print("\nFound inconsistent CA records:")
                print(f"Number of inconsistent rows: {len(inconsistent_rows)}")
                print("\nSample of inconsistent records:")
                print(inconsistent_rows[['Count_CC', 'ActBal_CC']].head())
                return inconsistent_rows
            else:
                print("\nNo inconsistencies found between Count_CA and ActBal_CA")
                return None

        except Exception as e:
            print(f"Error checking missing value consistency: {e}")
            return None

    def fill_missing_actbal(self):
        try:
            if self.merged_data is None:
                raise ValueError("No data available to fill missing values")
            
            mask = (self.merged_data['ActBal_CC'].isna()) & (self.merged_data['Count_CC'] == 1)
            
            median_bal_active = self.merged_data[self.merged_data['Count_CC'] == 1]['ActBal_CC'].median()
            
            self.merged_data.loc[mask, 'ActBal_CC'] = median_bal_active
            
            filled_count = mask.sum()
            print(f"\nFilled {filled_count} missing ActBal_CC values for accounts with Count_CC = 1")
            
            return self.merged_data

        except Exception as e:
            print(f"Error filling missing ActBal values: {e}")
            return None
    def check_feature_correlations(self):
        try:
            if self.merged_data is None:
                raise ValueError("No data available to check correlations")
            
            # Specify the primary columns of interest
            primary_cols = ['Sale_CL', 'Sale_CC', 'Sale_MF', 
                           'Revenue_MF', 'Revenue_CC', 'Revenue_CL']
            
            # Get all numeric columns
            numeric_cols = self.merged_data.select_dtypes(include=['int64', 'float64']).columns
            
            # Calculate correlation matrix for all numeric columns
            full_corr_matrix = self.merged_data[numeric_cols].corr()
            
            # Plot correlation matrix for primary columns
            # plt.figure(figsize=(10, 8))
            # primary_corr = full_corr_matrix.loc[primary_cols, numeric_cols]
            # mask = np.abs(primary_corr) < 0.1
            # sns.heatmap(primary_corr, 
            #            mask=mask,
            #            annot=True, 
            #            cmap='coolwarm', 
            #            center=0, 
            #            fmt='.2f')
            # plt.title('Sales/Revenue Correlation Matrix (|correlation| > 0.1)')
            # plt.tight_layout()
            # plt.savefig('correlation_matrix_sales_revenue.png')
            # plt.close()
            
            # Find relevant correlations
            primary_correlations = []
            other_high_correlations = []
            
            # Check all numeric columns
            for i in range(len(numeric_cols)):
                for j in range(i):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    corr_value = full_corr_matrix.loc[col1, col2]
                    
                    # If either column is in primary_cols, check for >0.2 correlation
                    if col1 in primary_cols or col2 in primary_cols:
                        if abs(corr_value) > 0.05:
                            primary_correlations.append(
                                (col1, col2, corr_value)
                            )
                    # For other columns, only check for >0.9 correlation
                    elif abs(corr_value) > 0.9:
                        other_high_correlations.append(
                            (col1, col2, corr_value)
                        )
            
            # Print results
            if primary_correlations:
                print("\nRelevant correlations with Sales/Revenue features (|correlation| > 0.05):")
                for feat1, feat2, corr in primary_correlations:
                    print(f"{feat1} - {feat2}: {corr:.3f}")
            
            if other_high_correlations:
                print("\nVery high correlations between other features (|correlation| > 0.9):")
                for feat1, feat2, corr in other_high_correlations:
                    print(f"{feat1} - {feat2}: {corr:.3f}")
            
            return {
                'primary_correlations': primary_correlations,
                'other_high_correlations': other_high_correlations
            }
                
        except Exception as e:
            print(f"Error checking feature correlations: {e}")
            return None

def test_sheet_merger(file_path):
    merger = SheetMerger(file_path)
    merged_result = merger.merge_sheets()
    if merged_result is not None:
        print("Successfully merged sheets")
        print(merged_result.head())
    else:
        print("Failed to merge sheets")
    merger.plot_distributions("./before_cleaning")
    get_data_description = merger.summary_statistics("summary_before_cleaning.xlsx")
    cleaned_data = merger.remove_anomalies()
    inconsistent_data = merger.check_missing_consistency()
    high_corr_pairs = merger.check_feature_correlations()
    cleaned_data = merger.fill_missing_actbal()
    get_data_description = merger.summary_statistics("summary_after_cleaning.xlsx")
    merger.plot_distributions("./after_cleaning")
    output_file = "./merged_output.xlsx"
    if merger.save_merged_data(output_file):
        print(f"Successfully saved merged data to {output_file}")
    else:
        print("Failed to save merged data")



if __name__ == "__main__":
    test_file = "./dataset/DataScientist_CaseStudy_Dataset.xlsx"
    test_sheet_merger(test_file)
