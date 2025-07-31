import pandas as pd

class QueryEngine:
    def __init__(self, data_file):
        # Step 1: Load your CSV file into a DataFrame
        self.df = pd.read_csv(data_file, encoding="ISO-8859-1")

    def query(self, prompt):
        """
        Step 2: Match prompt to columns or data insights.
        You’ll need to write logic to analyze or summarize the data.
        """
        if 'total sales' in prompt.lower():
            total_sales = self.df['Sales'].sum()
            return f"Total Sales: ${total_sales:,.2f}"
        elif 'average profit' in prompt.lower():
            average_profit = self.df['Profit'].mean()
            return f"Average Profit: ${average_profit:,.2f}"
        elif 'state wise sales' in prompt.lower():
            state_sales = self.df.groupby('State')['Sales'].sum().sort_values(ascending=False)
            return f"State Wise Sales:\n{state_sales.to_string()}"
        elif 'customer segments' in prompt.lower():
            segments = self.df['Segment'].value_counts()
            return f"Customer Segments:\n{segments.to_string()}"
        else:
            return "Response based on CSV data"    

if __name__ == "__main__":
    # Step 3: Test with an actual CSV
    engine = QueryEngine("SampleSuperstore.csv")
    prompt = input("Enter your query: ")  # Get user input for the query
    print(engine.query(prompt))