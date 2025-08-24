import pandas as pd
import logging

def generate_results_matrix(results, output_dir='reports'):
    """
    Generates a results matrix in markdown and CSV formats.

    Args:
        results (list): A list of dictionaries, where each dictionary contains
                        'model', 'strategy', and performance metrics.
        output_dir (str): The directory to save the output files.
    """
    if not results:
        logging.warning("No results to generate a matrix.")
        return

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Pivot the DataFrame to create the matrix
    # We will use Total Return as the metric for the matrix for now.
    # This can be extended to other metrics as well.
    pivot_df = results_df.pivot(index='model', columns='strategy', values='Total Return')

    # Format the values as percentages
    pivot_df = pivot_df.applymap(lambda x: f"{x:.2%}" if pd.notnull(x) else 'N/A')

    # Save to markdown
    md_output_path = f"{output_dir}/results_matrix.md"
    pivot_df.to_markdown(md_output_path)
    logging.info(f"Results matrix saved to {md_output_path}")

    # Save to CSV
    csv_output_path = f"{output_dir}/results_matrix.csv"
    pivot_df.to_csv(csv_output_path)
    logging.info(f"Results matrix saved to {csv_output_path}")
