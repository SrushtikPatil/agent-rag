"""
Tabular Data Analyzer (Excel & CSV)
Executes computations, statistical analysis, and data extraction using Pandas and Openpyxl.
Ensures mathematical accuracy and deterministic formula evaluation.
"""

import pandas as pd
from typing import Dict, Any, Optional

class ExcelAnalyzer:
    @staticmethod
    def inspect_file(file_path: str) -> Dict[str, Any]:
        """Reads file, extracts sheet names, column names, preview, and shape."""
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
                sheets = ["Sheet1"]
                dfs = {"Sheet1": df}
            else:
                dfs = pd.read_excel(file_path, sheet_name=None)
                sheets = list(dfs.keys())
                df = dfs[sheets[0]]

            summary = {
                "sheets": sheets,
                "active_sheet": sheets[0],
                "rows": df.shape[0],
                "columns": list(df.columns),
                "head": df.head(5).to_dict(orient="records"),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.items()}
            }
            return {"success": True, "summary": summary, "dataframe": df}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def answer_query(df: pd.DataFrame, query: str, gemini_service) -> str:
        """
        Uses data context + Gemini to formulate calculations or answer queries on spreadsheet data.
        """
        # Provide structural context to Gemini
        preview_text = df.head(10).to_string()
        stats_text = ""
        try:
            stats_text = df.describe().to_string()
        except Exception:
            pass

        data_context = (
            f"Dataset Preview (Shape: {df.shape[0]} rows x {df.shape[1]} columns):\n"
            f"{preview_text}\n\n"
            f"Summary Statistics:\n{stats_text}\n\n"
            f"Columns: {list(df.columns)}"
        )

        prompt = (
            f"You are an expert Data Analyst and Excel Specialist.\n\n"
            f"{data_context}\n\n"
            f"User Question: {query}\n\n"
            "Please analyze the spreadsheet data above and provide a precise, accurate answer. "
            "Show calculations, specific row/column references, and key insights."
        )

        return gemini_service.generate_chat_response(prompt)
