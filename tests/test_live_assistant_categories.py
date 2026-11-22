import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.agentic_router import AgenticAssistant

asst = AgenticAssistant()

def ask(q):
    s = {
        'user_id': 'usr_default', 'user_name': 'Srushti', 'query': q,
        'attached_file_path': None, 'attached_file_type': None,
        'attached_link': None, 'active_dataframe': None, 'route': None,
        'response': None, 'mode': 'auto', 'active_doc_id': None
    }
    res = asst.process_turn(s)
    print(f"=== Query: {q} ===")
    print(f"Route: {res.get('route')}")
    ans = res.get('response', '')
    print(f"Response (first 200 chars):\n{ans[:200]}")
    print(f"Has random Shamani citation: {'Raj Shamani' in ans}\n")

ask('Explain quantum computing simply')
ask('Write a python function to check if a number is prime')
ask('What is the CIA triad in database security?')
