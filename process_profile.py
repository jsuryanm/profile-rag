from src.processing.data_processing import process_profile

profile = {
    'name': 'Satya Nadella',
    'headline': 'Chairman and CEO at Microsoft',
    'location': 'Redmond, Washington, United States',
    'current_role': {'title': 'Chairman and CEO', 'company': 'Microsoft'},
    'education': [
        {'school': 'University of Chicago Booth School of Business', 'degree': 'MBA', 'years': '1994-1996'},
        {'school': 'Manipal Institute of Technology', 'degree': "Bachelor's Degree, Electrical Engineering", 'years': ''},
        {'school': 'University of Wisconsin-Milwaukee', 'degree': "Master's Degree, Computer Science", 'years': ''}
    ]
}

nodes = process_profile(profile, metadata={"source": "linkedin"})
for i, node in enumerate(nodes):
    print(f"\n--- Chunk {i+1} ---")
    print(node.text)