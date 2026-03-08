import os 
from pathlib import Path 
import logging 

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]: %(message)s")

project_name = "src"

list_of_files = [f"{project_name}/__init__.py",
                 f"{project_name}/processing/__init__.py",
                 f"{project_name}/processing/data_processing.py",
                 f"{project_name}/processing/resume_processing.py",
                 f"{project_name}/rag/__init__.py",
                 f"{project_name}/rag/query_engine.py",
                 f"{project_name}/rag/resume_index.py",
                 f"{project_name}/rag/eval.py",
                 f"{project_name}/llm/__init__.py",
                 f"{project_name}/llm/llm_interface.py",
                 f"{project_name}/agents/__init__.py",
                 f"{project_name}/agents/job_match_agent.py",
                 f"{project_name}/agents/orchestrator.py",
                 f"{project_name}/agents/recommendations_agent.py",
                 f"{project_name}/agents/retrieval_agent.py",
                 f"{project_name}/agents/supervisor_agent.py",
                 f"{project_name}/services/__init__.py",
                 f"{project_name}/services/profile_service.py",
                 f"{project_name}/services/resume_service.py",
                 f"{project_name}/schemas/__init__.py",
                 f"{project_name}/schemas/fit_analysis.py",
                 f"{project_name}/schemas/recommendations.py",
                 f"{project_name}/schemas/agent_outputs.py",
                 f"{project_name}/config/__init__.py",
                 f"{project_name}/config/settings.py",
                 f"mcp_client/__init__.py",
                 f"mcp_client/linkedin_client.py",
                 f"mcp_client/job_client.py",
                 f"api/__init__.py",
                 f"api/app.py",
                 f"api/schemas.py",
                 f"api/resume_router.py",
                 "gradio_app.py",
                 f"tests/__init__.py",
                 f"tests/test_mcp_client.py",
                 f"tests/test_mcp_tools.py",
                 f"tests/test_rag.py",
                 f"tests/test_router.py",
                 f"tests/test_process_profile.py",]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating an empty file:{file_path}")
    
    else:
        logging.info(f"{file_name} already exists")
