import gradio as gr
import requests

API = "http://127.0.0.1:8000"


# -------------------------
# Resume Upload
# -------------------------
def upload_resume(file, name):

    files = {"file": open(file, "rb")}
    params = {"candidate_name": name}

    r = requests.post(f"{API}/resume/load", files=files, params=params)

    if r.status_code != 200:
        return f"Error: {r.text}"

    return f"Resume loaded for {name}"


# -------------------------
# Load Job
# -------------------------
def load_job(url):

    r = requests.post(
        f"{API}/resume/load-job",
        json={"job_url": url}
    )

    if r.status_code != 200:
        return r.text

    data = r.json()

    return f"""
### Job Loaded

Role: **{data['job_title']}**

Company: **{data['company']}**
"""


# -------------------------
# Run Resume Analysis
# -------------------------
def run_analysis():

    r = requests.post(
        f"{API}/resume/analyze",
        json={"quick": False}
    )

    if r.status_code != 200:
        return r.text, None

    data = r.json()

    fit = data["analysis"]["fit_analysis"]

    score = fit["fit_score"]

    chart = {
        "Fit Score": score,
        "Gap": 100 - score
    }

    result_md = f"""
## Fit Score: {score}/100

### Strengths
{', '.join(fit['strengths'])}

### Weaknesses
{', '.join(fit['weaknesses'])}
"""

    return result_md, chart


# -------------------------
# Profile Analyzer
# -------------------------
def analyze_profile(url):

    r = requests.post(
        f"{API}/profile/load",
        json={"profile_url": url}
    )

    if r.status_code != 200:
        return r.text

    data = r.json()

    return f"""
### Profile Loaded

Name: **{data['full_name']}**

Skills:
{', '.join(data.get('skills', []))}
"""


# -------------------------
# Chat with Profile
# -------------------------
def chat_profile(message, history):

    r = requests.post(
        f"{API}/profile/ask",
        json={"question": message}
    )

    if r.status_code != 200:
        answer = r.text
    else:
        answer = r.json()["answer"]

    history.append((message, answer))

    return history, ""


# -------------------------
# UI
# -------------------------
with gr.Blocks(title="Profile RAG") as app:

    gr.Markdown("# Profile RAG Dashboard")

    with gr.Tabs():

        # ----------------------------------
        # Resume Analyzer
        # ----------------------------------
        with gr.Tab("Resume Analyzer"):

            with gr.Row():

                with gr.Column():

                    gr.Markdown("### Upload Resume")

                    resume = gr.File()
                    name = gr.Textbox(label="Candidate Name")

                    upload_btn = gr.Button("Upload Resume")

                    upload_out = gr.Textbox()

                    upload_btn.click(
                        upload_resume,
                        inputs=[resume, name],
                        outputs=upload_out
                    )

                with gr.Column():

                    gr.Markdown("### Load Job")

                    job_url = gr.Textbox(label="LinkedIn Job URL")

                    job_btn = gr.Button("Load Job")

                    job_out = gr.Markdown()

                    job_btn.click(load_job, job_url, job_out)

            gr.Markdown("### Run Analysis")

            analyze_btn = gr.Button("Analyze Resume")

            analysis_md = gr.Markdown()

            score_chart = gr.BarPlot(
                x="label",
                y="value",
                title="Fit Score",
            )

            analyze_btn.click(
                run_analysis,
                outputs=[analysis_md, score_chart]
            )

        # ----------------------------------
        # Profile Analyzer
        # ----------------------------------
        with gr.Tab("Profile Analyzer"):

            profile_url = gr.Textbox(label="LinkedIn Profile URL")

            load_profile = gr.Button("Load Profile")

            profile_info = gr.Markdown()

            load_profile.click(
                analyze_profile,
                profile_url,
                profile_info
            )

            gr.Markdown("### Chat with Profile")

            chatbot = gr.Chatbot(height=400)

            msg = gr.Textbox()

            msg.submit(
                chat_profile,
                [msg, chatbot],
                [chatbot, msg]
            )


app.launch(share=True)