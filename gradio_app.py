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

**Role:** {data['job_title']}

**Company:** {data['company']}
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

    data = r.json()["analysis"]

    fit = data["fit_analysis"]
    improvements = data.get("resume_improvements", {})
    cover = data.get("cover_letter", {})
    certs = data.get("cert_recommendations", {})

    score = fit["fit_score"]

    chart = {
        "Fit Score": score,
        "Gap": 100 - score
    }

    # ----------------------
    # Resume Improvements
    # ----------------------
    improvements_text = ""

    if improvements:

        improvements_text += "### Summary Improvements\n"
        improvements_text += "\n".join(
            [f"- {x}" for x in improvements.get("summary_improvements", [])]
        )

        improvements_text += "\n\n### Skills to Add\n"
        for s in improvements.get("skills_to_add", []):
            improvements_text += f"- **{s['skill']}** — {s['how_to_demonstrate']}\n"

        improvements_text += "\n### Keywords to Include\n"
        improvements_text += "\n".join(
            [f"- {k}" for k in improvements.get("keywords_to_include", [])]
        )

        improvements_text += "\n\n### Sections to Add\n"
        improvements_text += "\n".join(
            [f"- {s}" for s in improvements.get("sections_to_add", [])]
        )

        improvements_text += f"\n\n**Top Priority:** {improvements.get('overall_priority','')}"

    # ----------------------
    # Cover Letter
    # ----------------------
    cover_text = cover.get("cover_letter", "No cover letter generated.")

    # ----------------------
    # Certifications
    # ----------------------
    cert_text = ""

    for cert in certs.get("certifications", []):
        cert_text += f"""
**{cert['name']}**

Provider: {cert['provider']}  
Skill Addressed: {cert['addresses_skill']}  
Duration: {cert['estimated_duration']}  
Priority: {cert['priority']}

---
"""

    for course in certs.get("online_courses", []):
        cert_text += f"""
**{course['name']}**

Platform: {course['platform']}  
Skill Addressed: {course['addresses_skill']}  
Duration: {course['estimated_duration']}  
Priority: {course['priority']}

---
"""

    result_md = f"""
# Fit Score: {score}/100

## Strengths
{', '.join(fit['strengths'])}

## Weaknesses
{', '.join(fit['weaknesses'])}

---

# Resume Improvements
{improvements_text}

---

# Cover Letter
{cover_text}

---

# Recommended Certifications
{cert_text}
"""

    return result_md, chart


# -------------------------
# Profile Analyzer
# -------------------------
def analyze_profile(url):

    r = requests.post(
        f"{API}/profile/load",
        json={"linkedin_url": url}
    )

    if r.status_code != 200:
        return r.text

    data = r.json()

    return f"""
### Profile Loaded

**Name:** {data['name']}

**Headline:** {data.get('headline','')}

**Location:** {data.get('location','')}
"""


# -------------------------
# Chat with Profile
# -------------------------
def chat_profile(message, history):

    r = requests.post(
        f"{API}/ask",
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

                    upload_out = gr.Textbox(label="Status", interactive=False)

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