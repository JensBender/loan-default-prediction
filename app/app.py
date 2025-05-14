import gradio as gr

professions = ["profession_01", "profession_02", "profession_03", "profession_04", "profession_05"]


def predict_loan_default(income, age, experience, profession):
    return f"""
    Income: {income}
    Age: {age}
    Experience: {experience}
    Profession: {profession}
    
    To Do: Implement loan default prediction model...
    """


demo = gr.Interface(
    fn=predict_loan_default,
    inputs=[
        gr.Number(label="Income"), 
        gr.Number(label="Age"),
        gr.Number(label="Experience"),
        gr.Dropdown(label="Profession", choices=professions)
        ],
    outputs=gr.Textbox(),
    title="Loan Default Prediction",
    description="An app to predict loan default based on customer application data powered by machine learning."
    )

demo.launch()