import gradio as gr

def predict_loan_default(income, age):
    return f"Income: {income}\nAge: {age}\n\nTo Do: Implement loan default prediction model..."

demo = gr.Interface(
    fn=predict_loan_default,
    inputs=[
        gr.Number(label="Income"), 
        gr.Number(label="Age"),
        ],
    outputs=gr.Textbox(),
    title="Loan Default Prediction",
    description="An app to predict loan default based on customer application data powered by machine learning."
    )

demo.launch()