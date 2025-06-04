from flask import Flask, render_template, request, session
import grpc
import os
import llm_service_pb2
import llm_service_pb2_grpc

app = Flask(__name__)
app.secret_key = "dev"

GRPC_HOST = os.getenv("GRPC_HOST", "localhost:7860")

@app.route('/', methods=['GET', 'POST'])
def index():
    history = session.get('history', [])
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        if prompt:
            channel = grpc.insecure_channel(GRPC_HOST)
            stub = llm_service_pb2_grpc.LLMServiceStub(channel)
            req = llm_service_pb2.GenerateRequest(prompt=prompt, model_id='', temperature=0.7, max_new_tokens=64)
            resp = stub.GenerateText(req)
            history.append({'prompt': prompt, 'response': resp.generated_text})
            session['history'] = history
    return render_template('index.html', history=history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
